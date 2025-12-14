# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

"""
xs实验代码，生成分割结果

"""


import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
import dataset_root

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def get_argparser():

    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--mode', default='fix', type=str)
    parser.add_argument('--use_gn', default=True, type=str2bool)
    parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)

    # Inference parameters
    parser.add_argument('--dataset', default='PubDB', type=str, choices=['PubBUSI', 'PriBUTS', 'PubDB'])
    parser.add_argument('--architecture', default='DeepLabv2', type=str)
    parser.add_argument('--backbone', default='resnet101', type=str)
    parser.add_argument('--curtime', default='', type=str)
    parser.add_argument('--expName', default='../../', type=str)
    parser.add_argument('--compName', default='V2R101_CE', type=str)   # V3pR101_CEMS, V3pR101_CE, V2R101_CE
    parser.add_argument('--model_path', default='../../xxx.pth', type=str)
    parser.add_argument('--domain', default='fold_1_test', type=str)
    parser.add_argument('--iteration', default=0, type=int)
    parser.add_argument('--gpu', default='4', type=str)
    return parser

def inference(model, images, image_size):

    images = images.cuda()
    logits = model(images)
    logits = resize_for_tensors(logits, image_size)
    logits = logits[0] + logits[1].flip(-1)
    logits = get_numpy_from_tensor(logits).transpose((1, 2, 0))
    return logits


def main():

    args = get_argparser().parse_args()
   
    time_string = time.strftime("%Y_%m_%d_%H_%M_%S")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    pred_dir = create_directory(f'./experiments/predictions/{args.expName}/{args.compName}/')
    pred_path = create_directory(pred_dir + f'{time_string}/')

    if args.iteration > 0:
        savepng_path = create_directory(pred_path + 'segresult_dCRF/')
    else:
        savepng_path = create_directory(pred_path + 'segresult/')
    
    set_seed(args.seed)
    log_func = lambda string='': print(string)

    # Transform, Dataset, DataLoader
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    # for mIoU
    data_dir = dataset_root.PubBUSI_ROOT if args.dataset == 'PubBUSI' else dataset_root.PubDB_ROOT
    valid_dataset = Dataset_For_Evaluation_PUBBUSI(data_dir, args.domain, _dataset=args.dataset)

    # Network
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=3 if args.dataset == 'PubBUSI' else 3, use_group_norm=args.use_gn)
    elif args.architecture == 'DeepLabv2':
        model = DeepLabv2(args.backbone, num_classes=3 if args.dataset == 'PubBUSI' else 3, use_group_norm=args.use_gn)
    elif args.architecture == 'Unet':
        model = DeepLabv3_Plus(args.backbone, num_classes=3 if args.dataset == 'PubBUSI' else 3)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()
    log_func('[i] {}'.format(args.expName))
    log_func('[i] {}'.format(time_string))
    log_func('[i] {}'.format(args.iteration))
    log_func('[i] {}'.format(args.domain))
    log_func('[i] {}'.format(args.model_path))

    load_model(model, args.model_path, parallel=False)

    # Evaluation
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    with torch.no_grad():
        length = len(valid_dataset)
        for step, (ori_image, image_id, _, gt_mask) in enumerate(valid_dataset):
            ori_w, ori_h = ori_image.size

            cams_list = []

            for scale in scales:
                image = copy.deepcopy(ori_image)
                image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
                
                image = normalize_fn(image)
                image = image.transpose((2, 0, 1))

                image = torch.from_numpy(image)
                flipped_image = image.flip(-1)
                
                images = torch.stack([image, flipped_image])

                cams = inference(model, images, (ori_h, ori_w))
                cams_list.append(cams)
            
            preds = np.sum(cams_list, axis=0)
            preds = F.softmax(torch.from_numpy(preds), dim=-1).numpy()
            
            if args.iteration > 0:
                # h, w, c -> c, h, w
                num_classes=3 if args.dataset == 'PubBUSI' else 3
                preds = crf_inference(np.asarray(ori_image), preds.transpose((2, 0, 1)), t=args.iteration, labels=num_classes)
                pred_mask = np.argmax(preds, axis=0)
            else:
                pred_mask = np.argmax(preds, axis=-1)
       
            imageio.imwrite(savepng_path + image_id + '.png', pred_mask.astype(np.uint8))
            
            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
            sys.stdout.flush()
        print()
    
    if args.domain == 'val':
        print("python3 evaluate.py --experiment_name {} --domain {} --mode png".format(args.tag, args.domain))

    return

if __name__ == '__main__':

    main()



