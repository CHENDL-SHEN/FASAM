# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *
from core.loss import *

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

import os


def get_params():

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--backbone', default='resnet101', type=str)
    parser.add_argument('--use_gn', default=True, type=str2bool)
    parser.add_argument('--batch_size', default=4, type=int)  
    parser.add_argument('--max_epoch', default=200, type=int)  
    parser.add_argument('--lr', default=0.007, type=float)
    parser.add_argument('--wd', default=4e-5, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--min_image_size', default=256, type=int)
    parser.add_argument('--max_image_size', default=1024, type=int)
    parser.add_argument('--print_ratio', default=0.1, type=float)
    parser.add_argument('--curtime', default='00', type=str)

    # Training parameters
    parser.add_argument('--data_dir', default='../../', type=str)
    parser.add_argument('--dataset', default='PubDB', type=str, choices=['PubBUSI', 'PubDB'])
    parser.add_argument('--expName', default='SEG_ABLA_F/SEG_train1_PubBUSI', type=str) 
    parser.add_argument('--domain', default='fold_1_train', type=str)
    parser.add_argument('--compName', default='V2R101_CE', type=str)  
    parser.add_argument('--architecture', default='DeepLabv2', type=str)
    parser.add_argument('--pselabel_dir', default='../../', type=str)
    parser.add_argument('--gpu', default='2', type=str)
    
    args, _ = parser.parse_known_args()
    return args

def get_dataset(args, pselabel_dir, domain):

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])

    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])

    data_dir = dataset_root.PubBUSI_ROOT if args.dataset == 'PubBUSI' else dataset_root.PriBUTS_ROOT
    train_dataset = Dataset_with_SEG_FULL_PUBBUSI(data_dir, pselabel_dir, domain, train_transform, _dataset=args.dataset)  # default train_aug
    valid_dataset = Dataset_For_Evaluation_PUBBUSI(data_dir, domain[:7] + 'test', test_transform, _dataset=args.dataset)

    return train_dataset, valid_dataset, train_transform

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    time_string = time.strftime("%Y_%m_%d_%H_%M_%S")
    print(args.expName)

    # Arguments
    log_expName = create_directory(f'./experiments/logs/{args.expName}/')
    data_expName = create_directory(f'./experiments/data/{args.expName}/')
    model_expName = create_directory(f'./experiments/models/{args.expName}/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.expName}/{args.curtime}/')   

    log_path = log_expName + f'/%s_{time_string}.txt' % args.compName
    data_path = data_expName + f'/{time_string}.json'
    model_path = model_expName + f'/%s_{time_string}.pth' % args.compName

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(args.expName))
    log_func('[i] {}'.format(time_string))
    log_func('[i] {}'.format(args.domain))

    log_func(str(args))
    log_func()
    pselabel_dir = args.pselabel_dir
    # Dataset, DataLoader
    train_dataset, valid_dataset, train_transform = get_dataset(args, pselabel_dir, args.domain)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=True)

    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    # Network
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=3 if args.dataset == 'PubBUSI' else 3, use_group_norm=args.use_gn)
    elif args.architecture == 'DeepLabv2':
        model = DeepLabv2(args.backbone, num_classes=3 if args.dataset == 'PubBUSI' else 3, use_group_norm=args.use_gn)
    elif args.architecture == 'Unet':
        model = DeepLabv3_Plus(args.backbone, num_classes=3 if args.dataset == 'PubBUSI' else 3, use_group_norm=args.use_gn)

    param_groups = model.get_parameter_groups(None)
    params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ]

    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))

    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn_for_backup = lambda: save_model(model, model_path.replace('.pth', f'_backup.pth'),parallel=the_number_of_gpu > 1)

    # Loss, Optimizer
    class_loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()
    l_ms = MS_Loss()

    optimizer = PolyOptimizer(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration,nesterov=args.nesterov)

    # Train
    data_dic = {
        'train': [],
        'validation': [],
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss'])

    best_valid_mIoU = -1

    def evaluate(loader, args):
        model.eval()
        eval_timer.tik()

        class_num = 3 if args.dataset == 'PubBUSI' else 4
        meter = Calculator_For_mIoU(class_num)

        with torch.no_grad():
            length = len(loader)
            for step, (images, _, _, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()

                logits = model(images)
                predictions = torch.argmax(logits, dim=1)

                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    meter.add(pred_mask, gt_mask)

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()

        print(' ')
        model.train()

        return meter.get(clear=True)

    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    for iteration in range(max_iteration):
        images, _, _, labels = train_iterator.get()
        images, labels = images.cuda(), labels.cuda()

        # Inference
        logits = model(images)

        # The part is to calculate losses.
        if 'Seg' in args.architecture:
            labels = resize_for_tensors(labels.type(torch.FloatTensor).unsqueeze(1), logits.size()[2:], 'nearest',
                                        None)[:, 0, :, :]
            labels = labels.type(torch.LongTensor).cuda()

       
        loss = class_loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss': loss.item(),
        })

        if args.dataset != 'PubDB':
            # For Log
            if (iteration + 1) % log_iteration == 0:
                loss = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                data = {
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'time': train_timer.tok(clear=True),
                }
                data_dic['train'].append(data)
                write_json(data_path, data_dic)

                log_func('[i] \
                        iteration={iteration:,}, \
                        learning_rate={learning_rate:.4f}, \
                        loss={loss:.4f}, \
                        time={time:.0f}sec'.format(**data)
                        )

                writer.add_scalar('Train/loss', loss, iteration)
                writer.add_scalar('Train/learning_rate', learning_rate, iteration)

        # Evaluation
        if (iteration + 1) % val_iteration == 0:
            mIoU, _ = evaluate(valid_loader, args)

            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration': iteration + 1,
                'mIoU': mIoU,
                'best_valid_mIoU': best_valid_mIoU,
                'time': eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)

            log_func('[i] \
                    iteration={iteration:,}, \
                    mIoU={mIoU:.2f}%, \
                    best_valid_mIoU={best_valid_mIoU:.2f}%, \
                    time={time:.0f}sec'.format(**data)
                     )

            writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)

    write_json(data_path, data_dic)
    writer.close()

    return


if __name__ == '__main__':

    args =get_params()
    main(args)

    print("train finished")




