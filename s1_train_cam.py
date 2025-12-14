
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from cv2 import LMEDS, Tonemap, log, polarToCart
import numpy as np
import datetime

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from imageio import imsave
from core.networks import *
from core.loss import *

from core.datasets import *
from tools.general.Q_util import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

import bus_s1_infer_cam

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from datetime import datetime
import core.models as fcnmodel
import dataset_root

# import nni
# from nni.utils import merge_parameter
# from nni.experiment import Experiment

def get_params():
    parser = argparse.ArgumentParser()
    ###############################################################################
    # Dataset
    ###############################################################################
    parser.add_argument('--num_workers', default=8, type=int) 
    parser.add_argument('--image_size', default=480, type=int)
    parser.add_argument('--min_image_size', default=320, type=int)
    parser.add_argument('--max_image_size', default=640, type=int)
    parser.add_argument('--seed', default=0, type=int)
    
    ###############################################################################
    # Network
    ###############################################################################
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--batch_size', default=16, type=int)  
    parser.add_argument('--max_epoch', default=200, type=int) 
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', default=4e-5, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)
    ###############################################################################
    # others
    ###############################################################################
    parser.add_argument('--curtime', default='00', type=str)
    parser.add_argument('--print_ratio', default=0.1, type=float)
    parser.add_argument('--clamp_rate', default=0.001, type=float)
    parser.add_argument('--ig_th', default=0.1, type=float)
    ###############################################################################
    # CAM network & loss
    ###############################################################################
    # fixed para
    parser.add_argument('--afflossPara', default=0.3, type=float)
     
    # training para
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--dataset', default='PubDB', type=str, choices=['PubBUSI', 'PriBUTS', 'PubDB'])
    parser.add_argument('--domain', default='fold_1_train', type=str)
    parser.add_argument('--expName', default='CAM_SOTA/CAM/DB/CAM_fold1_PubDB', type=str)  
    parser.add_argument('--compName', default='CAM', type=str)  
    
    # anchor para
    parser.add_argument('--dist_hw', default=10, type=int)   
    parser.add_argument('--dist_the', default=10, type=int)        
    parser.add_argument('--patch_number', default=9, type=int)     
    
    args, _ = parser.parse_known_args()
    return args

def main(args):
    set_seed(args.seed)

    ###################################################################################
    # Arguments
    ###################################################################################
    time_string = time.strftime("%Y_%m_%d_%H_%M_%S")

    print(args.expName)
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.expName}/')   
    log_expName = create_directory(f'./experiments/logs/{args.expName}/')
    model_expName = create_directory(f'./experiments/models/{args.expName}/')
    
    if args.stopepoch != 0:
        log_path = log_expName + f'/{args.compName}_{time_string}_stopepoch@{args.stopepoch}_constvalue@{args.constant_value}.txt' 
        model_path = model_expName + f'/{args.compName}_{time_string}_stopepoch@{args.stopepoch}_constvalue@{args.constant_value}.pth' 
    else:
        constant_value = 0
        log_path = log_expName + f'/{args.compName}_{time_string}_stopepoch@{args.stopepoch}_constvalue@{constant_value}.txt' 
        model_path = model_expName + f'/{args.compName}_{time_string}_stopepoch@{args.stopepoch}_constvalue@{constant_value}.pth' 
    
    log_func = lambda string='': log_print(string, log_path)

    log_func('afflossPara: {}'.format(args.afflossPara))
    log_func('dist_the: {}'.format(args.dist_the))
    log_func('patch_number: {}'.format(args.patch_number))
    log_func('dist_hw: {}'.format(args.dist_hw))

    log_func('[i] {}'.format(args.expName))
    log_func(str(args))
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = [
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),
    ]
    
    train_transform = transforms.Compose(train_transforms + [Transpose_For_Segmentation()])

    domain=args.domain
        
    data_dir = dataset_root.PubBUSI_ROOT if args.dataset == 'PubBUSI' else dataset_root.PubDB_ROOT
    saliency_dir = dataset_root.PubBUSI_SAL_ROOT if args.dataset == 'PubBUSI' else dataset_root.PubDB_SAL_ROOT
    train_dataset = Dataset_with_CAM(data_dir, saliency_dir, domain, train_transform, _dataset=args.dataset)  
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] train_transform is {}'.format(train_transform))
    
    val_iteration = int(len(train_loader))
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration
    
    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    ###################################################################################
    # Network
    ###################################################################################
    model = CLSNet(args.backbone, num_classes=3 if args.dataset == 'PubBUSI' else 3)
    
    model = model.cuda()
    model.train()
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    log_func(args.gpu)
    the_number_of_gpu = len(args.gpu.split(','))

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    val_domain = args.domain
    evaluatorA = bus_s1_infer_cam.evaluator(args.dataset, domain=val_domain, refine_list=[0])

    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    param_groups = model.get_parameter_groups()
    params = [
        {'params': param_groups[0], 'lr': 1*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
        ]
    optimizer = PolyOptimizer(params, lr=args.lr, momentum=0.5, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    lossfn_pcms = Pro_Contrast_MS_Loss_GC(args=args)
    lossfn_pcms = torch.nn.DataParallel(lossfn_pcms).cuda()

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : [],
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'cls_loss', 'gc_loss'])

    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    best_valid_mIoU = -1
    for iteration in range(max_iteration):
        images, imgids, labels, sailencys = train_iterator.get()     
        images = images.cuda()
        labels = labels.cuda()
        prob = None
        b, c, h, w =images.shape
 
        logits, logitsmin, _ = model(images, pcm=0)

        b, c, h, w = logits.shape
        imgmin_mask = F.interpolate(images.float(), size=(h, w))               
        imgmin_mask = imgmin_mask.float().sum(dim=1, keepdim=True) != 0          
        
        sailencys = sailencys.cuda().unsqueeze(1)
        sam_mask = F.interpolate(sailencys.float(), size=(h, w), mode='bilinear', align_corners=False)
        sam_mask = torch.round(sam_mask).to(torch.uint8)

        tagpred = F.adaptive_avg_pool2d(logits, 1)          
        tagpred = logitsmin
        cls_loss = F.multilabel_soft_margin_loss(tagpred[:, 1:].view(tagpred.size(0), -1), labels[:, 1:])
        mask = labels[:, :].unsqueeze(2).unsqueeze(3).cuda() 
        
        fg_cam = make_cam(logits[:, 1:]) * mask[:, 1:]    
        fg_cam_ms = fg_cam.clone().detach()
        fg_cam_ms = F.softmax(fg_cam_ms, dim=1)

        target_feat = F.interpolate(images.float(), size=(h, w), mode='bilinear', align_corners=False )

        target_feat = target_feat.detach() * imgmin_mask   
        target_feat_title = tile_features(target_feat, args.patch_number)
        fg_cam_title = tile_features(fg_cam, args.patch_number)

        gc_loss = lossfn_pcms(fg_cam_title, target_feat_title).mean() * args.afflossPara     
        
        loss = cls_loss + gc_loss

        #################################################################################################
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        #################################################################################################
        # For Log
        #################################################################################################
        if args.dataset != 'PubDB':
            if (iteration + 1) % log_iteration == 0:
                train_meter.add({'loss': loss.item(), 'gc_loss': gc_loss.item(), 'cls_loss': cls_loss.item()})
                loss, cls_loss, gc_loss = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
                data = {
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'cls_loss': cls_loss,
                    'gc_loss': gc_loss,
                    'time': train_timer.tok(clear=True),
                }
                data_dic['train'].append(data)
                
                log_func('[i] \
                    iteration={iteration:,}, \
                    learning_rate={learning_rate:.4f}, \
                    loss={loss:.4f}, \
                    cls_loss={cls_loss:.4f}, \
                    gc_loss={gc_loss:.4f}, \
                    time={time:.0f}sec'.format(**data)
                )

        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            mIoU, para = evaluatorA.evaluate(model, ite=args.th, dCRF_iter=0)[0]
            
            if(mIoU < 35):
                log_func('miou is too low'+str(mIoU))
                # break

            refine_num, threshold = para
            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU
                if(mIoU > 22):
                    save_model_fn()
                    log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'threshold' : threshold,
                'refine_num' : refine_num,
                'mIoU' : mIoU,
                'best_valid_mIoU' : best_valid_mIoU,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
    
            log_func('[i] \
                iteration={iteration:,}, \
                mIoU={mIoU:.2f}%, \
                best_valid_mIoU={best_valid_mIoU:.2f}%, \
                threshold={threshold:.2f}%,\
                refine_num={refine_num:.0f},\
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Evaluation/threshold', threshold, iteration)
            writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)
    
    writer.close()

if __name__ == '__main__':
        args =get_params()
        print(args.domain)
        print(str(args))
        main(args)


       


        