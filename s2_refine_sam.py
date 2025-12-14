from __future__ import division
from __future__ import print_function




import os
import os.path as osp
from tqdm import tqdm
import pdb
import random
import importlib
import argparse
import logging
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
import cv2
import pickle
from PIL import Image
import time

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything.predictor import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import build_all_layer_point_grids, batch_iterator

import torch.nn.functional as F
import copy
import glob
import pdb
import tqdm
from pathlib import Path
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def get_hwcord(mask_path):

    mask = Image.open(mask_path)
    mask = copy.deepcopy(mask)
    mask_array = np.asarray(mask)

    h = mask_array.shape[0] 
    w = mask_array.shape[1]

    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    hmin, hmax = np.where(rows)[0][[0, -1]]
    wmin, wmax = np.where(cols)[0][[0, -1]]

    return hmin, hmax, wmin, wmax


def get_refine_pselabel_from_sam_with_cam_box_mask_png(img_path, mask_path, save_path, domain):
    

    savemask_path = create_directory(save_path)

    sam_checkpoint = "/media/ders/sdd1/XS/pipeline/weights/SAM/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    img_list_path = domain
    img_gt_name_list = open(img_list_path).read().splitlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    error_log_file = '/media/ders/sda1/XS/SPCAM_FAMS/error_images_BUSI.txt'

    with open(error_log_file, 'w') as f:
        f.write('Error images:\n')

    for name in tqdm.tqdm(img_name_list):
        try:
            image = cv2.imread(img_path + '/%s.png' % name)
            mask_path_ = mask_path + '/%s.png' % name
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            hmin, hmax, wmin, wmax = get_hwcord(mask_path_)
           
            input_box = np.array([wmin, hmin, wmax, hmax])   

            masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False,)
            
            temp = np.full((masks.shape[1], masks.shape[2]), -1, dtype=int)
            temp[masks[0]] = 255

            cv2.imwrite(savemask_path + '/%s.png' % name, temp) 
        
        except Exception as e:
            with open(error_log_file, 'a') as f:
                f.write(name + '.png\n')
            print(f"Error processing image {name}: {e}")

            src = mask_path_ 
            dst = savemask_path + '/%s.png' % name
            shutil.copy(src, dst)

            print(f"An error occurred while processing {name}: {e}")
            continue  
 

def get_refine_pselabel_from_sam_with_dino_box_mask_png(pselabelpath, savepath, domain, sam_mask_path):

    dinobox_labelsPath = sam_mask_path

    if not os.path.isdir(savepath):
        os.makedirs(savepath)
        
    imgIDALL = os.listdir(dinobox_labelsPath)
    data= [image_id.strip() for image_id in open(domain).readlines()]
    for imgID in data:
        imgID = imgID + '.png'
        if imgID in imgIDALL:
            src = dinobox_labelsPath + imgID 
            dst = savepath + imgID
            # shutil.copy(src, dst)
        else:
            src = pselabelpath + imgID 
            dst = savepath + imgID
            # shutil.copy(src, dst)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} does not exist, skipping.")


if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    img_path = '../img/'
    sam_mask_path = '../filtered_sam_mask/'   
    pselabel_path = '../../'
    domain = '../fold_1_train.txt'
    save_refine_pselabel_path = '../../xx_sam/'
    
    # online
    get_refine_pselabel_from_sam_with_cam_box_mask_png(img_path, pselabel_path, save_refine_pselabel_path, domain)
    print(domain+'done')

    # offline
    get_refine_pselabel_from_sam_with_dino_box_mask_png(pselabel_path, save_refine_pselabel_path, domain, sam_mask_path)
    print(domain+'done')
    




    

