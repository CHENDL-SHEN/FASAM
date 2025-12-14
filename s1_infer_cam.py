from operator import mod
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
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *
from tools.dataset.voc_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from tools.general.visualization import *
from datetime import datetime
from core.spnetwork_new import SANET_Model_new_base
import core.models as fcnmodel

import dataset_root

parser = argparse.ArgumentParser()


def get_params():

    parser.add_argument('--sp_cam', default=True, type=str2bool)
    parser.add_argument('--curtime', default='', type=str)
    
    parser.add_argument('--dataset', default='PubBUSI', type=str, choices=['PubBUSI', 'PriBUTS', 'PubDB'])
    parser.add_argument('--domain', default='fold_5_train', type=str)
    parser.add_argument('--Cmodel_path', default='/media/ders/sda1/XS/SPCAM_FAMS/experiments/models/CAM_SOTA/CAM/DB/CAM_fold5_PubDB/CAM_2025_06_23_19_44_27_stopepoch@0_constvalue@0.pth', type=str)  # 46.3 
    
    parser.add_argument('--tag', default='CAM_ABLA_COMP/BUSI/CAM_train5_PubBUSI', type=str)  # CAM_eval0_PubBUSI  CAM_ABLA_PARA/Const_Value  CAM_ABLA_COMP/CAM_train1_PubBUSI
    parser.add_argument('--savepng', default=True, type=str2bool)
    parser.add_argument('--savenpy', default=True, type=str2bool)
    
    parser.add_argument('--gpu', default='2', type=str)
    args = parser.parse_args()

    return args


class evaluator:
    def __init__(self, dataset='PubBUSI', domain='_', save_np_path=None, savepng_path=None,
                 muti_scale=False, th_list=list(np.arange(0.05, 0.4, 0.05)), refine_list=range(0, 50, 10)) -> None:
        self.C_model = None
    
        if (muti_scale):
            self.scale_list = [0.5, 1.0, 1.5, 2.0, -0.5, -1, -1.5, -2.0] 

        else:
            self.scale_list = [1] 

        self.th_list = th_list             
        self.refine_list = refine_list    
        self.parms = []

        for renum in self.refine_list:
            for th in self.th_list:
                self.parms.append((renum, th))  

        class_num = 3 if dataset == 'PubBUSI' else 3

        self.meterlist = [Calculator_For_mIoU(class_num) for x in self.parms]  
        self.save_png_path = savepng_path
        self.save_np_path = save_np_path

        if (self.save_png_path != None):
            if not os.path.exists(self.save_png_path):
                os.mkdir(self.save_png_path)

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose([
            Normalize_For_Segmentation(imagenet_mean, imagenet_std),
            Transpose_For_Segmentation()
        ])

        if (dataset == 'PubBUSI'):
            valid_dataset = Dataset_For_Evaluation_PUBBUSI(dataset_root.PubBUSI_ROOT, domain, test_transform, dataset)
        else:
            valid_dataset = Dataset_For_Evaluation_PUBBUSI(dataset_root.PriBUTS_ROOT, domain, test_transform, 'PubDB')
        
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=True)

    def get_cam(self, images, ids, Qs):    
        with torch.no_grad():
            cam_list = []
            _, _, h, w = images.shape   
                                                   
            for s, q in zip(self.scale_list, Qs):  
                target_size = (round(h * abs(s)), round(w * abs(s)))  
                scaled_images = F.interpolate(images, target_size, mode='bilinear', align_corners=False)     
                H_, W_ = int(np.ceil(target_size[0] / 16.) * 16), int(np.ceil(target_size[1] / 16.) * 16)      
                scaled_images = F.interpolate(scaled_images, (H_, W_), mode='bilinear', align_corners=False)   

                if (s < 0):
                    scaled_images = torch.flip(scaled_images, dims=[3])  
                logits, x4, _ = self.C_model(scaled_images)
                cam_list.append(logits)

        return cam_list

    def get_mutiscale_cam(self, cam_list, Q_list, affmat_list, refine_time=0):     
        _, _, h, w = cam_list[-1].shape    
        h *= 16
        w *= 16    
                                                                                             
        refine_cam_list = []                                                                 
        for cam, Q, affmat, s in zip(cam_list, Q_list, affmat_list, self.scale_list):                                      
            cam = F.interpolate(cam, (int(h), int(w)), mode='bilinear', align_corners=False) 
            if(s < 0):
                cam = torch.flip(cam, dims=[3])  
            refine_cam_list.append(cam)
        refine_cam = torch.sum(torch.stack(refine_cam_list), dim=0)                       

        return refine_cam                                                                 

    def getbest_miou(self, clear=True):
        iou_list = []
        for parm, meter in zip(self.parms, self.meterlist):
            cur_iou, mIoU_foreground, IoU_list, FP, FN = meter.get(clear=clear, detail=True)
            iou_list.append((cur_iou, parm))
        iou_list.sort(key=lambda x: x[0], reverse=True)
        return iou_list

    def evaluate(self, C_model):
        self.C_model= C_model
        self.C_model.eval()

        with torch.no_grad():
            length = len(self.valid_loader)    

            for step, (images, image_ids, tags, gt_masks) in enumerate(self.valid_loader): 
                
                images = images.cuda()      
                gt_masks = gt_masks.cuda()  
                _, _, h, w = images.shape   
            
                Qs = [images for x in range(len(self.scale_list))]
                affmats = [None for x in range(len(self.scale_list))]

                cams_list = self.get_cam(images, image_ids, Qs, ite)  
                mask = tags.unsqueeze(2).unsqueeze(3).cuda()  

                for renum in self.refine_list:  
                    refine_cams = self.get_mutiscale_cam(cams_list, Qs, affmats, renum)  
                    cams = (make_cam(refine_cams) * mask)     
                 
                    cams = F.interpolate(cams, (int(h), int(w)), mode='bilinear', align_corners=False)  
                                                                                               
                    if (self.save_np_path != None):    
                        
                        cams2 = F.interpolate(cams, (int(h), int(w)), mode='bilinear', align_corners=False)
                        np.save(os.path.join(self.save_np_path, image_ids[0] + '.npy'), cams2.cpu().numpy()) 
                        
                        img_8 = convert_to_tf(images[0])
                        
                        cams[0, 0] = cams[0, 1:].max(0, True)[0]   
                        nnn = cams[0, 1:].max(0, True)[0]
                        cams[0, 0] = nnn

                        ttt = cams.argmax(dim=1)[0]    
                        ttt = get_colored_mask(ttt.cpu().numpy())
                        ttt = cv2.cvtColor(ttt, cv2.COLOR_RGB2HLS)  
                        ttt[:, :, 1] = (230 * nnn[0].cpu().numpy()).astype(np.uint8)
                        ttt = cv2.cvtColor(ttt, cv2.COLOR_HLS2BGR)  
                       

                        saveimg = None
                        aa = True

                        for i in range(1, 3):  
                            if (tags[0][i] == 1):
                                ttt = torch.zeros(cams.shape)[0][0]
                                ttt[:, :] = i
                                ttt = get_colored_mask(ttt.cpu().numpy())
                                ttt = cv2.cvtColor(ttt, cv2.COLOR_RGB2HSV)  
                                ttt[:, :, 2] = (230 * cams[0][i].cpu().numpy()).astype(np.uint8) 
                                ttt = cv2.cvtColor(ttt, cv2.COLOR_HSV2BGR)  

                                if (aa):
                                    saveimg = ttt.astype(np.float32)
                                    aa = False
                                else:
                                    saveimg += ttt.astype(np.float32)

                        saveimg[saveimg > 255] = 255
                        saveimg = saveimg.astype(np.uint8)
                        cv2.imwrite(os.path.join(self.save_np_path, image_ids[0] + '_' + str(i) + '.png'), saveimg)  
                        CAM = generate_vis(cams[0].cpu().numpy(), None, img_8, func_label2color=VOClabel2colormap, threshold=None, norm=False)

                        for i in range(3):        
                            if (tags[0][i] == 1):
                                save_img = CAM[i].transpose(1, 2, 0) * 255
                                save_img = cv2.cvtColor(save_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                                cv2.imwrite(os.path.join(self.save_np_path, image_ids[0] + '_' + str(i) + '.png'), save_img)

                    if (step == 600) or (step == 200) or step == 100 or step == 1450:
                        print(self.getbest_miou(clear=False))

                    for th in self.th_list:    
                        cams[:, 0] = th 
                      
                        predictions = torch.argmax(cams, dim=1)  

                        for batch_index in range(images.size()[0]):
                            pred_mask = get_numpy_from_tensor(predictions[batch_index])
                            gt_mask = get_numpy_from_tensor( 
                                                                gt_masks[batch_index])
                            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                           

                            self.meterlist[self.parms.index((renum, th))].add(pred_mask, gt_mask) 

                            if (self.save_png_path != None):
                                cur_save_path = os.path.join(self.save_png_path, str(th))

                                if not os.path.exists(cur_save_path):
                                    os.mkdir(cur_save_path)
                                cur_save_path = os.path.join(cur_save_path, str(renum))
                                if not os.path.exists(cur_save_path):
                                    os.mkdir(cur_save_path)

                                img_path = os.path.join(cur_save_path, image_ids[batch_index] + '.png')
                                save_colored_mask(pred_mask, img_path)

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        self.C_model.train()

        if (self.save_png_path != None):
            savetxt_path = os.path.join(self.save_png_path, "result.txt")

            with open(savetxt_path, 'wb') as f:
                for parm, meter in zip(self.parms, self.meterlist):
                    cur_iou = meter.get(clear=False)[-2]
                    f.write('{:>10.2f} {:>10.2f} {:>10.2f}\n'.format(
                        cur_iou, parm[0], parm[1]).encode())

        ret = self.getbest_miou()

        return ret


if __name__ =="__main__":

    
    args =get_params()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    Cmodel_path = args.Cmodel_path

    if args.dataset == 'PubBUSI':
        tagA = Cmodel_path[69:-4]
    if args.dataset == 'PubDB':
        tagA = Cmodel_path[67:-4]

    time_string = time.strftime("%Y_%m_%d_%H_%M_%S")


    log_tag = create_directory(f'./experiments/logs/{args.tag}/')
    log_path = log_tag + f'/%s_{time_string}.txt' % tagA

    if (args.savepng or args.savenpy):
        prediction_tag = create_directory(f'./experiments/predictions/{args.tag}/{tagA}/')
        prediction_path = create_directory(prediction_tag + f'{time_string}/')

    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    log_func(str(args))

    model = CLSNet('resnet50', num_classes=3)

    model = model.cuda()
    model.train()
    model.load_state_dict(torch.load(args.Cmodel_path))

    _savepng_path = None
    _savenpy_path = None

    if (args.savepng):
        _savepng_path = create_directory(prediction_path + 'pseudo/')
    if (args.savenpy):
        _savenpy_path = create_directory(prediction_path + 'camnpy/')

    log_func(str(args.beta))

    evaluatorA = evaluator(dataset=args.dataset, domain=args.domain, muti_scale=True,
                           save_np_path=_savenpy_path, savepng_path=_savepng_path, refine_list=[0],
                           th_list=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    
    ret = evaluatorA.evaluate(model)

    log_func(str(ret))

    log_func('IMG_train')
    log_func(str(args.beta))
    log_func(str(args.gama))
