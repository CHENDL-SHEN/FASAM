

from weakref import ref
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from torch.nn.modules.loss import _Loss



class MS_Loss(nn.Module):
    def __init__(self):
        super(MS_Loss, self).__init__()
        self.beta = 1e-7
        self.lambdaTV = 0.001
        self.penalty = 'l1'

    def levelsetLoss(self, output, target):
        outshape = output.shape  # [b,21,512,512]
        tarshape = target.shape  # [b, 3, 512 , 512]
        loss = 0.0
        for ich in range(tarshape[1]):  # 每个通道处

            target_ = torch.unsqueeze(target[:, ich], 1)  # [b,h,w] [b,1,h,w]
            target_ = target_.expand(
                tarshape[0],
                outshape[1],
                tarshape[2],
                tarshape[3])  # # [b,21,h,w]
            with torch.no_grad():
                pcentroid = torch.sum(
                    target_ * output, (2, 3)) / torch.sum(output, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - \
                pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output
            loss += torch.sum(pLoss) / 16

        return loss

    def gradientLoss2d(self, output):
        dH = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
        dW = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
        if (self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW

        loss = (torch.sum(dH) + torch.sum(dW)) / 16
        return loss

    def forward(self, output, target):
        loss_L = self.levelsetLoss(output, target)  # 水平集LS（levelset）；和原图中的颜色有关
        loss_A = self.gradientLoss2d(output) * self.lambdaTV  # 全变分项TV；梯度
        loss_LS = (loss_L + loss_A) * self.beta

        print(
            "loss_L={},loss_A={},loss_LS={}".format(
                loss_L * self.beta,
                loss_A * self.beta,
                loss_LS))

        return loss_LS


class Pro_Contrast_MS_Loss_GC(_Loss):
    def __init__(self, args, size_average=None, reduce=None, reduction='mean'):
        super(Pro_Contrast_MS_Loss_GC, self).__init__(size_average, reduce, reduction)
        self.args = args
        self.fg_c_num = 20 if args.dataset == 'voc12' else 80
        self.class_loss_fn = nn.CrossEntropyLoss().cuda()
        self.dist = self.compute_uclid_spatial_distances(args.dist_hw, args.dist_the)

        self.beta = 1e-5
        self.lambdaTV = args.lambdaTV
        self.penalty = 'l1'


    def compute_uclid_spatial_distances(self, dist_hw, dist_the):
        h = dist_hw
        w = dist_hw
        distances = np.zeros((h, w, h, w), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                for k in range(h):
                    for l in range(w):
                        distances[i, j, k, l] = np.sqrt((i - k)**2 + (j - l)**2)
    
        distances[distances > dist_the] = 0
        distances[distances != 0] = 1

        distances = torch.from_numpy(distances)
        distances = distances.view(h*w, h*w)

        return distances
    
    def compute_pixel_semantic_affinity(self, probs):

        b, cp, h, w = probs.shape    # probs(b,cp,h,w)=(144,21,10,10)
        probs = probs.view(b, cp, h*w)  # probs(144,21,100)
        probs_tran = probs.transpose(1, 2)  # probs_tran(144,100,21)
        aff = torch.bmm(probs_tran, probs)  # aff.mean()   # aff:(144,100,100)
        aff_sum = torch.sum(aff, dim=1, keepdim=True)   # aff_sum(144,1,100)
        aff = aff / (aff_sum + 1e-5)  # ret(144,100,100)
        
        return aff
    
    def exert_space_constrain_foraff(self, aff):
        ## 施加空间约束
        dist = self.dist
        dist_expanded = dist.unsqueeze(0).expand_as(aff)
        aff_dist = dist_expanded.cuda() * aff

        return aff_dist

    def reconstruc_RGB(self, sailencys, origin_f, aff_dist):
        b_s, c_s, h_s, w_s = sailencys.shape
        origin_flat = torch.bmm(origin_f.view(b_s, c_s, -1), aff_dist)   
        up_f = origin_flat.view(-1, 3, h_s, w_s)  

        return up_f
    
    def pro_contrast_levelsetloss(self, fg_cam, sailencys):

        b, c, h, w = fg_cam.size()                 
        imgmin_mask = sailencys.sum(1, True) != 0  
        sailencys = F.interpolate(sailencys.float(), size=(h, w))   

        bg = 1-torch.max(fg_cam, dim=1, keepdim=True)[0] ** 1       

        nnn = torch.max((1 - bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th   
        nnn2 = torch.max((bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th     
        nnn = nnn * nnn2       
        if (nnn.sum() == 0):
          nnn = torch.ones(nnn.shape).cuda()
        imgmin_mask = nnn.view(b, 1, 1, 1) * imgmin_mask    

        probs = torch.cat([bg, fg_cam], dim=1)           
        probs1 = probs * imgmin_mask                       

        origin_f = F.normalize(sailencys.detach(), dim=1)  
        origin_f = origin_f * imgmin_mask                  

        aff = self.compute_pixel_semantic_affinity(probs1)
        aff_dist = self.exert_space_constrain_foraff(aff)
        up_f = self.reconstruc_RGB(sailencys, origin_f, aff_dist)
        sal_loss = F.mse_loss(up_f, origin_f, reduce=False)

        salloss_broadcasted = sal_loss.unsqueeze(2)  
        probs1_broadcasted = probs1.unsqueeze(1) 
        result = torch.mul(salloss_broadcasted, probs1_broadcasted)  
        result = result.sum(dim=2)  

        pcls_loss = (result * imgmin_mask).sum() / (torch.sum(imgmin_mask) + 1e-3)  

        return pcls_loss
  
  
    def gradientLoss2d(self, output):
        
        output = output.clone().detach()
        output = F.softmax(output, dim=1)

        outshape = output.shape  # [b,21,512,512]
        dH = torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :])
        dW = torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1])
        if (self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW

        tv_loss = (torch.sum(dH) + torch.sum(dW)) / outshape[0]
        return tv_loss
   
    def forward(self, fg_cam, sailencys):
        loss_L = self.pro_contrast_levelsetloss(fg_cam, sailencys)
        loss_A = self.gradientLoss2d(fg_cam) * self.lambdaTV * self.beta  
        loss_MS = loss_L + loss_A

        return loss_MS
