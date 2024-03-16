"""
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection
Ziyun Yang, Somayyeh Soltanian-Zadeh and Sina Farsiu
Codes from: https://github.com/Zyun-Y/BiconNets
Paper: https://arxiv.org/abs/2103.00334
"""

import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import scipy.io as scio

#Directable={'upper_left':[-1,-1],'up':[0,-1],'upper_right':[1,-1],'left':[-1,0],'right':[1,0],'lower_left':[-1,1],'down':[0,1],'lower_right':[1,1]}
#TL_table = ['lower_right','down','lower_left','right','left','upper_right','up','upper_left']
Directable={'up':[0,-1],'upper':[0,-2],'left':[-1,0],'lefter':[-2,0],'right':[1,0],'righter':[2,0],'down':[0,1],'downer':[0,2]}
TL_table = ['downer','down','righter','right','lefter','left','upper','up']

class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def soft_dice_coeff(self, y_pred,y_true):
        smooth = 0.001  # may change
        i = torch.sum(y_true,dim=(1,2))
        j = torch.sum(y_pred,dim=(1,2))
        intersection = torch.sum(y_true * y_pred,dim=(1,2))

        score = (2. * intersection + smooth) / (i + j + smooth)

        return (1-score)
    def soft_dice_loss(self, y_pred,y_true):
        loss = self.soft_dice_coeff(y_true, y_pred)
        return loss.mean()

    def __call__(self, y_pred,y_true):
        b = self.soft_dice_loss(y_true, y_pred)
        return b


def edge_loss(vote_out,con_target):
    # print(vote_out.shape,con_target.shape)
    sum_conn = torch.sum(con_target.clone(),dim=1)
    edge = torch.where((sum_conn<8) & (sum_conn>0),torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))

    pred_mask_min, _ = torch.min(vote_out.cuda(), dim=1)

    pred_mask_min = pred_mask_min*edge

    minloss = F.binary_cross_entropy(pred_mask_min,torch.full_like(pred_mask_min, 0))
    # print(minloss)
    return minloss#+maxloss

class bicon_loss(nn.Module):
    def __init__(self, size):
        super(bicon_loss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()
        self.dice_loss = dice_loss()
        self.num_class = 1
        hori_translation = torch.zeros([1,self.num_class, size[1],size[1]])
        for i in range(size[1]-1):
            hori_translation[:,:,i,i+1] = torch.tensor(1.0)
        verti_translation = torch.zeros([1,self.num_class,size[0],size[0]])
        for j in range(size[0]-1):
            verti_translation[:,:,j,j+1] = torch.tensor(1.0)
        self.hori_trans = hori_translation.float().cuda()
        self.verti_trans = verti_translation.float().cuda()

        
    def forward(self, c_map, target, con_target):
        
        batch_num = c_map.shape[0]

        con_target = con_target.type(torch.FloatTensor).cuda()

        #find edge ground truth
        #sum_conn = torch.sum(con_target,dim=1)
        #edge = torch.where(sum_conn<8,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        #edge0 = torch.where(sum_conn>0,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        #edge = edge*edge0


        target = target.type(torch.FloatTensor).cuda()
        c_map = F.sigmoid(c_map)
        
        # construct the translation matrix
        self.hori_translation = self.hori_trans.repeat(batch_num,1,1,1).cuda()
        self.verti_translation  = self.verti_trans.repeat(batch_num,1,1,1).cuda()

        #final_pred, bimap = self.Bilater_voting(c_map)

        # apply any loss you want below using final_pred, e.g., dice loss.
        #loss_dice = self.dice_loss(final_pred, target)
        # vote_out = vote_out.squeeze(1)
        #decouple loss
        # print(c_map.max(),bimap.max(),final_pred.max())
        #decouple_loss = edge_loss(bimap.squeeze(1),con_target)

        #bce_loss = self.cross_entropy_loss(final_pred,target)

        conmap_l = self.cross_entropy_loss(c_map,con_target)
        #bimap_l = self.cross_entropy_loss(bimap.squeeze(1),con_target)
        loss = conmap_l #+ bce_loss# + loss_dice ## add dice loss if needed for biomedical data  loss_dice = self.dice_loss(final_pred, target)  +0.2*bimap_l+0.2*bimap_l+
        # print(loss)
        return loss
    
    def shift_diag(self,img,shift):
        ## shift = [1,1] moving right and down
        # print(img.shape,self.hori_translation.shape)
        batch,class_num, row, column = img.size()

        if shift[0]: ###horizontal
            img = torch.bmm(img.view(-1,row,column),self.hori_translation.view(-1,column,column)) if shift[0]==1 else torch.bmm(img.view(-1,row,column),self.hori_translation.transpose(3,2).view(-1,column,column))
        if shift[1]: ###vertical
            img = torch.bmm(self.verti_translation.transpose(3,2).view(-1,row,row),img.view(-1,row,column)) if shift[1]==1 else torch.bmm(self.verti_translation.view(-1,row,row),img.view(-1,row,column))
        return img.view(batch,class_num, row, column)


    def Bilater_voting(self,c_map):
        c_map = c_map.view(c_map.shape[0],-1,8,c_map.shape[2],c_map.shape[3])
        batch,class_num,channel, row, column = c_map.size()


        shifted_c_map = torch.zeros(c_map.size()).cuda()
        for i in range(8):
            shifted_c_map[:,:,i] = self.shift_diag(c_map[:,:,7-i].clone(),Directable[TL_table[i]])
        vote_out = c_map*shifted_c_map

        pred_mask,_ = torch.max(vote_out,dim=2)
        # print(pred_mask)
        return pred_mask, vote_out#, bimap
