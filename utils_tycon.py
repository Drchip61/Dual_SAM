"""
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection
Ziyun Yang, Somayyeh Soltanian-Zadeh and Sina Farsiu
Codes from: https://github.com/Zyun-Y/BiconNets
Paper: https://arxiv.org/abs/2103.00334
"""

import os
import cv2
import torch
import glob
import numpy as np
import torch.nn.functional as F
import torch

#Directable={'upper_left':[-1,-1],'up':[0,-1],'upper_right':[1,-1],'left':[-1,0],'right':[1,0],'lower_left':[-1,1],'down':[0,1],'lower_right':[1,1]}
Directable={'upper':[0,-2],'up':[0,-1],'lefter':[-2,0],'left':[-1,0],'right':[1,0],'righter':[2,0],'down':[0,1],'downer':[0,2]}
TL_table = ['downer','down','righter','right','left','lefter','up','upper']

#a =  torch.ones((256,256))
def sal2conn(mask):
    ## converte the saliency mask into a connectivity mask
    ## mask shape: H*W, output connectivity shape: 8*H*W
    [rows, cols] = mask.shape
    conn = torch.zeros([8,rows, cols])
    up = torch.zeros([rows, cols])#move the orignal mask to up
    down = torch.zeros([rows, cols])
    left = torch.zeros([rows, cols])
    right = torch.zeros([rows, cols])
    upper = torch.zeros([rows, cols])
    righter = torch.zeros([rows, cols])
    lefter = torch.zeros([rows, cols])
    downer = torch.zeros([rows, cols])


    up[:rows-1, :] = mask[1:rows,:]
    down[1:rows,:] = mask[0:rows-1,:]
    left[:,:cols-1] = mask[:,1:cols]
    right[:,1:cols] = mask[:,:cols-1]
    upper[:rows-2, :] = mask[2:rows,:]
    downer[2:rows,:] = mask[0:rows-2,:]
    lefter[:,:cols-2] = mask[:,2:cols]
    righter[:,2:cols] = mask[:,:cols-2]
    #up_left[0:rows-1,0:cols-1] = mask[1:rows,1:cols]
    #up_right[0:rows-1,1:cols] = mask[1:rows,0:cols-1]
    #down_left[1:rows,0:cols-1] = mask[0:rows-1,1:cols]
    #down_right[1:rows,1:cols] = mask[0:rows-1,0:cols-1]

    conn[0] = mask*downer
    conn[1] = mask*down
    conn[2] = mask*righter
    conn[3] = mask*right
    conn[4] = mask*left
    conn[5] = mask*lefter
    conn[6] = mask*up
    conn[7] = mask*upper
    conn = conn.float()
    return conn
#print(sal2conn(a)[-2])


def bv_test(output_test):
    '''
    generate the continous global map from output connectivity map as final saliency output 
    via bilateral voting
    '''

    #construct the translation matrix
    num_class = 1
    hori_translation = torch.zeros([output_test.shape[0],num_class,output_test.shape[3],output_test.shape[3]])
    for i in range(output_test.shape[3]-1):
        hori_translation[:,:,i,i+1] = torch.tensor(1.0)
    verti_translation = torch.zeros([output_test.shape[0],num_class,output_test.shape[2],output_test.shape[2]])
    # print(verti_translation.shape)
    for j in range(output_test.shape[2]-1):
        verti_translation[:,:,j,j+1] = torch.tensor(1.0)

    hori_translation = hori_translation.float().cuda()
    verti_translation = verti_translation.float().cuda()
    output_test = F.sigmoid(output_test)
    pred =ConMap2Mask_prob(output_test,hori_translation,verti_translation)
    return pred


def shift_diag(img,shift,hori_translation,verti_translation):
        ## shift = [1,1] moving right and down
        # print(img.shape,hori_translation.shape)
        batch,class_num, row, column = img.size()

        if shift[0]: ###horizontal
            img = torch.bmm(img.view(-1,row,column),hori_translation.view(-1,column,column)) if shift[0]==1 else torch.bmm(img.view(-1,row,column),hori_translation.transpose(3,2).view(-1,column,column))
        if shift[1]: ###vertical
            img = torch.bmm(verti_translation.transpose(3,2).view(-1,row,row),img.view(-1,row,column)) if shift[1]==1 else torch.bmm(verti_translation.view(-1,row,row),img.view(-1,row,column))
        return img.view(batch,class_num, row, column)


def ConMap2Mask_prob(c_map,hori_translation,verti_translation):
    c_map = c_map.view(c_map.shape[0],-1,8,c_map.shape[2],c_map.shape[3])
    batch,class_num,channel, row, column = c_map.size()

    shifted_c_map = torch.zeros(c_map.size()).cuda()
    for i in range(8):
        shifted_c_map[:,:,i] = shift_diag(c_map[:,:,7-i].clone(),Directable[TL_table[i]],hori_translation,verti_translation)
    vote_out = c_map*shifted_c_map

    pred_mask,_ = torch.max(vote_out,dim=2)
    # print(pred_mask)
    return pred_mask
