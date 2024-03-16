#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from utils_tycon import *

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    
    def __call__(self, image,gamma, mask):
        image = (image - self.mean)/self.std
        gamma = (gamma - self.mean)/self.std
        mask /= 255
        return image, gamma,mask

class RandomCrop(object):
    def __call__(self, image, mask):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image,gamma, mask,mask1,mask2,mask3):
        if np.random.randint(2)==0:
            return image[:, ::-1].copy(),gamma[:, ::-1].copy(), mask[:, ::-1].copy(), mask1[:, ::-1].copy(), mask2[:, ::-1].copy(), mask3[:, ::-1].copy()
        else:
            return image,gamma, mask,mask1,mask2,mask3

class RandomRotate(object):
    def __call__(self, image, mask):
        degree = 10
        rows, cols, channels = image.shape
        random_rotate = random.random() * 2 * degree - degree
        rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), random_rotate, 1)
        '''
        第一个参数：旋转中心点
        第二个参数：旋转角度
        第三个参数：缩放比例
        '''
        image = cv2.warpAffine(image, rotate, (cols, rows))
        mask = cv2.warpAffine(mask, rotate, (cols, rows))
        # contour = cv2.warpAffine(contour, rotate, (cols, rows))

        return image,mask



class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image,gamma, mask):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        gamma = cv2.resize(gamma, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(512,512), interpolation=cv2.INTER_NEAREST)
        mask1  = cv2.resize( mask, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        mask2  = cv2.resize( mask, dsize=(128,128), interpolation=cv2.INTER_NEAREST) 
        mask3  = cv2.resize( mask, dsize=(64,64), interpolation=cv2.INTER_NEAREST)
        return image,gamma, mask, mask1, mask2, mask3

class ToTensor(object):
    def __call__(self, image,gamma, mask, mask1, mask2, mask3):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        gamma = torch.from_numpy(gamma)
        gamma = gamma.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        mask1  = torch.from_numpy(mask1)
        mask2  = torch.from_numpy(mask2)        
        mask3  = torch.from_numpy(mask3)
        return image,gamma, mask,mask1,mask2,mask3


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.randomrotate = RandomRotate()
        self.resize     = Resize(512, 512)
        #self.resize1     = cv2.resize((352, 352), interpolation=cv2.INTER_NEAREST)
        self.totensor   = ToTensor()

        self.root = cfg.datapath

        img_path = os.path.join(self.root, 'Image')
        gt_path = os.path.join(self.root, 'Masks')
        self.samples = [os.path.splitext(f)[0]
                    for f in os.listdir(gt_path) if f.endswith('.png')]



    def __getitem__(self, idx):
        name  = self.samples[idx]
        image = cv2.imread(self.root+'/Image/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        gamma = cv2.imread(self.root+'/gamma_Image/'+name+'.jpg')[:,:,::-1].astype(np.float32)
        mask  = cv2.imread(self.root+'/Masks/' +name+'.png', 0).astype(np.float32)
        


        shape = mask.shape

        if self.cfg.mode=='train':
            image, gamma,mask = self.normalize(image, gamma,mask)
            image, gamma,mask,mask1,mask2,mask3 = self.resize(image, gamma,mask)
#             # image, mask = self.randomcrop(image, mask)
            #image,gamma, mask,mask1,mask2,mask3 = self.randomflip(image,gamma, mask,mask1,mask2,mask3)
#             image, mask = self.randomrotate(image, mask)
            image,gamma, mask,mask1,mask2,mask3 = self.totensor(image,gamma, mask,mask1,mask2,mask3)
            return image,gamma, [mask,sal2conn(mask)],[mask1,sal2conn(mask1)],[mask2,sal2conn(mask2)],[mask3,sal2conn(mask3)]
        else:
            image, gamma,mask = self.normalize(image, gamma,mask)
            image, gamma,mask,mask1,mask2,mask3 = self.resize(image, gamma,mask)#image,gamma, mask = self.resize(image,gamma, mask)
            image,gamma, mask,mask1,mask2,mask3 = self.totensor(image,gamma, mask,mask1,mask2,mask3)#image, gamma,mask = self.totensor(image,gamma, mask)
            return image, gamma,sal2conn(mask), name

#     def collate(self, batch):
#         size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
#         image, mask = [list(item) for item in zip(*batch)]
#         for i in range(len(batch)):
#             image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
#             mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
#         image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
#         mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
#         return image, mask

    def __len__(self):
        return len(self.samples)

'''
train_path = 'TrainDataset'
cfg = Config(datapath=train_path, savepath='./saved_model/msnet', mode='train', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=50)
data = Data(cfg)
print(data[0][-1].shape)
'''
# def check(a):
#     for i in range(a.shape[0]):
#         for j in range(a.shape[1]):
#             if a[i][j] != 0 and a[i][j] != 1:
#                 print(a[i][j])
#             else:
#                 print('love')
                
# for i in range(len(data)):
#     a = data[i][1]
#     check(a)

