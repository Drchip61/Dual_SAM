import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
import pytorch_ssim
import pytorch_iou

import dataset_medical_mas as dataset_medical
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from connect_loss_ty import bicon_loss
from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry
from utils_tycon import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic

from sam_lora_image_encoder_modi import LoRA_Sam
import numpy as np
np.object=object
np.bool=bool
np.int=int
#np.typeDict=typeDict
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('ytylog_NO')

bicon_loss0 = bicon_loss((512,512))

bicon_loss1 = bicon_loss((256,256))

bicon_loss2 = bicon_loss((128,128))

bicon_loss3 = bicon_loss((64,64))
sam = sam_model_registry["vit_b"](checkpoint='sam_vit_b_01ec64.pth')#"sam_vit_b_01ec64.pth")
sam = sam[0]
model = LoRA_Sam(sam,4).cuda()

#pretrain = 'sam_vit_h_4b8939.pth'
#pretrain ="sam_vit_b_01ec64.pth" 
#model.load_lora_parameters(pretrain)

# from thop import profile
# input = torch.rand(1,3,512,512).cuda()
# flops,param = profile(model,(input,input,1,512,1))
# print(flops/1000000000,param/1000000)
# print(sum(p.numel()/1000000 for p in model.parameters() if p.requires_grad))
from thop import profile
input = torch.rand(1,3,512,512).cuda()
flops,param = profile(model,(input,input,1,512,1))
print(flops/1000000000,param/1000000)
print(sum(p.numel()/1000000 for p in model.parameters() if p.requires_grad))


#path ="usod_prompt.pth"
#model.load_state_dict(torch.load(path))

train_path = 'USOD10k/TR/'
cfg = dataset_medical.Config(datapath=train_path, savepath='./saved_model/msnet', mode='train', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=50)
data = dataset_medical.Data(cfg)


warnings.filterwarnings("ignore")
ssim_loss = pytorch_ssim.SSIM(window_size=7,size_average=True).cuda()
iou_loss = pytorch_iou.IOU().cuda()

model = model.train()
ce_loss = nn.CrossEntropyLoss()
deal = nn.Sigmoid()
base_lr = 0.005
EPOCH = 25
LR= 0.01
bceloss = nn.BCELoss()
warmup_period  = 2950
print(warmup_period)
b_ = base_lr/warmup_period

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, betas=(0.9, 0.999), weight_decay=0.1)





train_loader= DataLoader(data,
                      shuffle=False,
                      batch_size=5,
                      pin_memory=True,
                      num_workers=16,
                      )


losses0 = 0
losses1 = 0
losses2 = 0
losses3 = 0
losses4 = 0
losses5 = 0
losses6 = 0
losses7 = 0
losses8 = 0
losses9 = 0
losses10 = 0
losses11 = 0
losses12 = 0
losses13 = 0
losses14 = 0
losses15 = 0
yty1=0
yty2=0
yty3=0
yty4=0
print(len(train_loader))

def adjust_learning_rate(optimizer,epoch,start_lr):
    if epoch%10 == 0:  #epoch != 0 and
    #lr = start_lr*(1-epoch/EPOCH)
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"]*0.1
        print(param_group["lr"])

def xishu(epoch_num,epoch_all):
    xishu = 0.1*math.exp(-5*(1-epoch_num/epoch_all)**2)
    return xishu
iter_num = 0
LR=0.01
max_iterations = 29500
iter=0
for epoch_num in range(EPOCH):
    print(epoch_num)
    #if epoch_num == 52:
    #    break
    adjust_learning_rate(optimizer,epoch_num,LR)

    fake_index = xishu(epoch_num,EPOCH)
    #if epoch_num < 40:
    #    fake_index = fake_index
    #else:
    #    fake_index = xishu(40,EPOCH)
    print('LR is:',optimizer.state_dict()['param_groups'][0]['lr'])
    show_dict = {'epoch':epoch_num}
    for i_batch,(im1,gamma,mask,mask1,mask2,mask3) in enumerate(tqdm.tqdm(train_loader,ncols=60,postfix=show_dict)):  #,edge0,edge1,edge2,edge3

        im1 = im1.cuda().float()
        gamma = gamma.cuda().float()
        label0 = mask[1].cuda().float()
        label1 = mask[0].cuda().float()

        label0_1 = mask1[1].cuda().float()
        label1_1 = mask1[0].cuda().float()

        label0_2 = mask2[1].cuda().float()
        label1_2 = mask2[0].cuda().float()

        label0_3 = mask3[1].cuda().float()
        label1_3 = mask3[0].cuda().float()

        outputs = model(im1,gamma,1,512,1)#[:,:2,:,:]
        # if i_batch == 3 or i_batch== 53 or i_batch== 103 or i_batch== 153 or i_batch==203 or i_batch==253 or i_batch==303 or i_batch== 352:
        #     output_example = bv_test(outputs[0][0])[0][0]
        #     #print(output_example.size())
        #     image_numpy = np.array(output_example.cpu().detach().numpy())  # 转化成numpy数据类型
        #     writer.add_image('ori'+str(i_batch), image_numpy, epoch_num, dataformats='HW')  # 最后的参数指定维度格式
        # if i_batch == 3 or i_batch == 53 or i_batch == 103 or i_batch == 153 or i_batch == 203 or i_batch == 253 or i_batch == 303 or i_batch == 352:
        #     output_example1 = bv_test(outputs[1][0])[0][0]
        #     # print(output_example.size())
        #     image_numpy1 = np.array(output_example1.cpu().detach().numpy())  # 转化成numpy数据类型
        #     writer.add_image("gamma" + str(i_batch), image_numpy1, epoch_num, dataformats='HW')  # 最后的参数指定维度格式
        #ori_fke
        #for i in range(4):
        ori_fake0 = deal(outputs[0][0]).ge(0.5001).float()
        gamma_fake0 = deal(outputs[1][0]).ge(0.5001).float()

        ori_fake1 = deal(outputs[0][1]).ge(0.5001).float()
        gamma_fake1 = deal(outputs[1][1]).ge(0.5001).float()

        ori_fake2 = deal(outputs[0][2]).ge(0.5001).float()
        gamma_fake2 = deal(outputs[1][2]).ge(0.5001).float()

        ori_fake3 = deal(outputs[0][3]).ge(0.5001).float()
        gamma_fake3 = deal(outputs[1][3]).ge(0.5001).float()

        loss0 = bicon_loss0(outputs[0][0],label1.unsqueeze(1),label0)
        loss1 = bicon_loss1(outputs[0][1],label1_1.unsqueeze(1),label0_1)
        loss2 = bicon_loss2(outputs[0][2],label1_2.unsqueeze(1),label0_2)
        loss3 = bicon_loss3(outputs[0][3],label1_3.unsqueeze(1),label0_3)

        loss4 = bicon_loss0(outputs[1][0],label1.unsqueeze(1),label0)
        loss5 = bicon_loss1(outputs[1][1],label1_1.unsqueeze(1),label0_1)
        loss6 = bicon_loss2(outputs[1][2],label1_2.unsqueeze(1),label0_2)
        loss7 = bicon_loss3(outputs[1][3],label1_3.unsqueeze(1),label0_3)

        loss8 = fake_index*bceloss(deal(outputs[0][0]),gamma_fake0)
        loss9 = fake_index*bceloss(deal(outputs[1][0]),ori_fake0)

        loss10 = fake_index*bceloss(deal(outputs[0][1]),gamma_fake1)
        loss11 = fake_index*bceloss(deal(outputs[1][1]),ori_fake1)

        loss12 = fake_index*bceloss(deal(outputs[0][2]),gamma_fake2)
        loss13 = fake_index*bceloss(deal(outputs[1][2]),ori_fake2)

        loss14 = fake_index*bceloss(deal(outputs[0][3]),gamma_fake3)
        loss15 = fake_index*bceloss(deal(outputs[1][3]),ori_fake3)


        loss = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11+loss12+loss13+loss14+loss15#+0.05*loss5


        losses0 += loss0
        losses1 += loss1
        losses2 += loss2
        losses3 += loss3
        losses4 += loss4
        losses5 += loss5
        losses6 += loss6
        losses7 += loss7
        losses8 += loss8
        losses9 += loss9

        losses10 += loss10
        losses11 += loss11
        losses12 += loss12
        losses13 += loss13
        losses14 += loss14
        losses15 += loss15
        #
        # yty1+=loss0
        # yty2+=loss4
        # yty3+=loss8
        # yty4+=loss9
        #losses5 += 0.05*loss5


        optimizer.zero_grad()
        #scheduler(optimizer,i_batch,epoch_num)
        loss.backward()
        optimizer.step()

        if i_batch%50 == 0:
            print(i_batch,'|','losses0: {:.3f}'.format(losses0.data),'|','losses1: {:.3f}'.format(losses1.data),'|','losses2: {:.3f}'.format(losses2.data),'|','losses3: {:.3f}'.format(losses3.data),'|','losses4: {:.3f}'.format(losses4.data),'|','losses5: {:.3f}'.format(losses5.data),'|','losses6: {:.3f}'.format(losses6.data),'|','losses7: {:.3f}'.format(losses7.data),'|','losses8: {:.3f}'.format(losses8.data),'|','losses9: {:.3f}'.format(losses9.data),'|','losses10: {:.3f}'.format(losses10.data),'|','losses11: {:.3f}'.format(losses11.data),'|','losses12: {:.3f}'.format(losses12.data),'|','losses13: {:.3f}'.format(losses13.data),'|','losses14: {:.3f}'.format(losses14.data),'|','losses15: {:.3f}'.format(losses15.data))


            losses0=0
            losses1=0
            losses2=0
            losses3=0
            losses4=0
            losses5=0
            losses6=0
            losses7=0
            losses8=0
            losses9=0
            losses10=0
            losses11=0
            losses12=0
            losses13=0
            losses14=0
            losses15=0
    # writer.add_scalars("Train_Loss", {'Ori': yty1.item(),'Gamma': yty2.item(),'P_Ori': yty3.item(),'P_Gamma': yty4.item()}, epoch_num)
    #
    # yty1=0
    # yty2=0
    # yty3=0
    # yty4=0
    # for name, param in model.named_parameters():
    #     if 'desam' in name and 'desam.de' not in name and 'sam.mask_decoder' not in name and 'sam.prompt_encoder' not in name and 'sam.image_encoder' not in name:
    #         print(name)
    #         writer.add_histogram(name + '_grad', param.grad.clone().cpu().data.numpy(), epoch_num)
    #         writer.add_histogram(name + '_data', param, epoch_num)

    torch.save(model.state_dict(),'usod_prompt2.pth')
writer.close()
