import cv2
import numpy as np
import math
import os

def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def nothing(x):
    pass

data_base_dir = 'image'  # 输入文件夹的路径
outfile_dir = 'gamma_image'  # 输出文件夹的路径

image_name = os.listdir(data_base_dir)
#list.sort()
#list2 = os.listdir(outfile_dir)
#list2.sort()
for i in range(len(image_name)):  # 遍历目标文件夹图片
    read_img_name = data_base_dir + '/' + image_name[i]  # 取图片完整路径
    image = cv2.imread(read_img_name)  # 读入图片
    img_gray = cv2.imread(read_img_name, 0)  # 灰度图读取，用于计算gamma值

    mean = np.mean(img_gray)
    gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma

    image_gamma_correct = gamma_trans(image, gamma_val)  # gamma变换

    out_img_name = outfile_dir + '/' + image_name[i]
    cv2.imwrite(out_img_name, image_gamma_correct)
    #print("The photo which is processed is {}".format(file))
''' 
read_img_name = '1.png'
image = cv2.imread(read_img_name)  # 读入图片
img_gray = cv2.imread(read_img_name, 0)  # 灰度图读取，用于计算gamma值

mean = np.mean(img_gray)
gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma

image_gamma_correct = gamma_trans(image, gamma_val)  # gamma变换

out_img_name = 'gamma.png'
cv2.imwrite(out_img_name, image_gamma_correct)
'''
