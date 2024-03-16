<div align="center">
<h1>Dual-SAM </h1>
<h3>Fantastic Animals and Where to Find Them: Segment Any Marine Animal with Dual SAM</h3>

[Pingping Zhang*](http://faculty.dlut.edu.cn/zhangpingping/en/index.htm)<sup>1</sup>,[Tianyu Yan](https://github.com/Drchip61)<sup>1</sup>,[Yang Liu](http://faculty.dlut.edu.cn/liuyang1/zh_CN/index.htm)<sup>1</sup>, [Huchuan Lu](http://faculty.dlut.edu.cn/Huchuan_Lu/zh_CN/index.htm)<sup>1</sup>

[Dalian University of Technology, IIAU-Lab](https://futureschool.dlut.edu.cn/IIAU.htm)<sup>1</sup>


## Abstract

As an important pillar of underwater intelligence, Marine Animal Segmentation (MAS) involves segmenting animals within marine environments. Previous methods don't excel in extracting long-range contextual features and overlook the connectivity between pixels. Recently, Segment Anything Model (SAM) offers a universal framework for general segmentation tasks. Unfortunately, trained with natural images, SAM does not obtain the prior knowledge from marine images. In addition, the single-position prompt of SAM is very insufficient for prior guidance. To address these issues, we propose a novel learning framework, named Dual-SAM for high-performance MAS. To this end, we first introduce a dual structure with SAM's paradigm to enhance feature learning of marine images. Then, we propose a Multi-level Coupled Prompt (MCP) strategy to instruct comprehensive underwater prior information, and enhance the multi-level features of SAM's encoder with adapters. Subsequently, we design a Dilated Fusion Attention Module (DFAM) to progressively integrate multi-level features from SAM's encoder. With dual decoders, it generates pseudo-labels and achieves mutual supervision for harmonious feature representations. Finally, instead of directly predicting the masks of marine animals, we propose a Criss-Cross Connectivity Prediction (
C3P) paradigm to capture the inter-connectivity between pixels. It shows significant improvements over previous techniques. Extensive experiments show that our proposed method achieve state-of-the-art performances on five widely-used MAS datasets.
## Overview

* [**Dual-SAM**] is a novel learning framework for high performance Marine Animal Segmentation (MAS). The framework inherits the ability of SAM and adaptively incorporate prior knowledge of underwater scenarios.

<p align="center">
  <img src="github_show/framework.png" alt="accuracy" width="80%">
</p>

* **2D-Selective-Scan of VMamba**

<p align="center">
  <img src="assets/ss2d.png" alt="arch" width="60%">
</p>

* **VMamba has global effective receptive field**

<p align="center">
  <img src="assets/erf_comp.png" alt="erf" width="50%">
</p>


## Main Results

We will release all the pre-trained models/logs in few days!

* **Classification on ImageNet-1K**


| name | pretrain | resolution |acc@1 | #params | FLOPs | checkpoints/logs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeiT-S | ImageNet-1K | 224x224 | 79.8 | 22M | 4.6G | -- |
| DeiT-B | ImageNet-1K | 224x224 | 81.8 | 86M | 17.5G | -- |
| DeiT-B | ImageNet-1K | 384x384 | 83.1 | 86M | 55.4G | -- |
| Swin-T | ImageNet-1K | 224x224 | 81.2 | 28M | 4.5G | -- |
| Swin-S | ImageNet-1K | 224x224 | 83.2 | 50M | 8.7G | -- |
| Swin-B | ImageNet-1K | 224x224 | 83.5 | 88M | 15.4G | -- |
| VMamba-T | ImageNet-1K | 224x224 | 82.2 | 22M | 4.5G | [ckpt](https://drive.google.com/file/d/1ml7nZM-YPYbQurHiodf4dpXHw88dXFfP/view?usp=sharing)/[log](https://drive.google.com/file/d/1mVooWXl1Zj8ZALr1iYuoMLdG_yDbZpRx/view?usp=sharing) |
| VMamba-S | ImageNet-1K | 224x224 | 83.5 | 44M | 9.1G | [ckpt](https://drive.google.com/file/d/1gUlRxeHxkn4JG2QR_DoAPbzSFYAoSxDy/view?usp=sharing)/[log](https://drive.google.com/file/d/12l81-VsPcCRjyIByWQzyO_EsovVj_00v/view?usp=sharing) |
| VMamba-B | ImageNet-1K | 224x224 | 84.0 | 75M | 15.2G | waiting |

* **Object Detection on COCO**
  
| Backbone | #params | FLOPs | Detector | box mAP | mask mAP | checkpoints/logs |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | 48M | 267G | MaskRCNN@1x | 42.7| 39.3 |-- |
| VMamba-T | 42M | 262G | MaskRCNN@1x | 46.5| 42.1 |[ckpt](https://drive.google.com/file/d/1SIQFPpXkVBPB4mx1VO9P9nH4ebvTH0W5/view?usp=sharing)/[log](https://drive.google.com/file/d/15nd3AZuOkHpqlZhVUEXilnsVzd1qn8Kc/view?usp=sharing) |
| Swin-S | 69M | 354G | MaskRCNN@1x | 44.8| 40.9 |-- |
| VMamba-S | 64M | 357G | MaskRCNN@1x | 48.2| 43.0 |[ckpt](https://drive.google.com/file/d/1LzytVo2wTKgOxyBadstzacslwol8Dvhq/view?usp=sharing)/[log](https://drive.google.com/file/d/1TbYZhban4VqC-9kQ8-kuZOPSBX484sSj/view?usp=sharing) |
| Swin-B | 107M | 496G | MaskRCNN@1x | 46.9| 42.3 |-- |
| VMamba-B | 96M | 482G | MaskRCNN@1x | 48.5| 43.1 |[ckpt](https://huggingface.co/sunsmarterjieleaf/VMamba/tree/main)/[log](https://huggingface.co/sunsmarterjieleaf/VMamba/tree/main) |
| Swin-T | 48M | 267G | MaskRCNN@3x | 46.0| 41.6 |-- |
| VMamba-T | 42M | 262G | MaskRCNN@3x | 48.5| 43.2 |[ckpt](https://drive.google.com/file/d/1SmsgM2SR_GbKjq1EkcLcQXCEIlPhA-_r/view?usp=sharing)/[log](https://drive.google.com/file/d/1EVUKFsPQI3bqelX7-WlTKFdToXjwmcXU/view?usp=sharing) |
| Swin-S | 69M | 354G | MaskRCNN@3x | 48.2| 43.2 |-- |
| VMamba-S | 64M | 357G | MaskRCNN@3x | 49.7| 44.0 |[ckpt](https://huggingface.co/sunsmarterjieleaf/VMamba/tree/main)/[log](https://huggingface.co/sunsmarterjieleaf/VMamba/tree/main) |

* **Semantic Segmentation on ADE20K**

| Backbone | Input|  #params | FLOPs | Segmentor | mIoU | checkpoints/logs |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | 512x512 | 60M | 945G | UperNet@160k | 44.4| -- |
| VMamba-T| 512x512 | 55M | 939G | UperNet@160k | 47.3| [ckpt](https://drive.google.com/file/d/1hLAGFBRJfaFSzyPlqsGbKXXN_gQJMLzn/view?usp=sharing)/[log](https://drive.google.com/file/d/17nh9_hdF9QQxyqj81U86HoGUnMxZQ4nN/view?usp=sharing) |
| Swin-S | 512x512 | 81M | 1039G | UperNet@160k | 47.6| -- |
| VMamba-S| 512x512 | 76M | 1037G | UperNet@160k | 49.5| [ckpt](https://drive.google.com/file/d/18GReI1A6LckwnPrnEFPXp9at7VB8GiJW/view?usp=sharing)/[log](https://drive.google.com/file/d/1m-Pd4_kPgF6Dt2E33sfIf_g9jVWxfPnG/view?usp=sharing) |
| Swin-B | 512x512 | 121M | 1188G | UperNet@160k | 48.1| -- |
| VMamba-B| 512x512 | 110M | 1167G | UperNet@160k | 50.0| [ckpt](https://huggingface.co/sunsmarterjieleaf/VMamba/tree/main)/[log](https://huggingface.co/sunsmarterjieleaf/VMamba/tree/main) |
| Swin-S | 640x640 | 81M | 1614G | UperNet@160k | 47.9| -- |
| VMamba-S| 640x640 | 76M | 1620G | UperNet@160k | 50.8| [ckpt](https://huggingface.co/sunsmarterjieleaf/VMamba/tree/main)/[log](https://huggingface.co/sunsmarterjieleaf/VMamba/tree/main) |



## Getting Started

### Installation

**step1:Clone the VMamba repository:**

To get started, first clone the VMamba repository and navigate to the project directory:

```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba

```

**step2:Environment Setup:**

VMamba recommends setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment:
#### Create and activate a new conda environment

```bash
conda create -n vmamba
conda activate vmamba
```
#### Install Dependencies.
```bash
pip install -r requirements.txt
# Install selective_scan and its dependencies
cd selective_scan && pip install . && pytest
```



Optional Dependencies for Model Detection and Segmentation:
```bash
pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

### Model Training and Inference

**Classification:**

To train VMamba models for classification on ImageNet, use the following commands for different configurations:

```bash
# For VMamba Tiny
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg configs/vssm/vssm_tiny_224.yaml --batch-size 128 --data-path /dataset/ImageNet2012 --output /tmp

# For VMamba Small
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg configs/vssm/vssm_small_224.yaml --batch-size 128 --data-path /dataset/ImageNet2012 --output /tmp

# For VMamba Base
python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg configs/vssm/vssm_base_224.yaml --batch-size 128 --data-path /dataset/ImageNet2012 --output /tmp

```

**Detection and Segmentation:**

For detection and segmentation tasks, follow similar steps using the appropriate config files from the `configs/vssm` directory. Adjust the `--cfg`, `--data-path`, and `--output` parameters according to your dataset and desired output location.

### Analysis Tools

VMamba includes tools for analyzing the effective receptive field, FLOPs, loss, and scaling behavior of the models. Use the following commands to perform analysis:

```bash
# Analyze the effective receptive field
CUDA_VISIBLE_DEVICES=0 python analyze/get_erf.py > analyze/show/erf/get_erf.log 2>&1

# Analyze FLOPs
CUDA_VISIBLE_DEVICES=0 python analyze/get_flops.py > analyze/show/flops/flops.log 2>&1

# Analyze loss
CUDA_VISIBLE_DEVICES=0 python analyze/get_loss.py

# Further analysis on scaling behavior
python analyze/scaleup_show.py

```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MzeroMiko/VMamba&type=Date)](https://star-history.com/#MzeroMiko/VMamba&Date)

## Citation

```
@article{liu2024vmamba,
  title={VMamba: Visual State Space Model},
  author={Liu, Yue and Tian, Yunjie and Zhao, Yuzhong and Yu, Hongtian and Xie, Lingxi and Wang, Yaowei and Ye, Qixiang and Liu, Yunfan},
  journal={arXiv preprint arXiv:2401.10166},
  year={2024}
}
```

## Acknowledgment

This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Swin-Transformer ([paper](https://arxiv.org/pdf/2103.14030.pdf), [code](https://github.com/microsoft/Swin-Transformer)), ConvNeXt ([paper](https://arxiv.org/abs/2201.03545), [code](https://github.com/facebookresearch/ConvNeXt)), [OpenMMLab](https://github.com/open-mmlab),
and the `analyze/get_erf.py` is adopted from [replknet](https://github.com/DingXiaoH/RepLKNet-pytorch/tree/main/erf), thanks for their excellent works.

* We release [Fast-iTPN](https://github.com/sunsmarterjie/iTPN/tree/main/fast_itpn) recently, which reports the best performance on ImageNet-1K at Tiny/Small/Base level models as far as we know. (Tiny-24M-86.5%, Small-40M-87.8%, Base-85M-88.75%)
