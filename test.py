# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial
import json
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp 
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed

# from visualizer import get_local
# get_local.activate()
from networks import BNTransformer, Baseline, DMSTransformer, MaskFormer, ViT_m, ViT_mask, SwinTransformer, USMaskFormer

from utils.data_utils import get_loader
from tools.utils import find_surf

from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather
import datetime
import time
import SimpleITK as sitk
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from skimage import morphology, measure
from scipy.ndimage import binary_erosion, binary_dilation


parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
#general settings
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--modality", default=4, type=int, help="number of modalities")
#parser.add_argument("--modality", default=0, type=int, help="index of modalities")

parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory")
parser.add_argument("--dataset", default="RCC_500", type=str, help="choose dataset")
# parser.add_argument("--data_dir", default="../../Data/RCC_500", type=str, help="dataset directory")
parser.add_argument("--json_list", default="data.json", type=str, help="dataset json file")
parser.add_argument("--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name")
parser.add_argument("--pretrained", action="store_true", help="use pretrained UNETR parameters")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--erate", default=0.0, type=float, help="enhance rate")
parser.add_argument("--frate", default=0.0, type=float, help="enhance rate")
# parser.add_argument("--open_spatten", action="store_true", help="open same pos attention")
# parser.add_argument("--open_mcatten", action="store_true", help="open masked cross attention")
parser.add_argument("--num_classes", default=4, type=int, help="number of classes")
parser.add_argument("--num_BN", default=4, type=int, help="number of bottleneck tokens")

#training settings
parser.add_argument("--max_epochs", default=200, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-5, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--lrschedule", default="cosine_anneal", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")

#distributed settings
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")

#model parameters
parser.add_argument("--model_name", default="bntransformer", type=str, help="model name")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")

# ViT-B/16
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")

parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")

#preprocessing parameters
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

def connected_component(image, savenum=1):
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image
       
    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num+1)]
    area_list = [region[i-1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    # 去除面积较小的连通域
    if len(num_list_sorted) > savenum:
        # for i in range(3, len(num_list_sorted)):
        for i in num_list_sorted[savenum:]:
            # label[label==i] = 0
            label[region[i-1].slice][region[i-1].image] = 0
        num_list_sorted = num_list_sorted[:savenum]
    return label

def get_holes(image):
    image = ~image
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image
       
    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(1, num+1)]
    area_list = [region[i-1].area for i in num_list]
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]

    label[region[num_list_sorted[0]].slice][region[num_list_sorted[0]].image] = 0
    return label

def avg_dice(model, dataset='rcc500'):

    if dataset=='rcc500':
        folder='/home/star/data1/ghf/Data/RCC_500/data_processed_with_mask/'
        reg_channel = 3
        fixed_path = '/home/star/data1/ghf/Data/kits23/data_processed/case_00003_2/'
        print('fixed_path:', fixed_path)
        fixed = sitk.GetArrayFromImage(sitk.ReadImage(fixed_path + 'imaging.nii.gz'))
        fixed = torch.from_numpy(fixed.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()

    elif dataset=='brats2020':
        folder='/home/star/data1/ghf/Data/BraTs2020_training_data/data_processed_with_mask/'
        reg_channel = 2
        fixed_path = '/home/star/data1/ghf/Data/BraTs2020_training_data/data_erase_tumor/'
        print('fixed_path:', fixed_path)

    else:
        raise ValueError('no such dataset')
    
    model.eval()
    model = model.cuda()
    model.init(0,dataset)

    print('calculating dice')
    dices = []
    means = []

    # volume_101 和 volume_10 肿瘤位置差异比较大
    # 小肿瘤样例 102 105 107 11 118

    # 110看不出来 
    # 113、119是因为肿瘤贴近脑边缘，后处理给去掉了
    # 240、280很差

    names = sorted(os.listdir(folder))
    for name in ['volume_1.npy']:
        data = torch.from_numpy(np.load(folder+name).astype(np.float32)).unsqueeze(0).cuda()
        mask = data.squeeze()[-1].cpu().numpy()

        mask[mask>0]=1
        mask = mask.astype(np.uint8)
        compute_mask = np.ones_like(mask)

        fixed_names = sorted(os.listdir(fixed_path))
        for fixed_name in ['volume_105.npy']:
            # fixed = sitk.GetArrayFromImage(sitk.ReadImage(fixed_path + fixed_name + '/image.nii.gz'))
            # fixed = torch.from_numpy(fixed.astype(np.float32)).permute(1,2,0).unsqueeze(0).unsqueeze(0).cuda()
            fixed = torch.from_numpy(np.load(fixed_path+fixed_name).astype(np.float32)).unsqueeze(0).cuda()

            input_seg, tmp_mask, _, w_moving = model.pre_register(fixed[:,reg_channel].unsqueeze(1), data[:,reg_channel].unsqueeze(1), seg2=data[:,-1].unsqueeze(1))
            tmp_mask = tmp_mask.squeeze().cpu().numpy()

            _, _, differ, w_fixed = model.pre_register(data[:,reg_channel].unsqueeze(1), fixed[:,reg_channel].unsqueeze(1), seg2=data[:,-1].unsqueeze(1))
            differ = differ.squeeze().cpu().numpy()
            # differ = abs(differ)
            thres = 0.1
            differ[differ>=thres]=1
            differ[differ<thres]=0
            differ = differ.astype(np.uint8)

            # compute_mask = compute_mask & tmp_mask
            compute_mask = compute_mask & differ

            
            # differ = torch.from_numpy(differ.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
            # save_image(differ[::2],'differ.png',nrow=8,pad_value=1)

            w_moving = w_moving.squeeze().cpu().transpose(1,2).transpose(0,1).unsqueeze(1)
            save_image(w_moving[::2],'w_moving.png',nrow=8,pad_value=1)

            w_fixed = w_fixed.squeeze().cpu().transpose(1,2).transpose(0,1).unsqueeze(1)
            save_image(w_fixed[::2],'w_fixed.png',nrow=8,pad_value=1)

            fixed = fixed.squeeze()[reg_channel].cpu().transpose(1,2).transpose(0,1).unsqueeze(1)
            save_image(fixed[::2],'fixed.png',nrow=8,pad_value=1)
            # assert 0
        
        # 后处理-去除头部和尾部的结果
        compute_mask[:,:,:32]=0
        compute_mask[:,:,-16:]=0

        # 后处理-去除边界
        # boundary_region = find_surf(torch.from_numpy(image).bool(), kernel=9, thres=0.95)
        # compute_mask[boundary_region] = 0.5
        organ_mask = data.squeeze()[reg_channel].cpu().numpy()
        organ_mask[organ_mask>0]=1
        organ_mask = organ_mask.astype(np.uint8)
        organ_border = organ_mask - binary_erosion(organ_mask, iterations=10)
        compute_mask[organ_border>0] = 0

        # 后处理-腐蚀膨胀操作去除噪声
        # compute_mask = binary_erosion(compute_mask, iterations=2)
        # compute_mask = binary_dilation(compute_mask, iterations=2)

        # 后处理-保留最大连通域
        # compute_mask = connected_component(compute_mask)
        # compute_mask[compute_mask>0] = 1

        # 后处理-填补连通域内的空洞
        # holes = get_holes(compute_mask)
        # holes[holes>0] = 1
        # compute_mask[holes] = 1

        dice = dice_coefficient(mask,compute_mask)
        dices.append(dice)
        print(f'{name}:{dice}')

        means.append(compute_mask.mean())
        print('mean:', compute_mask.mean())

        image = data.squeeze()[reg_channel].cpu().numpy()
        input_seg = input_seg.squeeze().cpu().numpy()

        organ_border = torch.from_numpy(organ_border.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(organ_border[::2],'organ_border.png',nrow=8,pad_value=1)
        input_seg = torch.from_numpy(input_seg.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(input_seg[::2],'input_seg.png',nrow=8,pad_value=1)
        compute_mask = torch.from_numpy(compute_mask.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(compute_mask[::2],'compute_mask.png',nrow=8,pad_value=1)
        mask = torch.from_numpy(mask.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(mask[::2],'mask.png',nrow=8,pad_value=1)
        image = torch.from_numpy(image.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(image[::2],'image.png',nrow=8,pad_value=1)
        assert 0

    plt.hist(dices, bins=10)
    plt.savefig('hist.png')

    print('mean dice:', np.mean(np.array(dices)))
    print('mean mean:', np.mean(np.array(means)))

def avg_dice_song(model):
    
    model.eval()
    model = model.cuda()
    model.init(0, 'song')

    print('calculating dice')
    dices = []
    means = []

    pairs = [['/home/star/data1/ghf/Data/BraTs2020_training_data/song/1-1/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/1-2/'],
             ['/home/star/data1/ghf/Data/BraTs2020_training_data/song/2-1/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/2-2/'],
             ['/home/star/data1/ghf/Data/BraTs2020_training_data/song/3-1/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/3-2/'],
             ['/home/star/data1/ghf/Data/BraTs2020_training_data/song/4-1/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/4-2/'],
             ['/home/star/data1/ghf/Data/BraTs2020_training_data/song/5-1/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/5-2/'],
             ['/home/star/data1/ghf/Data/BraTs2020_training_data/song/1-2/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/1-1/'],
             ['/home/star/data1/ghf/Data/BraTs2020_training_data/song/2-2/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/2-1/'],
             ['/home/star/data1/ghf/Data/BraTs2020_training_data/song/3-2/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/3-1/'],
             ['/home/star/data1/ghf/Data/BraTs2020_training_data/song/4-2/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/4-1/'],
             ['/home/star/data1/ghf/Data/BraTs2020_training_data/song/5-2/', '/home/star/data1/ghf/Data/BraTs2020_training_data/song/5-1/']]
    for pair in pairs:
        
        fixed = torch.from_numpy(np.load(pair[0]+'image.npy').astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        fixed_mask = np.load(pair[0]+'tumor.npy')
        data = torch.from_numpy(np.load(pair[1]+'image.npy').astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        data_mask = torch.from_numpy(np.load(pair[1]+'tumor.npy').astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()

        input_seg, compute_mask = model.pre_register(fixed, data, seg2=data_mask)
        compute_mask = compute_mask.squeeze().cpu().numpy()
        mask = data_mask.squeeze().cpu().numpy()

        # print(sorted(list(set(mask.flatten()))))
        # print(sorted(list(set(compute_mask.flatten()))))

        print(compute_mask.shape)
        print(mask.shape)
        dice = dice_coefficient(mask,compute_mask)
        dices.append(dice)
        print(f'{pair[0]}:{dice}')

        means.append(compute_mask.mean())
        print('mean:', compute_mask.mean())

        input_seg = input_seg.squeeze().cpu().numpy()
        image = data.squeeze().cpu().numpy()

        input_seg = torch.from_numpy(input_seg.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(input_seg[::2],'input_seg.png',nrow=8,pad_value=1)
        compute_mask = torch.from_numpy(compute_mask.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(compute_mask[::2],'soft_mask.png',nrow=8,pad_value=1)
        mask = torch.from_numpy(mask.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(mask[::2],'mask.png',nrow=8,pad_value=1)
        image = torch.from_numpy(image.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(image[::2],'image.png',nrow=8,pad_value=1)
        fixed_mask = torch.from_numpy(fixed_mask.astype(np.float32)).transpose(1,2).transpose(0,1).unsqueeze(1)
        save_image(fixed_mask[::2],'fixed_mask.png',nrow=8,pad_value=1)
        assert 0

    plt.hist(dices, bins=10)
    plt.savefig('hist.png')

    print('mean dice:', np.mean(np.array(dices)))
    print('mean mean:', np.mean(np.array(means)))


def dice_coefficient(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def main():
    args = parser.parse_args()  
    args.amp = not args.noamp
    args.data_dir = "../../Data/" + args.dataset
    torch.autograd.set_detect_anomaly = True

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    args.test_mode = True
    # loader = get_loader(args)

    
    enhance_rate = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    fuse_rate = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    #enhance_rate[args.epos-1] = args.erate
    fuse_rate[3] = args.frate
    fuse_rate[6] = args.frate
    fuse_rate[9] = args.frate
    
    enhance_rate[3] = args.erate
    enhance_rate[6] = args.erate
    enhance_rate[9] = args.erate
    
    if (args.model_name is None) or args.model_name == "bntransformer":
        
        model = BNTransformer(
            in_channels=args.in_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            dropout_rate=args.dropout_rate,
            modality = args.modality,
            enhance_rate = enhance_rate,
            fuse_rate = fuse_rate,
            pretrained = args.pretrained,
            num_BN = args.num_BN,
            num_classes = args.num_classes,
            fusion_layer = [3,6,9,12],
            pretrained_model_name = args.pretrained_model_name,
        )

    elif args.model_name == "baseline":
        
        model = Baseline(
            in_channels=args.in_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            dropout_rate=args.dropout_rate,
            modality = args.modality,
            enhance_rate = enhance_rate,
            fuse_rate = fuse_rate,
            pretrained = args.pretrained,
            num_classes = args.num_classes,
            pretrained_model_name = args.pretrained_model_name,
        )

    elif args.model_name == "dmstransformer":
        
        model = DMSTransformer(
            in_channels=args.in_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            dropout_rate=args.dropout_rate,
            modality = args.modality,
            enhance_rate = enhance_rate,
            fuse_rate = fuse_rate,
            pretrained = args.pretrained,
            num_classes = args.num_classes,
            pretrained_model_name = args.pretrained_model_name,
            open_spatten = True,
            has_cls = False,
            open_mcatten = False,
        )

    elif args.model_name == "maskformer":
        
        model = MaskFormer(
            in_channels=args.in_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            dropout_rate=args.dropout_rate,
            modality = args.modality,
            enhance_rate = enhance_rate,
            fuse_rate = fuse_rate,
            pretrained = args.pretrained,
            num_classes = args.num_classes,
            pretrained_model_name = args.pretrained_model_name,
            has_cls = False,
        )

    elif args.model_name == "ViT_m":
        
        model = ViT_m(
            in_channels=args.in_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            dropout_rate=args.dropout_rate,
            modality = args.modality,
            enhance_rate = enhance_rate,
            fuse_rate = fuse_rate,
            pretrained = args.pretrained,
            num_classes = args.num_classes,
            pretrained_model_name = args.pretrained_model_name,
        )    
    
    elif args.model_name == "ViT_mask":

        model = ViT_mask(
            in_channels=args.in_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            dropout_rate=args.dropout_rate,
            modality = args.modality,
            enhance_rate = enhance_rate,
            fuse_rate = fuse_rate,
            pretrained = args.pretrained,
            num_classes = args.num_classes,
            pretrained_model_name = args.pretrained_model_name,
            has_cls = False,
        )

    elif args.model_name == "swin":

        model = SwinTransformer(
            in_chans=args.modality,
            embed_dim=48,
            window_size = (7,7,7),
            patch_size = (2,2,2),
            depths = [2, 2, 2, 2],
            num_heads = [3, 6, 12, 24],
            mlp_ratio = 4.0,
            qkv_bias = True,
            drop_rate = 0.0,
            attn_drop_rate = 0.0,
            drop_path_rate = 0.0,
            norm_layer = nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
        )

    elif args.model_name == "usmaskformer":
        
        model = USMaskFormer(
            in_channels=args.in_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            dropout_rate=args.dropout_rate,
            modality = args.modality,
            enhance_rate = enhance_rate,
            fuse_rate = fuse_rate,
            pretrained = args.pretrained,
            num_classes = args.num_classes,
            pretrained_model_name = args.pretrained_model_name,
            has_cls = False,
        )

    else:
        raise ValueError("Unsupported model " + str(args.model_name))
    

    avg_dice(model, dataset='brats2020')
    # avg_dice_song(model)
    return

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = {}
    for key in checkpoint["state_dict"]:
        if not (key.endswith("_ops") or key.endswith("_params")):
            state_dict[key] = checkpoint["state_dict"][key]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(device)

    start_time = time.time()
    with torch.no_grad():
        
        pred_all = []
        target_all = []
        
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            
            with autocast(enabled=args.amp):
                logits = model(data)
                    
            if not logits.is_cuda:
                target = target.cpu()
            
            if args.distributed:
                logits_list = distributed_all_gather([logits], out_numpy=False, is_valid=idx < loader.sampler.valid_length)
                logits = torch.cat(logits_list[0],dim=0)
                target_list = distributed_all_gather([target], out_numpy=False, is_valid=idx < loader.sampler.valid_length)
                target = torch.cat(target_list[0],dim=0)

            pred = logits.detach().cpu().numpy()
            pred_all.extend(pred.astype(float))
            target = target.detach().cpu().numpy()
            target_all.extend(target.astype(float))    
            
            if idx%5==0 and args.rank == 0:
                print(
                    "Val {}/{}".format(idx, len(loader)),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()  


        pred_all = np.stack(pred_all, axis=0)
        target_all = np.stack(target_all, axis=0)
        pred_all = np.argmax(pred_all, axis=1)
        target_all = np.argmax(target_all, axis=1)

        with open(os.path.join(args.data_dir, args.json_list),'r', encoding='UTF-8') as f:
            val_samples = json.load(f)['validation']
            val_samples = [sample['image'][-14:-4] for sample in val_samples]

        records = []
        for idx, tup in enumerate(zip(pred_all, target_all)):
            if(tup[0]!=tup[1]):
                record = val_samples[idx] + ' predict:' + str(tup[0]) + ' ' + 'truth:' + str(tup[1])
                records.append(record)
        
        jsondata = json.dumps(records,indent=4,separators=(',', ': '))
        f = open('./rcc500_'+args.model_name+'_record.json', 'w')
        f.write(jsondata)
        f.close()



if __name__ == "__main__":
    main()


# python test.py  --modality 4  --dataset RCC_500  --num_classes 3  --model_name baseline  --checkpoint ./runs/rcc500/baseline/2024-03-04_02-38-34/model.pt  --json_list data_wo_mask.json

# python test.py  --modality 4  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --checkpoint ./runs/brats2020/baseline/2024-02-03_18-41-29/model.pt  --json_list data_wo_mask.json

# python test.py  --modality 4  --num_classes 3  --model_name usmaskformer