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

from visualizer import get_local
get_local.activate()

from networks import BNTransformer, Baseline, DMSTransformer, MaskFormer
from utils.data_utils import get_loader

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description="visualize parameters")

#general settings
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--modality", default=4, type=int, help="number of modalities")
#parser.add_argument("--modality", default=0, type=int, help="index of modalities")
parser.add_argument("--datapath", default="volume_28", type=str, help="sample to load")

parser.add_argument("--dataset", default="RCC_500", type=str, help="choose dataset")
parser.add_argument("--num_classes", default=3, type=int, help="number of classes")
parser.add_argument("--num_BN", default=4, type=int, help="number of bottleneck tokens")

parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory")
parser.add_argument("--data_dir", default="../../Data/BraTs2020_training_data", type=str, help="dataset directory")
parser.add_argument("--json_list", default="data.json", type=str, help="dataset json file")
parser.add_argument("--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name")
parser.add_argument("--pretrained", action="store_true", help="use pretrained UNETR parameters")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--epos", default=1, type=int, help="enhance layer index")
parser.add_argument("--erate", default=0.0, type=float, help="enhance rate")
parser.add_argument("--frate", default=0.0, type=float, help="enhance rate")
parser.add_argument("--open_spatten", action="store_true", help="open same pos attention")
parser.add_argument("--open_mcatten", action="store_true", help="open masked cross attention")
parser.add_argument("--has_cls", action="store_true", help="use cls when fuse multi modal")
#training settings
parser.add_argument("--max_epochs", default=300, type=int, help="max number of training epochs")
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
parser.add_argument("--model_name", default="dmstransformer", type=str, help="model name")
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



def grid_show(to_shows, cols):
    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
    plt.show()

def visualize_head(att_maps, layer_idx, head_idx, savepath):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_maps[layer_idx-1][0,head_idx-1])
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.savefig('./pics/base_stagefuse_mca/'+savepath[:-4]+f'/layer_{layer_idx}_head_{head_idx}'+'.png')
    plt.close()

def visualize_layer(att_maps, layer_idx, savepath):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_maps[layer_idx-1].mean(axis=1)[0])
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.savefig('./pics/base_stagefuse_mca/'+savepath[:-4]+f'/layer_{layer_idx}'+'.png')
    plt.close()

def visualize_attn_map(att_map, args):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    if not os.path.exists('./pics/'+args.dataset+'/'+args.model_name+'/'+args.datapath[:-4]+'/'):
        os.makedirs('./pics/'+args.dataset+'/'+args.model_name+'/'+args.datapath[:-4]+'/')
    plt.savefig('./pics/'+args.dataset+'/'+args.model_name+'/'+args.datapath[:-4]+'/attn_map.png')
    plt.close()
    
def visualize_heads(att_map, cols):
    to_shows = []
    att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    grid_show(to_shows, cols=cols)

def gray2rgb(image):
    return np.repeat(image[...,np.newaxis],3,2)
    

def visualize_grid_to_grid(att_map, image, args, grid_size=8, alpha=0.5, modality=4):

    NUM = att_map.size//modality
    fig, ax = plt.subplots(grid_size, modality, figsize=(30,30))
    fig.tight_layout()

    for idx in range(modality):
        mask = att_map[idx*NUM:idx*NUM+NUM].reshape(grid_size, grid_size, grid_size)
        for layer in range(grid_size):
            image_layer = image[idx,:,:,layer*16+8].numpy()
            image_layer = (image_layer-np.mean(image_layer))/(np.std(image_layer))
            mask_layer = Image.fromarray(mask[:,:,layer]).resize(image_layer.shape)
            
            plt.subplot(grid_size, modality, layer*modality+idx+1)
            plt.imshow(image_layer, cmap='gray')
            plt.imshow(mask_layer/np.max(mask), alpha=alpha, cmap='rainbow', vmin=0.0, vmax=1.0)
            plt.colorbar()
    
    plt.savefig('./pics/'+args.dataset+'/'+args.model_name+'/'+args.datapath[:-4]+'/'+args.datapath[:-4]+'.png')
    plt.close()
    

def visualize_image(image, args, grid_size=8, modality=4):
    
    fig, ax = plt.subplots(grid_size, modality, figsize=(30,30))
    fig.tight_layout()

    for idx in range(modality):
        for layer in range(grid_size):
            image_layer = image[idx,:,:,layer*16+8].numpy()
            image_layer = (image_layer-np.mean(image_layer))/(np.std(image_layer))
            
            plt.subplot(grid_size, modality, layer*modality+idx+1)
            plt.imshow(image_layer, cmap='gray')
    
    plt.savefig('./pics/'+args.dataset+'/'+args.model_name+'/'+args.datapath[:-4]+'/original.png')
    plt.close()




def main():
    args = parser.parse_args()
    args.test_mode = True

    datapath = args.datapath
    
    data_validation = torch.from_numpy(np.load("../../Data/"+args.dataset+"/data_processed/"+datapath)).unsqueeze(0) # (1,4,128,128,128)

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

    else:
        raise ValueError("Unsupported model " + str(args.model_name))



    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = {}
    for key in checkpoint["state_dict"]:
        if not (key.endswith("_ops") or key.endswith("_params")):
            state_dict[key] = checkpoint["state_dict"][key]
    model.load_state_dict(state_dict, strict=True)


    model.eval()
    with torch.no_grad():
        logits = model(data_validation)

    # with open("../../Data/"+args.dataset+"/data_4classes.json", 'r') as f:
    #     dict = json.load(fp=f)
    #     target = torch.Tensor(dict['training'][-2]['label']).unsqueeze(0)
    
    target = torch.Tensor([0.0, 0.0, 1.0]).unsqueeze(0)

    print(f'loss:{nn.CrossEntropyLoss()(logits,target)}')
    cache = get_local.cache
    print(len(cache['SABlock.forward']))  #len=4
    print(cache['SABlock.forward'][-1].shape) #shape=(1,12,2053,2053)

    # assert 0

    attention_maps = cache['SABlock.forward']

    # print(attention_maps[0].mean(axis=1).shape)

    # for layer_idx in range(1,5):
    #         visualize_layer(attention_maps,layer_idx,datapath)

    # save_image(data_validation[:,:,:,:,64].view(4,128,128).unsqueeze(1),'tt.png',nrow=1,pad_value=1)
    
    # NxN, 注意力矩阵可视化
    cls_attn = attention_maps[-1].mean(axis=1)[0]
    visualize_attn_map(cls_attn,args)

    # N, 热力图可视化
    cls_attn = cls_attn.mean(axis=0)
    visualize_grid_to_grid(cls_attn, data_validation[0], args)                

    visualize_image(data_validation[0], args) 
    

if __name__ == "__main__":
    main()


# python visualize.py  --modality 4  --dataset RCC_500  --num_classes 3  --model_name maskformer  --checkpoint ./runs/rcc500/BNTransformer/2024-03-03_09-35-14/model.pt  --datapath volume_696.npy