import sys
import os

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-1])
root_path += '/cam/'
sys.path.append(root_path)


import argparse
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


from networks import BNTransformer, Baseline, DMSTransformer, MaskFormer
from utils.data_utils import get_loader

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import cv2
from cam.pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from cam.pytorch_grad_cam import GuidedBackpropReLUModel
# from cam.pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from cam.pytorch_grad_cam.ablation_layer import AblationLayerVit
import nibabel as nib 

parser = argparse.ArgumentParser(description="visualize parameters")

#general settings
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--modality", default=4, type=int, help="number of modalities")
#parser.add_argument("--modality", default=0, type=int, help="index of modalities")

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

parser.add_argument("--datapath", default="volume_28", type=str, help="sample to load")
parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet of cam_weights*activations')
parser.add_argument('--method', type=str, default='gradcam', help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')


def reshape_transform(tensor, modality=4, height=8, width=8, thickness=8):
    # tensor.shape 1x2052x768
    # print("tensor:", tensor.shape)
    result = tensor.reshape(tensor.size(0),modality,-1, tensor.size(2))
    # result = result[:,:,1:,:].reshape(tensor.size(0),modality,
    #                                   height, width, thickness, tensor.size(2))

    result = result.reshape(tensor.size(0),modality,
                                      height, width, thickness, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(4, 5).transpose(3, 4).transpose(2, 3).transpose(1,2)
    return result


def show_cam_on_image(cam, image, savepath, args, grid_size=8, alpha=0.5, modality=4, savename='/gradcam.png'):

    fig, ax = plt.subplots(grid_size, modality, figsize=(30,30))
    fig.tight_layout()

    for idx in range(modality):
        for layer in range(grid_size):
            image_layer = image[idx,:,:,layer*16+8]
            image_layer = (image_layer-np.mean(image_layer))/(np.std(image_layer))
            mask_layer = cam[idx,:,:,layer*16+8]
            
            plt.subplot(grid_size, modality, layer*modality+idx+1)
            plt.imshow(image_layer, cmap='gray')
            # plt.imshow(mask_layer/np.max(cam), alpha=alpha, cmap='rainbow', vmin=0.0, vmax=1.0)
            plt.imshow(mask_layer, alpha=alpha, cmap='rainbow', vmin=0.0, vmax=1.0)
            plt.colorbar()
    
    plt.savefig(savepath + savename)
    plt.close()
    

def show_image(image, savepath, args, grid_size=8, modality=4, savename='/original.png'): # 4 128 128 128
    
    fig, ax = plt.subplots(grid_size, modality, figsize=(30,30))
    fig.tight_layout()

    for idx in range(modality):
        for layer in range(grid_size):
            image_layer = image[idx,:,:,layer*16+8]
            image_layer = (image_layer-np.mean(image_layer))/(np.std(image_layer))
            
            plt.subplot(grid_size, modality, layer*modality+idx+1)
            plt.imshow(image_layer, cmap='gray')
    
    plt.savefig(savepath + savename)
    plt.close()


def main():
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    args.test_mode = True


    datapath = args.datapath
    raw_image = np.load("../../Data/"+args.dataset+"/data_processed_with_mask/"+datapath)
    input_tensor = torch.from_numpy(raw_image).unsqueeze(0) # (1,4,128,128,128)

    # raw_images = []
    # for i in range(436):
    #     raw_image = np.load("../../Data/"+args.dataset+"/data_processed_with_mask/volume_{:0=3d}.npy".format(i+1))
    #     raw_image = torch.from_numpy(raw_image)
    #     raw_images.append(raw_image)
    # input_tensor = torch.stack(raw_images, dim=0)
    # print(input_tensor.shape)
    # assert 0

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


    target_layers = [model.fusion.norm1]


    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    
    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                #    use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform)


    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = None

    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)


    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    savepath = './pics/'+args.dataset+'/'+args.model_name+'/'+args.datapath[:-4]+'/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    show_image(raw_image, savepath, args)
    show_cam_on_image(grayscale_cam, raw_image, savepath, args)

    # for idx in range(args.modality):
    #     new_image = nib.Nifti1Image(grayscale_cam[idx], np.eye(4))
    #     nib.save(new_image, "tumor.nii.gz")





if __name__ == "__main__":
    main()


# python visualize2.py  --modality 4  --dataset RCC_500  --num_classes 3  --model_name baseline  --checkpoint ./runs/rcc500/baseline/2024-03-04_02-38-34/model.pt  --datapath volume_696.npy  --method gradcam
# python visualize2.py  --modality 4  --dataset RCC_500  --num_classes 3  --model_name maskformer  --checkpoint ./runs/rcc500/maskformer/2024-03-03_09-35-14/model.pt  --datapath $DATAPTH  --method gradcam
