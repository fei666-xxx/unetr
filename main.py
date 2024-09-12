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


from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader

from thop import profile

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
parser.add_argument("--frate", default=0.0, type=float, help="fusion rate")
# parser.add_argument("--open_spatten", action="store_true", help="open same pos attention")
# parser.add_argument("--open_mcatten", action="store_true", help="open masked cross attention")
parser.add_argument("--num_classes", default=4, type=int, help="number of classes")
parser.add_argument("--num_BN", default=4, type=int, help="number of bottleneck tokens")

#training settings
parser.add_argument("--max_epochs", default=200, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
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

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

def main():

    # for name, param in MaskFormer.named_parameters():
    #     print(name, '      ', param.size())


    args = parser.parse_args()  
    args.amp = not args.noamp
    TIMESTAMP = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.now())
    args.logdir = "./runs/" + args.logdir + "/" + TIMESTAMP
    args.data_dir = "../../Data/" + args.dataset
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=7, args=args)  


def main_worker(gpu, args):
    torch.autograd.set_detect_anomaly = True
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    pretrained_dir = args.pretrained_dir
    
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

        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict)
            print("Use pretrained weights")

        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            
            
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
    
    loss = nn.CrossEntropyLoss()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    

    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "f1score" in checkpoint:
            f1score = checkpoint["f1score"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestF1Score {})".format(args.checkpoint, start_epoch, f1score))

    # model.init(args.gpu)
    model.cuda(args.gpu)

    input = torch.randn(args.batch_size, args.modality, args.roi_x, args.roi_y, args.roi_z).cuda(args.gpu)
    flops, _ = profile(model, inputs=(input,))
    

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
        
     
    #for name, param in model.named_parameters():
	      #print(name, '      ', param.size())
    
        
        
    metrics = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=loss,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
    )
    
    if args.rank == 0:
        log_stats1 = {
                        'dataset':args.dataset,
                        'model':args.model_name,
                        'epoch':args.max_epochs,
                        'batch_size':args.batch_size,
                        'num_heads':args.num_heads,
                        'mlp_dim':args.mlp_dim,
                        'hidden_size':args.hidden_size,
                        'num_layers':len(enhance_rate),
                        'modality':args.modality,
                        'pretrained':args.pretrained,
                        'num_classes':args.num_classes,
                        'num_BN':args.num_BN,}
        
        log_stats2 = {
                        'fuse_rate':fuse_rate,
                        'enhance_rate':enhance_rate,}
        
        metrics.update(
                    {
                        'num_params':pytorch_total_params,
                        'flops':flops,
                    }
        )

        with open("./log.txt", "a") as f:
            f.write(json.dumps(log_stats1) + "\n")
            f.write(json.dumps(log_stats2) + "\n")
            f.write(json.dumps(metrics) + "\n\n")

        
if __name__ == "__main__":
    main()
