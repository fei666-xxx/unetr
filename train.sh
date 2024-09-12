#! /bin/bash

# python main.py  --modality 4  --distributed  --pretrained  --erate 0  --frate 0  --logdir base_maskCA
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0.0  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/e005
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.0  --frate 0.0  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/base
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0.3  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/e005_f03_4710_new
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0.3  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/e005_f03_4710_CSA
# python main.py  --modality 4  --distributed  --pretrained  --erate 0  --frate 0  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/base_mca

# ----------------------------------------------------------------------------------------
# 2023-09-15

# python main.py  --modality 4  --distributed  --pretrained  --erate 0  --frate 0  --optim_lr 0.00001 --max_epochs 300 --logdir pickup/base_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0.3  --optim_lr 0.00001 --max_epochs 300 --logdir pickup/e005_f03_4710_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/e005_4710_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0.1  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/e005_f01_4710_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0.2  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/e005_f02_4710_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0.4  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/e005_f04_4710_mca


# ----------------------------------------------------------------------------------------
# 2023-09-21

# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0  --optim_lr 0.00001 --max_epochs 300 --logdir pickup/e005_4710_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.05  --frate 0.1  --optim_lr 0.00001 --max_epochs 300 --logdir pickup/e005_f01_4710_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.0  --frate 0.0  --optim_lr 0.00001 --max_epochs 300  --open_spatten --logdir lre_5/base_spatten_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0.0  --frate 0.0  --optim_lr 0.00001 --max_epochs 300  --open_spatten --logdir lre_5/base_spatten2_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0  --frate 0  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/base_stagefuse


# ----------------------------------------------------------------------------------------
# 2023-10-07
# python main.py  --modality 4  --distributed  --pretrained  --erate 0  --frate 0  --optim_lr 0.00001 --max_epochs 300 --logdir pickup/base_stagefuse
# python main.py  --modality 4  --distributed  --pretrained  --erate 0  --frate 0  --optim_lr 0.00001 --max_epochs 300 --logdir lre_5/base_stagefuse
# python main.py  --modality 4  --distributed  --pretrained  --erate 0  --frate 0  --optim_lr 0.00001 --max_epochs 300 --open_spatten --logdir lre_5/base_stagefuse_spa_mca
# python main.py  --modality 4  --distributed  --pretrained  --erate 0  --frate 0  --optim_lr 0.00001 --max_epochs 300 --open_spatten --logdir lre_5/base_stagefuse_spa


# ----------------------------------------------------------------------------------------
# 为了方便，以下省略 --erate 0  --frate 0  --optim_lr 0.00001 --max_epochs 300
# ----------------------------------------------------------------------------------------
# python main.py  --modality 4  --distributed  --pretrained   --open_spatten --open_mcatten --has_cls --logdir lre_5/base_stagefuse_spa_mca

# 前面只有一个spa，这里也用四个,每个阶段不同
# python main.py  --modality 4  --distributed  --pretrained   --open_spatten --open_mcatten --has_cls --logdir lre_5/base_stagefuse_mca_spa

# ----------------------------------------------------------------------------------------
#开题后
#新增LGG_precision、LGG_recall、HGG_precision、HGG_recall评价指标
# ----------------------------------------------------------------------------------------
# python main.py  --modality 4  --distributed  --pretrained  --has_cls --open_mcatten --save_checkpoint --logdir share_backbone/base+stage
# python main.py  --modality 4  --distributed  --pretrained  --has_cls --open_mcatten --save_checkpoint --logdir share_backbone/base+stage_4branch

# python main.py  --modality 4  --distributed  --pretrained  --has_cls --open_mcatten --open_spatten --save_checkpoint --logdir lre_5/base_stagefuse_mca_spa
# python main.py  --modality 4  --distributed  --pretrained  --has_cls --open_mcatten --save_checkpoint --logdir lre_5/base_stagefuse_mca

# spatten γ=4
# python main.py  --modality 4  --distributed  --pretrained  --has_cls --open_mcatten --open_spatten --save_checkpoint --logdir lre_5/base_stagefuse_mca_spa

# visualize!!!
# python visualize.py  --modality 4   --has_cls --open_mcatten --save_checkpoint --checkpoint ./runs/lre_5/base_stagefuse_mca/2023-12-07_13-19-34/model.pt  --datapath volume_103.npy
# python visualize.py  --modality 4   --has_cls --open_mcatten --save_checkpoint --checkpoint ./runs/lre_5/base_stagefuse_mca/2023-12-07_13-19-34/model.pt  --datapath volume_166.npy
# python visualize.py  --modality 4   --has_cls --open_mcatten --save_checkpoint --checkpoint ./runs/lre_5/base_stagefuse_mca/2023-12-07_13-19-34/model.pt  --datapath volume_186.npy
# python visualize.py  --modality 4   --has_cls --open_mcatten --save_checkpoint --checkpoint ./runs/lre_5/base_stagefuse_mca/2023-12-07_13-19-34/model.pt  --datapath volume_240.npy
# python visualize.py  --modality 4   --has_cls --open_mcatten --save_checkpoint --checkpoint ./runs/lre_5/base_stagefuse_mca/2023-12-07_13-19-34/model.pt  --datapath volume_251.npy
# python visualize.py  --modality 4   --has_cls --open_mcatten --save_checkpoint --checkpoint ./runs/lre_5/base_stagefuse_mca/2023-12-07_13-19-34/model.pt  --datapath volume_258.npy




# ----------------------------------------------------------------------------------------
# 2024-01-25
# BNTransformer 系列
# ----------------------------------------------------------------------------------------


# rcc500, num_BN=4, independent backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 4  --model_name bntransformer  --logdir BNTransformer

# rcc500, baseline, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 4  --model_name baseline  --logdir baseline

# rcc500, num_BN=4, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 4  --model_name bntransformer  --logdir BNTransformer


# brats2020, baseline, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir brats2020/baseline

# brats2020, num_BN=4, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name bntransformer  --logdir brats2020/BNTransformer

# brats2020, num_BN=2, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name bntransformer  --logdir brats2020/BNTransformer  --num_BN 2

# brats2020, num_BN=8, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name bntransformer  --logdir brats2020/BNTransformer  --num_BN 8

# brats2020, num_BN=16, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name bntransformer  --logdir brats2020/BNTransformer  --num_BN 16

# rcc500, num_BN=2, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 4  --model_name bntransformer  --logdir rcc500/BNTransformer  --num_BN 2

# rcc500, num_BN=8, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 4  --model_name bntransformer  --logdir rcc500/BNTransformer  --num_BN 8

# rcc500, num_BN=16, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 4  --model_name bntransformer  --logdir rcc500/BNTransformer  --num_BN 16


#---------------------
# imagenet pretrained
#---------------------

# # rcc500, baseline, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir rcc500/baseline 

# # rcc500, num_BN=4, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name bntransformer  --logdir rcc500/BNTransformer   

# # brats2020, baseline, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir brats2020/baseline  

# # brats2020, num_BN=4, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name bntransformer  --logdir brats2020/BNTransformer  

# rcc500, dmstransformer(open_spatten), common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name dmstransformer  --logdir rcc500/dmstransformer 

# rcc500, maskformer, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name maskformer  --logdir rcc500/maskformer


#---------------------
# back to UNETR pretrained
#---------------------

# rcc500, maskformer, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name maskformer  --logdir rcc500/maskformer


# rcc500, baseline, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir rcc500/baseline


# rcc500, maskformer(hard=false), common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name maskformer  --logdir rcc500/maskformer


# rcc500, ViT_m,
# python main.py  --modality 0  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name ViT_mask  --logdir rcc500/ViT_mask
# python main.py  --modality 1  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name ViT_mask  --logdir rcc500/ViT_mask
# python main.py  --modality 2  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name ViT_mask  --logdir rcc500/ViT_mask
# python main.py  --modality 3  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name ViT_mask  --logdir rcc500/ViT_mask


# rcc500_tumor, baseline,
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir rcc500/baseline  --json_list data_tumor.json


# brats2020, maskformer, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name maskformer  --logdir brats2020/maskformer  --json_list  data_with_mask.json


# brats2020_tumor, baseline,
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir brats2020/baseline  --json_list data_tumor.json


# brats2020, ViT_m,
# python main.py  --modality 0  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name ViT_mask  --logdir brats2020/ViT_mask  --json_list data_wo_mask.json
# python main.py  --modality 1  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name ViT_mask  --logdir brats2020/ViT_mask  --json_list data_wo_mask.json
# python main.py  --modality 2  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name ViT_mask  --logdir brats2020/ViT_mask  --json_list data_wo_mask.json
# python main.py  --modality 3  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name ViT_mask  --logdir brats2020/ViT_mask  --json_list data_wo_mask.json


# brats2020, MIL, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir brats2020/MIL  --json_list  data_wo_mask.json


# rcc500, new-maskformer(mask*raw), common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name maskformer  --logdir rcc500/maskformer  --json_list data_with_mask.json  --optim_lr 0.0001

# brats2020, new-maskformer(mask*raw), common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name maskformer  --logdir brats2020/maskformer  --json_list data_with_mask.json  --optim_lr 0.00001

# rcc500, new-maskformer(mask*raw+raw), common backbone
# brats2020, new-maskformer(mask*raw+raw), common backbone

# brats2020, baseline, common backbone, --optim_lr 0.00001
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir brats2020/baseline  --json_list data_wo_mask.json  --optim_lr 0.00001

# brats2020, baseline, common backbone, --optim_lr 0.0001
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir brats2020/baseline  --json_list data_wo_mask.json  --optim_lr 0.0001

# brats2020, new-maskformer(mask*raw), common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name maskformer  --logdir brats2020/maskformer  --json_list data_with_mask.json

# rcc500, new-maskformer(mask*raw+raw, with noise), common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name maskformer  --logdir rcc500/maskformer  --json_list data_with_mask.json

# brats2020, new-maskformer(mask*raw+raw, with noise), common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name maskformer  --logdir brats2020/maskformer  --json_list data_with_mask.json

# brats2020, baseline, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir brats2020/baseline  --json_list data_wo_mask.json

# rcc500, usmaskformer(mask*raw+raw, soft mask), common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name usmaskformer  --logdir rcc500/usmaskformer  --json_list data_with_mask.json

# rcc500, usmaskformer(mask*raw+raw, computed organ mask), common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name usmaskformer  --logdir rcc500/usmaskformer  --json_list data_with_mask.json



# -------------中期实验-------------
# brats2020, baseline, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir mid/brats2020/baseline  --json_list data_wo_mask.json

# brats2020, enhance, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir mid/brats2020/enhance  --erate 0.025  --json_list data_wo_mask.json  --batch_size 16    
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir mid/brats2020/enhance  --erate 0.05  --json_list data_wo_mask.json  --batch_size 8
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir mid/brats2020/enhance  --erate 0.1  --json_list data_wo_mask.json  --batch_size 4

# brats2020, enhance 0.05, fuse, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir mid/brats2020/fuse01  --erate 0.05  --frate 0.1 --json_list data_wo_mask.json  --batch_size 16
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir mid/brats2020/fuse02  --erate 0.05  --frate 0.2 --json_list data_wo_mask.json  --batch_size 16
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir mid/brats2020/fuse03  --erate 0.05  --frate 0.3 --json_list data_wo_mask.json  --batch_size 16
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir mid/brats2020/fuse04  --erate 0.05  --frate 0.4 --json_list data_wo_mask.json  --batch_size 16
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset BraTs2020_training_data  --num_classes 2  --model_name baseline  --logdir mid/brats2020/fuse05  --erate 0.05  --frate 0.5 --json_list data_wo_mask.json  --batch_size 16




# rcc500, enhance 0.05, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse01  --erate 0.05  --frate 0.1 --json_list data_wo_mask.json  --batch_size 16
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse02  --erate 0.05  --frate 0.2 --json_list data_wo_mask.json  --batch_size 16
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse03  --erate 0.05  --frate 0.3 --json_list data_wo_mask.json  --batch_size 16
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse04  --erate 0.05  --frate 0.4 --json_list data_wo_mask.json  --batch_size 16
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse05  --erate 0.05  --frate 0.5 --json_list data_wo_mask.json  --batch_size 16

# rcc500, baseline, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/baseline  --json_list data_wo_mask.json

# rcc500, enhance, common backbone
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/enhance  --erate 0.025  --json_list data_wo_mask.json  --batch_size 16
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/enhance  --erate 0.05  --json_list data_wo_mask.json  --batch_size 8
# python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/enhance  --erate 0.1  --json_list data_wo_mask.json  --batch_size 4

# rcc500, enhance 0.025, common backbone
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse01  --erate 0.025  --frate 0.1 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse01  --erate 0.025  --frate 0.1 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse01  --erate 0.025  --frate 0.1 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse01  --erate 0.025  --frate 0.1 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse01  --erate 0.025  --frate 0.1 --json_list data_wo_mask.json  --batch_size 16

python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse02  --erate 0.025  --frate 0.2 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse02  --erate 0.025  --frate 0.2 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse02  --erate 0.025  --frate 0.2 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse02  --erate 0.025  --frate 0.2 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse02  --erate 0.025  --frate 0.2 --json_list data_wo_mask.json  --batch_size 16

python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse03  --erate 0.025  --frate 0.3 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse03  --erate 0.025  --frate 0.3 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse03  --erate 0.025  --frate 0.3 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse03  --erate 0.025  --frate 0.3 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse03  --erate 0.025  --frate 0.3 --json_list data_wo_mask.json  --batch_size 16

python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse04  --erate 0.025  --frate 0.4 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse04  --erate 0.025  --frate 0.4 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse04  --erate 0.025  --frate 0.4 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse04  --erate 0.025  --frate 0.4 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse04  --erate 0.025  --frate 0.4 --json_list data_wo_mask.json  --batch_size 16

python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse05  --erate 0.025  --frate 0.5 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse05  --erate 0.025  --frate 0.5 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse05  --erate 0.025  --frate 0.5 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse05  --erate 0.025  --frate 0.5 --json_list data_wo_mask.json  --batch_size 16
python main.py  --modality 4  --distributed  --pretrained  --save_checkpoint  --dataset RCC_500  --num_classes 3  --model_name baseline  --logdir mid/rcc500/fuse05  --erate 0.025  --frate 0.5 --json_list data_wo_mask.json  --batch_size 16
