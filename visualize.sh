#!/bin/bash

# for ((VAR=1; VAR<=436; VAR++))
# do
#     IDX=$(echo "$VAR" | awk '{printf("%03d",$1)}')
#     DATAPTH="volume_${IDX}.npy"
#     echo $DATAPTH
#     python visualize2.py  --modality 4  --dataset RCC_500  --num_classes 3  --model_name baseline  --checkpoint ./runs/rcc500/baseline/2024-03-04_02-38-34/model.pt  --datapath $DATAPTH  --method gradcam
# done


for ((VAR=1; VAR<=369; VAR++))
do
    # IDX=$(echo "$VAR" | awk '{printf("%03d",$1)}')
    # DATAPTH="volume_${IDX}.npy"
    DATAPTH="volume_${VAR}.npy"
    echo $DATAPTH
    python visualize2.py  --modality 4  --dataset BraTs2020_training_data  --num_classes 2  --model_name maskformer  --checkpoint ./runs/brats2020/maskformer/2024-04-03_00-53-31/model.pt  --datapath $DATAPTH  --method gradcam
done

# python visualize2.py  --modality 4  --dataset RCC_500  --num_classes 3  --model_name maskformer  --checkpoint ./runs/rcc500/maskformer/2024-03-03_09-35-14/model.pt  --datapath volume_406.npy  --method gradcam