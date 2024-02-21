#!/bin/bash

#SBATCH --account=plgiris-gpu-a100
#SBATCH --constraint=memfs
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --job-name=ame
#SBATCH --mem=60GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=2800

# setup
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate wtln

MAE_PATH=/net/pr2/projects/plgrid/plggicv/mae_pretrain_vit_base.pth

#SUN360_DIR=/net/pr2/projects/plgrid/plggicv/datasets/sun360
#python train.py Sun360Classification AttentionClsMae --data-dir $SUN360_DIR --pretrained-mae-path $MAE_PATH --no-wandb --tensorboard --ddp

IMAGENET_DIR=/net/pr2/projects/plgrid/plggicv/datasets/imagenet
python train.py ImageNet1kClassification AttentionClsMae --data-dir $IMAGENET_DIR --pretrained-mae-path $MAE_PATH --no-wandb --tensorboard --ddp

#FMOV_DIR=/net/pr2/projects/plgrid/plggicv/datasets/fmow/baseline/data/
#python train.py FMoVClassification AttentionClsMae --data-dir $FMOV_DIR --pretrained-mae-path $MAE_PATH --no-wandb --tensorboard --ddp
