#!/bin/bash

#SBATCH --account=plgiris-gpu-a100
#SBATCH --constraint=memfs
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --job-name=deit
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --time=2800

# setup
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate deit


# model_ckpt=$SCRATCH/deit_3_small_224_1k.pth
model_ckpt=$SCRATCH/compdeitsmall_ep800.pth

model=deit_small_patch16_LS
#com=run_with_submitit.py
com=main.py

# finetune cifar10
data_set=CIFAR10
data_path=$SCRATCH/datasets/cifar10
# tore
ouput_dir=transfer_tore_cifar10_from_compdeit800
#python $com --sample_divisions \
#    --finetune $model_ckpt --output_dir $ouput_dir --model $model --data-set $data_set --data-path $data_path --batch 256 --lr 1e-4 --epochs 100 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment &

# baseline
ouput_dir=transfer_cifar10_from_compdeit800
#python $com \
#    --finetune $model_ckpt --output_dir $ouput_dir --model $model --data-set $data_set --data-path $data_path --batch 256 --lr 1e-4 --epochs 100 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment &


# finetune cifar100
data_set=CIFAR100
data_path=$SCRATCH/datasets/cifar100
# tore
ouput_dir=transfer_tore_cifar100_from_compdeit800
#python $com --sample_divisions \
#    --finetune $model_ckpt --output_dir $ouput_dir --model $model --data-set $data_set --data-path $data_path --batch 256 --lr 1e-4 --epochs 100 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment &

# baseline
ouput_dir=transfer_cifar100_from_compdeit800
#python $com \
#    --finetune $model_ckpt --output_dir $ouput_dir --model $model --data-set $data_set --data-path $data_path --batch 256 --lr 1e-4 --epochs 100 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment &


# finetune flowers
data_set=FLOWERS
data_path=$SCRATCH/datasets/flowers
# tore
ouput_dir=transfer_tore_flowers_from_compdeit800
#python $com --sample_divisions \
#    --finetune $model_ckpt --output_dir $ouput_dir --model $model --data-set $data_set --data-path $data_path --batch 256 --lr 1e-4 --epochs 1000 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment &

# baseline
ouput_dir=transfer_flowers_from_compdeit800
python $com \
   --finetune $model_ckpt --output_dir $ouput_dir --model $model --data-set $data_set --data-path $data_path --batch 256 --lr 1e-4 --epochs 1000 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment &
wait
