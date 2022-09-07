#!/bin/bash

#:<<BLOCK
:<<BLOCK
python imagenet_main.py \
    --arch ResNet50 \
	--rank 0 \
	--batch-size 512 \
	--workers 8 \
	--mixed-precision \
	--epochs 88 \
	--lr 0.1 \
	--target-sparsity 20 \
	--iter-period 8 \
    --optimizer adam \
	--lmbda 1e-8 \
	--wd 0 \
	--lr-schedule cosine_lr \
	--subfolder results_imagenet_nonaffine_adam_01_lr_fixed_threshold_sp20 \
	--data '/data/imagenet/' \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 #> debug_imagenet_log 2>&1 &
BLOCK

# finetune
# :<<BLOCK
python imagenet_main.py \
    --arch ResNet50 \
	--rank 0 \
	--batch-size 512 \
	--workers 8 \
	--mixed-precision \
	--epochs 88 \
	--lr 0.01 \
	--finetune \
	--iter-period 8 \
    --optimizer sgd \
	--lmbda 0 \
	--wd 1e-4 \
	--lr-schedule step_lr \
	--subfolder results_imagenet_nonaffine_adam_01_lr_fixed_threshold \
	--checkpoint model_before_finetune_epoch_87.pth \
	--data '/data/imagenet/' \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 #> debug_imagenet_log 2>&1 &

# BLOCK

:<<BLOCK
python imagenet_main.py \
        --arch ResNet50 \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 \
	--rank 0 \
	--batch-size 512 \
	--workers 8 \
	--mixed-precision \
	--epochs 88 \
	--lr 0.002 \
	--finetune \
	--checkpoint 'results_reg_1e-10_lr_0.02_iter_8_fullrun/model_before_finetune_epoch_87.pth' \
	--lmbda 0.0 \
	--data '/home/ubuntu/ILSVRC2012'
BLOCK

:<<BLOCK
python imagenet_main_bkp.py \
	--arch resnet50 \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 \
	--rank 0 \
	--batch-size 1024 \
	--workers 8 \
	--mixed-precision \
	--epochs 5 \
	--lr 0.4 \
	--data '/home/ubuntu/ILSVRC2012'
BLOCK
