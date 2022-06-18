#!/bin/bash

#:<<BLOCK
python imagenet_main.py \
        --arch ResNet50 \
	--rank 0 \
	--batch-size 512 \
	--workers 8 \
	--mixed-precision \
	--epochs 88 \
	--lr 0.001 \
	--target-sparsity 20 \
	--iter-period 5 \
	--lmbda 0.0000000001 \
	--subfolder results_reg_1e-10_lr_0.01_iter_5_fullrun \
	--data '/data/imagenet/' \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1

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
    --subfolder results_reg_1e-10_lr_0.01_iter_5_fullrun \
    --checkpoint 'results_reg_1e-10_lr_0.01_iter_5_fullrun/model_before_finetune_epoch_87.pth' \
	--lmbda 0.0 \
	--data '/data/imagenet'

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
