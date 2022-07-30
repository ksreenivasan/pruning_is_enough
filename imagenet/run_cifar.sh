#!/bin/bash

#:<<BLOCK
CUDA_VISIBLE_DEVICES=1,2,3 python imagenet_main_cifar.py \
        --arch ResNet20 \
	--rank 0 \
	--batch-size 64 \
	--workers 8 \
	--mixed-precision \
	--epochs 10 \
	--lr 0.01 \
	--target-sparsity 5 \
	--iter-period 100 \
	--lmbda 0 \
    --subfolder results_cifar10_debug \
    --lr-schedule cosine_lr \
	--data '/home/ubuntu/ILSVRC2012' \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1

:<<BLOCK
CUDA_VISIBLE_DEVICES=1,2,3 python imagenet_main_cifar.py \
        --arch ResNet20 \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 \
	--rank 0 \
	--batch-size 64 \
	--workers 8 \
	--mixed-precision \
	--epochs 88 \
	--lr 0.01 \
	--finetune \
	--checkpoint 'model_before_finetune_epoch_9.pth' \
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
