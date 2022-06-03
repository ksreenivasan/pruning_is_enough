#!/bin/bash

#:<<BLOCK
python imagenet_main_cifar.py \
        --arch ResNet50 \
	--rank 0 \
	--batch-size 64 \
	--workers 8 \
	--mixed-precision \
	--epochs 10 \
	--lr 0.01 \
	--target-sparsity 5 \
	--iter-period 5 \
	--lmbda 0.0000001 \
	--data '/home/ubuntu/ILSVRC2012' \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1

python imagenet_main_cifar.py \
        --arch ResNet50 \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 \
	--rank 0 \
	--batch-size 64 \
	--workers 8 \
	--mixed-precision \
	--epochs 88 \
	--lr 0.004 \
	--finetune \
	--checkpoint 'model_before_finetune_epoch_9.pth' \
	--lmbda 0.0 \
	--data '/home/ubuntu/ILSVRC2012'
#BLOCK

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
