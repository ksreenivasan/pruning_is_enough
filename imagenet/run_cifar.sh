#!/bin/bash

:<<BLOCK
python imagenet_main_cifar.py \
        --arch ResNet20 \
	--rank 0 \
	--batch-size 64 \
	--workers 8 \
	--mixed-precision \
	--epochs 150 \
	--lr 0.01 \
	--target-sparsity 1.44 \
	--iter-period 5 \
	--lmbda 0.0000001 \
    --subfolder 'cifar10_results' \
	--data '/home/ubuntu/ILSVRC2012' \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1
BLOCK

python imagenet_main_cifar.py \
        --arch ResNet20 \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 \
	--rank 0 \
	--batch-size 64 \
	--workers 8 \
	--mixed-precision \
	--epochs 150 \
	--lr 0.01 \
	--finetune \
    --subfolder 'cifar10_results' \
	--checkpoint 'cifar10_results/model_before_finetune_epoch_149.pth' \
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
