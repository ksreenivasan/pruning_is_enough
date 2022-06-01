#!/bin/bash

#:<<BLOCK
python imagenet_main.py \
        --arch ResNet50 \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 \
	--rank 0 \
	--batch-size 1024 \
	--workers 8 \
	--mixed-precision \
	--epochs 3 \
	--lr 0.4 \
	--finetune \
	--lmbda 0.000000 \
	--data '/home/ubuntu/ILSVRC2012' #> "resnet50_imagenet_log" 2>&1 &
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
