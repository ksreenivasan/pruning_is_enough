#!/bin/bash

#:<<BLOCK
python imagenet_main.py \
        --arch WideResNet50_2 \
	--rank 0 \
	--batch-size 256 \
	--workers 8 \
	--mixed-precision \
	--epochs 88 \
	--lr 0.0256 \
	--target-sparsity 20 \
    --weight-decay 0.000030517578125 \
    --momentum 0.875 \
	--iter-period 100 \
    --optimizer sgd \
	--lmbda 0 \
	--lr-schedule cosine_lr \
	--subfolder results_wideresnet_reg_0_no_iter_sgd_cosine_lr \
	--data '/data/imagenet/' \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1

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
