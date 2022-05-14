#!/bin/bash

python imagenet_main_bkp.py \
       	-a resnet50 \
	--dist-url 'tcp://127.0.0.1:2500' \
	--dist-backend 'nccl' \
	--multiprocessing-distributed \
	--world-size 1 \
	--rank 0 \
	--batch-size 1024 \
	--workers 12 \
	--mixed-precision \
	--data '/home/ubuntu/ILSVRC2012' > "resnet50_imagenet_log" 2>&1 &

