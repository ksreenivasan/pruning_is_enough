import pdb
import numpy as np
import os
import pathlib
import random
import time
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp

import sys
import re


import copy
import time
import torch
# import tqdm
import copy
import pdb

from torch import optim
import psutil, sys
from args_helper import parser_args
import psutil


def main():
    print(parser_args)
    print("\n\nBeginning of process.")
    print_time()
    set_seed(parser_args.seed * parser_args.trial_num)
    # set_seed(parser_args.seed + parser_args.trial_num - 1)

    # world size = ngpus_per_node since we are assuming single node
    ngpus_per_node = torch.cuda.device_count()

    if parser_args.multiprocessing_distributed:
        # assert ngpus_per_node >= 2, f"Requires at least 2 GPUs to run, but got {ngpus_per_node}"
        mp.spawn(main_worker, args=(ngpus_per_node,), nprocs=ngpus_per_node, join=True)
    else:
        # Simply call main_worker function
        main_worker(parser_args.gpu, ngpus_per_node)


# set seed for experiment
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # set this=True if you want deterministic runs
    torch.backends.cudnn.deterministic = False
    # set this=False if you want deterministic runs
    torch.backends.cudnn.benchmark = True
    print("Seeded everything: {}".format(seed))


def setup_distributed(rank, ngpus_per_node):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '{}'.format(parser_args.port)

    dist.init_process_group("nccl", rank=rank, world_size=ngpus_per_node)

def main_worker(gpu, ngpus_per_node):
    parser_args.gpu = gpu
    if parser_args.multiprocessing_distributed:
        parser_args.rank = parser_args.gpu
        setup_distributed(parser_args.rank, ngpus_per_node)
        # if using ddp, divide batch size per gpu
        parser_args.batch_size = int(parser_args.batch_size / ngpus_per_node)


    #########################DATA LOADING CODE#########################
    from torchvision import datasets, transforms
    import torch.multiprocessing
    from torch.utils.data import random_split
    torch.multiprocessing.set_sharing_strategy("file_system")
    data_root = parser_args.data

    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(
                            traindir,
                            transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    if parser_args.use_full_data:
        train_dataset = dataset
        # use_full_data => we are not tuning hyperparameters
        validation_dataset = test_dataset
    else:
        # train_size = 1000
        # val_size = len(dataset) - train_size
        val_size = 10000
        train_size = len(dataset) - val_size
        train_dataset, validation_dataset = random_split(dataset, [train_size, val_size])

    if parser_args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))

    train_loader = torch.utils.data.DataLoader(
                                train_dataset, batch_size=parser_args.batch_size,
                                shuffle=(train_sampler is None),
                                num_workers=parser_args.num_workers,
                                pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=parser_args.batch_size, shuffle=False,
                    num_workers=parser_args.num_workers, pin_memory=True)

    actual_val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=parser_args.batch_size, shuffle=False,
        num_workers=parser_args.num_workers, pin_memory=True
    )

    #########################DATA LOADING CODE#########################


    for epoch in range(1, 20):
        print("STARTING TRAINING: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
        if parser_args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        print("Skipping training, just gonna round")
        print("Before Round: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
        # cp_model = round_model(model, parser_args.round, noise=parser_args.noise,
                                #ratio=parser_args.noise_ratio, rank=parser_args.gpu)
        if True:#(parser_args.multiprocessing_distributed and parser_args.gpu == 0) or not parser_args.multiprocessing_distributed:
            my_validate(actual_val_loader)
            acc1 = -1
        else:
            acc1 = -1
        print("GPU: {} | acc1={}".format(parser_args.gpu, acc1))
        print("After Round: Epoch {} | Memory Usage: {}".format(epoch, psutil.virtual_memory()))
        dist.barrier()
        continue

    print("GPU:{} | WE DID IT!!!")

def my_validate(val_loader):
    # batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    # losses = AverageMeter("Loss", ":.3f", write_val=False)
    # top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    # top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    # top10 = AverageMeter("Acc@10", ":6.2f", write_val=False)
    # progress = ProgressMeter(
    #    len(val_loader), [batch_time, losses, top1, top5, top10], prefix="Test: "
    # )
    top1 = 0
    top5 = 0
    top10 = 0
    num_images = 1

    # switch to evaluate mode

    with torch.no_grad():
        end = time.time()
        # for i, (images, target) in tqdm.tqdm(
        #     enumerate(val_loader), ascii=True, total=len(val_loader)
        # ):
        # time.sleep(0.1)
        # return -1, -1, -1
        for i, (images, target) in enumerate(val_loader):
            continue
            images = images.to(args.gpu)
            target = target.to(args.gpu)

            #print(images.shape, target.shape)

            # compute output
            # output = model(images)

            # loss = criterion(output, target)
            loss = torch.Tensor([0])

            # measure accuracy and record loss
            # acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
            acc1, acc5, acc10 = torch.Tensor([5]), torch.Tensor([5]), torch.Tensor([5])
            # losses.update(loss.item(), images.size(0))
            # top1.update(acc1.item(), images.size(0))
            # top5.update(acc5.item(), images.size(0))
            # top10.update(acc10.item(), images.size(0))
            # compute weighted sum for each accuracy so we can average it later
            top1 += acc1.item()# * images.size(0)
            top5 += acc5.item()# * images.size(0)
            top10 += acc10.item()# * images.size(0)
            num_images += 1#images.size(0)

            # measure elapsed time
            # batch_time.update(time.time() - end)
            batch_time = time.time() - end
            end = time.time()

            if i % args.print_freq == 0:
                # progress.display(i)
                print("GPU:{} | Epoch: {} | loss={} | Batch Time={}".format(args.gpu, epoch, loss.item(), acc1.item(), batch_time))

        # progress.display(len(val_loader))

        # if writer is not None:
        #     progress.write_to_tensorboard(
        #         writer, prefix="test", global_step=epoch)

    print("Model top1 Accuracy: {}".format(top1/num_images))
    return top1/num_images, top5/num_images, top10/num_images

def print_time():
    print("\n\n--------------------------------------")
    print("TIME: The current time is: {}".format(time.ctime()))
    print("TIME: The current time in seconds is: {}".format(time.time()))
    print("--------------------------------------\n\n")

if __name__ == "__main__":
    main()

