import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import math
import numpy as np
import pandas as pd
import re
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.models as torchvision_models
import models

model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/ubuntu/ILSVRC2012/',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--mixed-precision",
                    action='store_true',
                    default=False,
                    help="Use mixed precision or not")
parser.add_argument('--lmbda',
                    type=float,
                    default=0.001,
                    help='regularization coefficient lambda')
parser.add_argument("--iter-period",
                    type=int,
                    default=5,
                    help="period [epochs] for iterative pruning")
parser.add_argument("--invert-sanity-check",
                    action="store_true",
                    default=False,
                    help="Enable this to run the inverted sanity check (for HC)")
parser.add_argument("--prune-rate",
                    default=0.5,
                    type=float,
                    help="Decides fraction of weights TO PRUNE when calling prune()")
parser.add_argument("--target-sparsity",
                    default=5,
                    type=float,
                    help="Decides target sparsity in % when running GM")
parser.add_argument("--finetune",
                    action="store_true",
                    default=False,
                    help="Enable this to run the FT step after finding the subnetwork")
parser.add_argument('--checkpoint',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--bias",
                    action="store_true",
                    default=False,
                    help="Enable this to allow pruning biases")


best_acc1 = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained:
        print("==> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        # args.arch = 'resnet50'
        print("==> creating model '{}'".format(args.arch))
        # model = models.__dict__[args.arch]()
        model = models.ResNet50()
        # print("GPU: {} | Switching to wt to see if that works well!".format(args.gpu))
        # print("GPU: {} | First round model to all ones score".format(args.gpu))
        # model = round_model(model, round_scheme='all_ones')
        # model = switch_to_wt(model)

    if args.finetune:
        print("Finetuning Rare Gem. Loading checkpoint")
        #ckpt = torch.load(args.checkpoint)
        #model.load_state_dict(ckpt)
        print("Successfully loaded checkpoint: {}".format(args.checkpoint))
        model = round_model(model, round_scheme='all_ones')
        model = switch_to_wt(model)


    args.prune_rate = get_prune_rate(args.target_sparsity, args.iter_period, args.epochs)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # adding a scaler for mixed-precision training
    if args.mixed_precision:
        scaler = torch.cuda.amp.GradScaler(enabled=True) # mixed precision
    else:
        scaler = None
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    epoch_list = []
    test_acc_list = []
    model_sparsity_list = []
    val_acc_list = []
    train_acc_list = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if not args.finetune and epoch % (args.iter_period) == 0 and epoch != 0:
            prune(model, args)
            torch.distributed.barrier()

        print_time("Epoch: {} | Starting Train".format(epoch))
        start_train = time.time()
        # train for one epoch
        train_acc1 = train(train_loader, model, criterion, optimizer, epoch, args, scaler)

        train_time = train_time = (time.time() - start_train) / 60
        print("Epoch: {} | Train Time {}".format(epoch, train_time))

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        epoch_time = (time.time() - start_train) / 60
        print("Epoch: {} | Train + Val Time {}".format(epoch, epoch_time))

        # check sparsity of model
        cp_model = round_model(model, 'naive')
        avg_sparsity = get_model_sparsity(cp_model, threshold=0, args=args)

        epoch_list.append(epoch)
        train_acc_list.append(train_acc1.item())
        test_acc_list.append(acc1.item())
        val_acc_list.append(acc1.item())
        model_sparsity_list.append(avg_sparsity)

        scheduler.step()

        results_df = pd.DataFrame({'epoch': epoch_list,
                                   'test_acc': test_acc_list,
                                   'val_acc': val_acc_list,
                                   'train_acc': train_acc_list,
                                   'model_sparsity': model_sparsity_list})

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
            results_df.to_csv("acc_and_sparsity.csv", index=False)

        save_flag = ((epoch+1)%10 == 0) or (epoch > 85) or (epoch == args.epochs-1)
        if save_flag and (not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0)):
            torch.save(model.module.state_dict(), 'model_before_fineune_epoch_{}.pth'.format(epoch))
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_acc1': best_acc1,
            #     'optimizer' : optimizer.state_dict(),
            #     'scheduler' : scheduler.state_dict()
            # }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, scaler=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        if not args.finetune:
            # project to [0, 1] in every gradient step
            for name, params in model.named_parameters():
                # make sure param_name ends with .scores and not bias_scores
                if re.match('.*\.scores', name) and not re.match('.*\.bias_scores', name):
                    with torch.no_grad():
                        params.data = torch.clamp(params.data, 0.0, 1.0)

        # compute output
        if scaler is None:
            output = model(images)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                output = model(images)
                loss = criterion(output, target)

        if not args.finetune:
            regularization_loss = get_regularization_loss(model, args)
            # print("Regularization loss: {}".format(regularization_loss.item()))
            loss += regularization_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    if not args.finetune:
        # project to [0, 1] before returning model
        for name, params in model.named_parameters():
            # make sure param_name ends with .scores and not bias_scores
            if re.match('.*\.scores', name) and not re.match('.*\.bias_scores', name):
                with torch.no_grad():
                    params.data = torch.clamp(params.data, 0.0, 1.0)

    return top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


"""
@ksreenivasan: Dumping new code here
"""
def print_time(msg):
    print("\n\n----------------------------------------------------------------------------")
    print("{}".format(msg))
    print("TIME: The current time is: {}".format(time.ctime()))
    print("TIME: The current time in seconds is: {}".format(time.time()))
    print("----------------------------------------------------------------------------\n\n")

def get_regularization_loss(model, args):
    conv_layers, linear_layers = get_layers(args.arch, model)
    regularization_loss = torch.tensor(0.).to(args.gpu)

    # reg_loss =  ||p||_2^2
    for name, params in model.named_parameters():
        if ".bias_score" in name:
            # do nothing, because I'm pretending there are no biases
            regularization_loss += 0

        elif ".score" in name:
            regularization_loss += torch.norm(params, p=2)**2
    regularization_loss = args.lmbda * regularization_loss
    return regularization_loss

# return layer objects of conv layers and linear layers so we can parse them
# efficiently
def get_layers(arch='ResNet50', dist_model=None):
    if isinstance(dist_model, nn.parallel.DistributedDataParallel):
         model = dist_model.module
    else:
        model = dist_model

    if arch == 'ResNet50':
        conv_layers = [model.conv1]
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for basic_block_id in [i for i in range(len(layer))]:
                conv_layers.append(layer[basic_block_id].conv1)
                conv_layers.append(layer[basic_block_id].conv2)
                conv_layers.append(layer[basic_block_id].conv3)
                # handle shortcut
                # if len(layer[basic_block_id].shortcut) > 0:
                #     conv_layers.append(layer[basic_block_id].shortcut[0])
        linear_layers = [model.fc]

    return (conv_layers, linear_layers)


# switches off gradients for scores and flags and switches it on for weights and biases
def switch_to_wt(model):
    print('Switching to weight training by switching off requires_grad for scores and switching it on for weights.')

    for name, params in model.named_parameters():
        # make sure param_name ends with .weight or .bias
        if re.match('.*\.weight', name):
            params.requires_grad = True
        elif re.match('.*\.bias$', name):
            params.requires_grad = True
        elif "score" in name:
            params.requires_grad = False
        else:
            # flags and everything else
            params.requires_grad = False

    return model

# switches off gradients for weights and biases and switches it on for scores and flags
def switch_to_prune(model):
    print('Switching to pruning by switching off requires_grad for weights and switching it on for scores.')

    for name, params in model.named_parameters():
        # make sure param_name ends with .weight or .bias
        if re.match('.*\.scores', name) and re.match('.*\.bias_scores', name):
            params.requires_grad = True
        else:
            # weights, biases, bias_scores, flags and everything else
            params.requires_grad = False

    return model


def round_model(model, round_scheme='naive'):
    quantize_threshold=0.5
    print("Rounding model with scheme: {}".format(round_scheme))
    cp_model = copy.deepcopy(model)
    if isinstance(model, nn.parallel.DistributedDataParallel):
       # cp_model = copy.deepcopy(model.module)
       named_params = cp_model.module.named_parameters()
    else:
        # cp_model = copy.deepcopy(model)
        named_params = cp_model.named_parameters()
    for name, params in named_params:
        if re.match('.*\.scores', name):
            if round_scheme == 'naive':
                params.data = torch.gt(params.data, torch.ones_like(
                    params.data)*quantize_threshold).int().float()
            elif round_scheme == 'prob':
                params.data = torch.clamp(params.data, 0.0, 1.0)
                params.data = torch.bernoulli(params.data).float()
            elif round_scheme == 'all_ones':
                params.data = torch.ones_like(params.data)
            else:
                print("INVALID ROUNDING")
                print("EXITING")
                exit()

    # if isinstance(model, nn.parallel.DistributedDataParallel):
    #     cp_model = nn.parallel.DistributedDataParallel(
    #        cp_model, device_ids=[parser_args.gpu], find_unused_parameters=False)

    return cp_model


def prune(model, args, update_thresholds_only=False, update_scores=False):
    print("Pruning Model:")

    scores_threshold = bias_scores_threshold = -np.inf
    conv_layers, linear_layers = get_layers(args.arch, model)

    # prune the bottom k of scores
    num_active_weights = 0
    num_active_biases = 0
    active_scores_list = []
    active_bias_scores_list = []

    for layer in (conv_layers + linear_layers):
        num_active_weights += layer.flag.data.sum().item()
        active_scores = (layer.scores.data[layer.flag.data == 1]).clone()
        active_scores_list.append(active_scores)
        if args.bias:
            num_active_biases += layer.bias_flag.data.sum().item()
            active_biases = (
                layer.bias_scores.data[layer.bias_flag.data == 1]).clone()
            active_bias_scores_list.append(active_biases)

    number_of_weights_to_prune = np.ceil(
        args.prune_rate * num_active_weights).astype(int)
    number_of_biases_to_prune = np.ceil(
        args.prune_rate * num_active_biases).astype(int)

    agg_scores = torch.cat(active_scores_list)
    agg_bias_scores = torch.cat(
        active_bias_scores_list) if args.bias else torch.tensor([])

    # if invert_sanity_check, then threshold is based on sorted scores in descending order, and we prune all scores ABOVE it
    scores_threshold = torch.sort(
        torch.abs(agg_scores), descending=args.invert_sanity_check).values[number_of_weights_to_prune-1].item()

    if args.bias:
        bias_scores_threshold = torch.sort(
            torch.abs(agg_bias_scores), descending=args.invert_sanity_check).values[number_of_biases_to_prune-1].item()
    else:
        bias_scores_threshold = -1

    if update_thresholds_only:
        for layer in (conv_layers + linear_layers):
            layer.scores_prune_threshold = scores_threshold
        if args.bias:
            layer.bias_scores_prune_threshold = bias_scores_threshold

    else:
        for layer in (conv_layers + linear_layers):
            if args.invert_sanity_check:
                layer.flag.data = (layer.flag.data + torch.lt(layer.scores.abs(),  # TODO
                                   torch.ones_like(layer.scores)*scores_threshold).int() == 2).int()
            else:
                layer.flag.data = (layer.flag.data + torch.gt(layer.scores.abs(),  # TODO
                                   torch.ones_like(layer.scores)*scores_threshold).int() == 2).int()
            if update_scores:
                layer.scores.data = layer.scores.data * layer.flag.data
            if args.bias:
                if args.invert_sanity_check:
                    layer.bias_flag.data = (layer.bias_flag.data + torch.lt(layer.bias_scores, torch.ones_like(
                        layer.bias_scores)*bias_scores_threshold).int() == 2).int()
                else:
                    layer.bias_flag.data = (layer.bias_flag.data + torch.gt(layer.bias_scores, torch.ones_like(
                        layer.bias_scores)*bias_scores_threshold).int() == 2).int()
                if update_scores:
                    layer.bias_scores.data = layer.bias_scores.data * layer.bias_flag.data


    return scores_threshold, bias_scores_threshold


# returns num_nonzero elements, total_num_elements so that it is easier to compute
# average sparsity in the end
def get_layer_sparsity(layer, threshold=0, args=None):
    # assume the model is rounded, compute effective scores
    eff_scores = layer.scores * layer.flag
    if args.bias:
        eff_bias_scores = layer.bias_scores * layer.bias_flag
    num_middle = torch.sum(torch.gt(eff_scores,
                           torch.ones_like(eff_scores)*threshold) *
                           torch.lt(eff_scores,
                           torch.ones_like(eff_scores.detach()*(1-threshold)).int()))
    if num_middle > 0:
        print("WARNING: Model scores are not binary. Sparsity number is unreliable.")
        raise ValueError
    w_numer, w_denom = eff_scores.detach().sum().item(), eff_scores.detach().flatten().numel()

    if args.bias:
        b_numer, b_denom = eff_bias_scores.detach().sum().item(), eff_bias_scores.detach().flatten().numel()
    else:
        b_numer, b_denom = 0, 0

    return w_numer, w_denom, b_numer, b_denom


# returns avg_sparsity = number of non-zero weights!
def get_model_sparsity(model, threshold=0, args=None):
    conv_layers, linear_layers = get_layers(args.arch, model)
    numer = 0
    denom = 0

    # TODO: find a nicer way to do this (skip dropout)
    # TODO: Update: can't use .children() or .named_modules() because of the way things are wrapped in builder
    for conv_layer in conv_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(
            conv_layer, threshold, args)
        numer += w_numer
        denom += w_denom
        if args.bias:
            numer += b_numer
            denom += b_denom

    for lin_layer in linear_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(
            lin_layer, threshold, args)
        numer += w_numer
        denom += w_denom
        if args.bias:
            numer += b_numer
            denom += b_denom
    # print('Overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))
    return 100*numer/denom


def get_prune_rate(target_sparsity=0.5, iter_period=5, max_epochs=88):
    print("Computing prune_rate for target_sparsity {} with iter_period {}".format(
        target_sparsity, iter_period))
    num_prune_iterations = np.floor((max_epochs-1)/iter_period)
    # if algo is HC, iter_HC or anything that uses prune() then, prune_rate represents number of weights to prune
    prune_rate = 1 - np.exp(np.log(target_sparsity/100)/num_prune_iterations)
    return prune_rate


if __name__ == '__main__':
    main()
