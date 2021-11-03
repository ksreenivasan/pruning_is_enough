"""
 Just a generic debug script that creates the objects that we need
 and then reads a model from a checkpoint so that we can inspect it.
 Sometimes this is preferable to throwing in an ipdb.set_trace() debug point.

 Note that this WILL NOT work out of the box. I wrote this for Edge-Popup
 and therefore imports certain pieces that do not exist in our codebase.

 But I'm still putting it in here since it should be quite easy to modify
 and use.

 As a courtesy to others, if you were to make a change/addition that may
 be useful to others, please add a few comments and push the code.
"""


import os
import pathlib
import random
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
)
from utils.schedulers import get_policy


from args import *
import importlib

import data
import models

from main import *
from utils.conv_type import GetSubnet as GetSubnetConv

# load this guy: resnet18-sc-unsigned.yaml
yaml_txt = open("configs/smallscale/resnet18/resnet18-sc-unsigned.yaml").read()
# override args
loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
args.__dict__.update(loaded_yaml)
# this is EP. no bias
args.bias=False

model = get_model(args)
model = set_gpu(args, model)

ckpt = torch.load("kartik_ep_final_model.pt")
model.load_state_dict(ckpt)

# arch == 'cResNet18':
def get_layers(model):
    conv_layers = [model.conv1]
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for basic_block_id in [0, 1]:
            conv_layers.append(layer[basic_block_id].conv1)
            conv_layers.append(layer[basic_block_id].conv2)
            # handle shortcut
            if len(layer[basic_block_id].shortcut) > 0:
                conv_layers.append(layer[basic_block_id].shortcut[0])
    linear_layers = [model.fc]
    return (conv_layers, linear_layers)

# get layer sparsity
def get_layer_sparsity(layer, threshold=0):
    weight_mask= GetSubnetConv.apply(layer.scores.abs(), args.prune_rate)
    w_numer, w_denom = weight_mask.sum().item(), weight_mask.flatten().numel()
    b_numer, b_denom = 0, 0
    return w_numer, w_denom, b_numer, b_denom

def get_model_sparsity(model, threshold=0):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    conv_layers, linear_layers = get_layers(model)
    numer = 0
    denom = 0

    # TODO: find a nicer way to do this (skip dropout)
    # TODO: Update: can't use .children() or .named_modules() because of the way things are wrapped in builder
    for conv_layer in conv_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(conv_layer, threshold)
        numer += w_numer
        denom += w_denom
        if args.bias:
            numer += b_numer
            denom += b_denom

    for lin_layer in linear_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(lin_layer, threshold)
        numer += w_numer
        denom += w_denom
        if args.bias:
            numer += b_numer
            denom += b_denom
    # print('Overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))
    return 100*numer/denom

get_model_sparsity(model)