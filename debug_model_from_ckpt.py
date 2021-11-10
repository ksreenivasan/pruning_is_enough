"""
 Just a generic debug script that creates the objects that we need
 and then reads a model from a checkpoint so that we can inspect it.
 Sometimes this is preferable to throwing in an ipdb.set_trace() debug point.

 Note that this won't do what you want out of the box. I've been hacking
 it together to do whatever I want.
 But I'm still putting it in here since it should be quite easy to modify
 and use.

 As a courtesy to others, if you were to make a change/addition that may
 be useful to others, please add a few comments and push the code.
"""

from args import *
import importlib

import data
import models

from main import *
from utils.conv_type import GetSubnet

import re

# load this guy: resnet18-sc-unsigned.yaml
yaml_txt = open("configs/hypercube/resnet20/resnet20_quantized_hypercube_reg.yml").read()
# override args
loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
args.__dict__.update(loaded_yaml)

model = get_model(args)
model = set_gpu(args, model)

weight_params = []
bias_params = []
other_params = []
for name, param in model.named_parameters():
    # make sure param_name ends with .weight
    if re.match('.*\.weight', name):
        weight_params.append(name)
        # param.requires_grad = True
    # make sure param_name ends with .bias
    elif args.bias and re.match('.*\.bias$', name):
        bias_params.append(name)
        # param.requires_grad = True
    else:
        other_params.append(name)
        # param.requires_grad = False

# ckpt = torch.load("kartik_ep_final_model.pt")
# model.load_state_dict(ckpt)

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
    weight_mask= GetSubnet.apply(layer.scores.abs(), args.prune_rate)
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
        print("Sparsity of layer {}={}%".format(conv_layer, 100.0*w_numer/w_denom))
        print("Size of layer {}={}".format(conv_layer, w_denom))
        numer += w_numer
        denom += w_denom
        if args.bias:
            numer += b_numer
            denom += b_denom

    for lin_layer in linear_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(lin_layer, threshold)
        print("Sparsity of layer {}={}%".format(lin_layer, 100.0*w_numer/w_denom))
        print("Size of layer {}={}".format(lin_layer, w_denom))
        numer += w_numer
        denom += w_denom
        if args.bias:
            numer += b_numer
            denom += b_denom
    # print('Overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))
    return 100*numer/denom

sparsity = get_model_sparsity(model)
print("Sparsity of final model={}".format(sparsity))

train, validate, modifier = get_trainer(args)
criterion = nn.CrossEntropyLoss().cuda()
data = get_dataset(args)
acc1, acc5 = validate(data.val_loader, model, criterion, args, writer=None, epoch=-1)
print("Accuracy of final model={}".format(acc1))

