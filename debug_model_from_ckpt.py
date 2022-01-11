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

import importlib

import data
import models

from main import *
from utils.conv_type import GetSubnet
from utils.net_utils import get_model_sparsity, get_layer_sparsity, prune

import re


# load this guy: resnet18-sc-unsigned.yaml
yaml_txt = open("configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_0_5_highreg.yml").read()
# override args
#loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
#args.__dict__.update(loaded_yaml)
parser_args.bias = False

model = get_model(parser_args)
model = set_gpu(parser_args, model)

device = torch.device("cuda:{}".format(parser_args.gpu))

# enter checkpoint here
ckpt1 = torch.load("/workspace/results_repo_pruning/resnet20_exps/final_results/target_sparsity_0_5_highreg/model_before_finetune.pth")
ckpt2 = torch.load("/workspace/results_repo_pruning/resnet20_exps/final_results/target_sparsity_0_5_medreg/model_before_finetune.pth")
ckpt3 = torch.load("/workspace/results_repo_pruning/resnet20_exps/final_results/target_sparsity_1_4_highreg/model_before_finetune.pth")
ckpt4 = torch.load("/workspace/results_repo_pruning/resnet20_exps/final_results/target_sparsity_1_4_medreg/model_before_finetune.pth")

# note that if you are loading ckpt from the ramanujan-style savepoints, you need to add ckpt['state_dict']
# otherwise, we typically save the state dict directly, so you can just use ckpt
# model.load_state_dict(ckpt['state_dict'])
for ckpt in [ckpt1, ckpt2, ckpt3, ckpt4]:
    model.load_state_dict(ckpt)
    cp_model = round_model(model, 'naive')
    conv_layers, lin_layers = get_layers(arch='resnet20', model=cp_model)
    for conv_layer in conv_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(conv_layer)
        print("Layer: {} | {}/{} weights | Sparsity = {}".format(conv_layer, w_numer, w_denom, 100.0*w_numer/w_denom))

    for lin_layer in lin_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(lin_layer)
        print("Layer: {} | {}/{} weights | Sparsity = {}".format(lin_layer, w_numer, w_denom, 100.0*w_numer/w_denom))

# test global ep
parser_args.algo = 'global_ep'
parser_args.prune_rate = 0.992

# update parser_args.ep_threshold
prune(model, update_thresholds_only=True)

conv_layers, lin_layers = get_layers(arch='resnet20', model=model)

train, validate, modifier = get_trainer(parser_args)
criterion = nn.CrossEntropyLoss().cuda()
data = get_dataset(parser_args)
acc1, acc5, acc10 = validate(data.val_loader, model, criterion, parser_args, writer=None, epoch=-1)
print("Accuracy of final model={}".format(acc1))

# get sample image
for idx, (images, target) in enumerate(data.train_loader):
    images = images.to(device)
    target = target.to(device)
    break

# check if model has bottlenecks
out = model(images)
print("Output sum is: {}".format(out.sum()))

for conv_layer in conv_layers:
    subnet, bias_subnet = GetSubnet.apply(conv_layer.scores, conv_layer.bias_scores, parser_args.prune_rate)
    print("Layer: {}".format(conv_layer))
    print("Mask: {}/{}".format(subnet.sum(), subnet.flatten().size(dim=0)))


# debug traditional HC style algo
cp_model = round_model(model, 'naive')
conv_layers, lin_layers = get_layers(arch='resnet20', model=cp_model)

for conv_layer in conv_layers:
    w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(conv_layer)
    print("Layer: {} | {}/{} weights | Sparsity = {}".format(conv_layer, w_numer, w_denom, 100.0*w_numer/w_denom))

for lin_layer in lin_layers:
    w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(lin_layer)
    print("Layer: {} | {}/{} weights | Sparsity = {}".format(lin_layer, w_numer, w_denom, 100.0*w_numer/w_denom))

print(args.arch)

"""
weight_params = []
bias_params = []
other_params = []
for name, param in model.named_parameters():
    # make sure param_name ends with .weight
    if re.match('.*\.weight', name):
        weight_params.append(param)
        # param.requires_grad = True
    # make sure param_name ends with .bias
    elif parser_args.bias and re.match('.*\.bias$', name):
        bias_params.append(param)
        # param.requires_grad = True
    else:
        other_params.append(param)
        # param.requires_grad = False
"""

sparsity = get_model_sparsity(cp_model)
print("Sparsity of final model={}".format(sparsity))

