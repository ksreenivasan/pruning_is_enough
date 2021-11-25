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

from args_helper import ArgsHelper
import importlib

import data
import models

from main import *
from utils.conv_type import GetSubnet
from utils.net_utils import get_model_sparsity, get_layer_sparsity

import re

parser_args = ArgsHelper().get_args()


# load this guy: resnet18-sc-unsigned.yaml
#yaml_txt = open("configs/hypercube/resnet20/resnet20_quantized_hypercube_reg_bottom_K.yml").read()
# override args
#loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
#args.__dict__.update(loaded_yaml)
parser_args.bias = False

model = get_model(parser_args)
model = set_gpu(parser_args, model)

# enter checkpoint here
#ckpt = torch.load("results/SGD_finetune/results_pruning_CIFAR10_resnet20_hc_iter_0_2_8_reg_L1_0_0001_sgd_constant_lr_0_1_0_1_50_fan_0_1_False_signed_constant_width_unif_seed_1_0/model_before_finetune.pth")
#ckpt = torch.load("results/SGD_finetune/results_pruning_CIFAR10_resnet20_hc_iter_0_2_8_reg_L1_0_0001_sgd_constant_lr_0_1_0_1_50_fan_0_1_False_signed_constant_width_unif_seed_1_0/model_after_finetune.pth")


ckpt = torch.load("results/SGD_finetune/results_pruning_CIFAR10_resnet20_hc_iter_0_2_8_reg_L1_5e-05_sgd_cosine_lr_0_1_0_1_50_fan_0_01_False_signed_constant_width_unif_seed_1_0/model_before_finetune.pth")
#ckpt = torch.load("results/SGD_finetune/results_pruning_CIFAR10_resnet20_hc_iter_0_2_8_reg_L1_5e-05_sgd_cosine_lr_0_1_0_1_50_fan_0_01_False_signed_constant_width_unif_seed_1_0/model_after_finetune.pth")

# note that if you are loading ckpt from the ramanujan-style savepoints, you need to add ckpt['state_dict']
# otherwise, we typically save the state dict directly, so you can just use ckpt
#model.load_state_dict(ckpt['state_dict'])
model.load_state_dict(ckpt)

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

train, validate, modifier = get_trainer(args)
criterion = nn.CrossEntropyLoss().cuda()
data = get_dataset(args)
acc1, acc5, acc10 = validate(data.val_loader, model, criterion, args, writer=None, epoch=-1)
print("Accuracy of final model={}".format(acc1))

