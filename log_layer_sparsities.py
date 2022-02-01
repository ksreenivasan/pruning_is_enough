"""
	Just a little script I made to record layer sparsities for HC models
"""

import importlib

import data
import models
import pandas as pd

from main import *
from utils.conv_type import GetSubnet
from utils.net_utils import get_model_sparsity, get_layer_sparsity, prune

import re
import yaml


# load this guy: resnet18-sc-unsigned.yaml
# yaml_txt = open("configs/hypercube/resnet20/resnet20_target_sparsity_50.yml").read()
yaml_txt = open("configs/sr/resnet20/resnet20_sr.yml").read()
parser_args.gpu = 0

model = get_model(parser_args)
model = set_gpu(parser_args, model)

device = torch.device("cuda:0")

# enter checkpoint here
# ckpt1 = torch.load("/workspace/results_repo_pruning/resnet20_exps/final_results/target_sparsity_0_5_highreg/model_before_finetune.pth")
# ckpt2 = torch.load("/workspace/results_repo_pruning/resnet20_exps/final_results/target_sparsity_0_5_medreg/model_before_finetune.pth")
# ckpt3 = torch.load("/workspace/results_repo_pruning/resnet20_exps/final_results/target_sparsity_1_4_highreg/model_before_finetune.pth")
# ckpt4 = torch.load("/workspace/results_repo_pruning/resnet20_exps/final_results/target_sparsity_1_4_medreg/model_before_finetune.pth")

sparsity_dict = {}

# for sparsity in ['50', '3_72', '1_4', '0_59']:
for smart_ratio in [0.9941, 0.9856, 0.9628, 0.95, 0.9, 0.8, 0.5]:
        sparsity_list = []
        ckpt = torch.load("results/check_sr_sparsity/results_pruning_CIFAR10_resnet20_hc_iter_0_5_5_reg_None__sgd_None_0_001_0_1_50_finetune_0_1_MAML_-1_10_fan_True_kaiming_normal_unif_width_1_0_seed_42_idx_None/init_model_{}.pth".format(smart_ratio))
        # ckpt = torch.load("results/ckpt_resnet20_sp{}_results_trial_1/model_before_finetune.pth".format(sparsity))
        model.load_state_dict(ckpt)
        cp_model = round_model(model, 'all_ones')
        conv_layers, lin_layers = get_layers(arch='resnet20', model=cp_model)
        print("\n\n\n---------------------------------------------------------------------------------------------")
        print("Overall sparsity: {} === Target sparsity: {}".format(get_model_sparsity(cp_model), (1 - smart_ratio) * 100))
        print("---------------------------------------------------------------------------------------------\n\n\n")
        for conv_layer in conv_layers:
            w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(conv_layer)
            print("Layer: {} | {}/{} weights | Sparsity = {}".format(conv_layer, w_numer, w_denom, 100.0*w_numer/w_denom))
            sparsity_list.append(100.0*w_numer/w_denom)


        for lin_layer in lin_layers:
            w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(lin_layer)
            print("Layer: {} | {}/{} weights | Sparsity = {}".format(lin_layer, w_numer, w_denom, 100.0*w_numer/w_denom))
            sparsity_list.append(100.0*w_numer/w_denom)
        sparsity_dict[round((1 - smart_ratio) * 100, 2)] = sparsity_list


df = pd.DataFrame(sparsity_dict)
df.to_csv("resnet20_smart_ratio_layerwise_sparsity.csv", index=False)
# df.to_csv("resnet20_layerwise_sparsity.csv", index=False)
