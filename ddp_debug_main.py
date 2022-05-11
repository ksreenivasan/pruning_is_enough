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
import psutil
import argparse
import sys
import yaml
from configs import parser as _parser

global parser_args

parser = argparse.ArgumentParser(description="Pruning random networks")
# Config/Hyperparameters
parser.add_argument(
    "--data",
    default="data/datasets/",
    help="path to dataset base directory"
)
parser.add_argument(
    "--log-dir",
    default=None,
    help="Where to save the runs. If None use ./runs"   
    )
parser.add_argument(
    "--name",
    default=None,
    type=str,
    help="Name of experiment"
)
parser.add_argument(
    "--config",
    default='configs/hypercube/resnet20/resnet20_quantized_iter_hc_target_sparsity_1_4_highreg.yml',
    help="Config file to use"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    metavar="N",
    help="Input batch size for training (default: 64)"
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="Number of epochs to use in training (default: 10)"
)
parser.add_argument(
    "--submask-size",
    type=int,
    default=1,
    metavar="S",
    help="Size of random 0/1 submask to create among a random set of weights in the network (default: 1)"
)
parser.add_argument(
    "--metric",
    type=str,
    default="loss",
    metavar="M",
    help="Metric used to determine whether a weight or activation should be pruned (default: \"loss\")"
)
parser.add_argument(
    "--pruning-strategy",
    type=str,
    default=None,
    metavar="PS",
    help="Strategy for pruning. Can be \"weights\", \"activations\", \"activations_and_weights\", \"simulated_annealing\", or None. If None, train a model via vanilla SGD (default: None)"
)
parser.add_argument(
    "--how-to-prune",
    type=str,
    default="random",
    help="How the pruning strategy will be executed. Can be \"random\", \"random_with_replacement\", \"layer\", or \"layer_reversed.\" This argument has changes how the weights or activations in the network are selected for pruning. \"random\" will choose the weights/activations without replacement, \"random_with_replacement\" will choose the weights/activations with replacement, and \"layer\" will choose every weight/activation in the network in layer by layer, and \"layer_reversed\" is the same as \"layer,\" but will start with the last layer and move to the first (default: \"random\")"
)
parser.add_argument(
    "--start-from-nothing",
    action="store_true",
    default=False,
    help="Prunes all weights in the network, but leaves one weight per layer to connect the input layer to the output layer. Only applies to masked layers (default: False)"
)
parser.add_argument(
    "--flips",
    type=list,
    default=None,
    nargs="+",
    metavar="R",
    help="List of epoch indices (starting from and including 0) where flipping occurs. At each milestone, 10% of the mask parameters are randomly chosen and they are flipped. Restarting is accomplished at the beginning of an epoch. Applies only when pruning weights (default: None)"
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    metavar="LR",
    help="Learning rate (default: 0.001)"
)
parser.add_argument(
    "--lr-policy",
    type=str,
    default=None,
    help="Learning rate scheduler"
)
parser.add_argument(
    "--lr-gamma",
    type=float,
    default=0.1,
    help="Multistep lr decay ratio (default: 0.1)"
)
parser.add_argument(
    "--lr-adjust",
    type=int,
    default=50,
    help="Multistep lr decay period (default: 50)"
)

parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="Momentum (default: 0.9)"
)
parser.add_argument(
    "--wd",
    type=float,
    default=0.0005,
    metavar="WD",
    help="Weight decay (default: 0.0005)"
)
parser.add_argument(
    "--nesterov",
    type=bool,
    default=False,
    metavar="N",
    help="Nesterov acceleration (default: False)"
)
parser.add_argument(
    "--milestones",
    type=list,
    default=[50, 100, 150, 200],
    nargs="+",
    metavar="M",
    help="List of milestones where learning rate is multiplied by args.gamma (default: [50, 100, 150, 200])"
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.1,
    metavar="G",
    help="Multiplicative factor to reduce the learning rate at every milestone (default: 0.1)"
)
parser.add_argument(
    "--td",
    type=float,
    default=0.99,
    metavar="TD",
    help="Temperature decay constant for simulated annealing (default: 0.99)"
)
parser.add_argument(
    "--temp",
    type=int,
    default=100000,
    metavar="T",
    help="Temperature used in simulated annealing (default: 100000)"
)
parser.add_argument(
    "--max-iter",
    type=int,
    default=100000,
    metavar="MI",
    help="Maximum number of iterations to run simulated annealing for before terminating. It's recommended to set this to a value that at is at least as big as the number of parameters in the network (default: 100000)"
)
parser.add_argument(
    "--algo",
    type=str,
    default='ep',
    help="pruning algo to use |ep|pt_hack|pt_reg|hc|ep+greedy|greedy+ep|hc_iter|global_ep|global_ep_iter|imp"
)
parser.add_argument(
    "--iter_start", 
    type=int, 
    default=0,
    help="starting epoch for iterative pruning"
)
parser.add_argument(
    "--iter-period", 
    type=int,
    default=5,
    help="period [epochs] for iterative pruning"
)
parser.add_argument(
    "--optimizer",
    type=str,
    default='sgd',
    help="optimizer option to use |sgd|adam|"
)
parser.add_argument(
    '--evaluate-only',
    action='store_true',
    default=False,
    help='just use rounding techniques to evaluate a saved model'
)
parser.add_argument(
    "--round",
    type=str,
    default='naive',
    help='rounding technique to use |naive|prob|pb|'
    # naive: threshold(0.5), prob: probabilistic rounding, pb: pseudo-boolean paper's choice (RoundDown)
)
parser.add_argument(
    '--noise',
    action='store_true',
    default=False,
    help='flag that decides if we add noise to the rounded p_i'
)
parser.add_argument(
    "--noise-ratio",
    type=float,
    default=0.0,
    help="portion of score flipping"
)
parser.add_argument(
    "--score-init",
    type=str,
    default="unif",
    help="initial score for hypercube |unif|bern|"
)
parser.add_argument(
    "--interpolate",
    type=str,
    default="prob",
    help="way of interpolating masks/weights |prob|linear|"
)
parser.add_argument(
    '--plot-hc-convergence',
    action='store_true',
    default=False,
    help='flag that decides if we plot convergence of hc'
)
parser.add_argument(
    "--hc-warmup",
    default=9999,
    type=int,
    help="warmup epochs for hypercube"
)
parser.add_argument(
    "--hc-period",
    default=1,
    type=int,
    help="rounding period for hypercube"
)
parser.add_argument(
    "--num-round",
    type=int,
    default=1,
    help='number of different models testing in rounding'
)
# do we need it?
parser.add_argument(
    "--num-test",
    type=int,
    default=1,
    help='number of different models testing in prob rounding'
)
parser.add_argument(
    "--save-model",
    action='store_true',
    default=False,
    help='For Saving the current Model'
)
# Architecture and training
parser.add_argument(
    "--arch",
    type=str,
    default="TwoLayerFC",
    # KS: gotta find a better way to do this. causing circular import issues
    # help="Model architecture: " + " | ".join(models.__dict__["__all__"]) + " | (default: TwoLayerFC)"
)
parser.add_argument(
    "--width",
    type=float,
    default=1.0,
    help="portion of additional width compared with original width"
)


parser.add_argument(
    "--hidden-size",
    type=int,
    default=500,
    metavar="H",
    help="Number of nodes in the FC layers of the network"
)
parser.add_argument(
    "--bias",
    action="store_true",
    default=False,
    help="Boolean flag to indicate whether to use bias"
)
parser.add_argument(
    "--freeze-weights",
    action="store_true",
    default=True,
    help="Boolean flag to indicate whether weights should be frozen. Used when sparsifying (default: False)"
)
parser.add_argument(
    "--conv-type",
    type=str,
    default=None,
    help="What kind of sparsity to use"
)
parser.add_argument(
    "--mode",
    default="fan_in",
    help="Weight initialization mode"
)
parser.add_argument(
    "--nonlinearity",
    default="relu",
    help="Nonlinearity used by initialization"
)
parser.add_argument(
    "--bn-type",
    default=None,
    help="BatchNorm type"
)
parser.add_argument(
    "--init",
    default="kaiming_normal",
    help="Weight initialization modifications"
)
parser.add_argument(
    "--no-bn-decay",
    action="store_true",
    default=False,
    help="No batchnorm decay"
)
parser.add_argument(
    "--scale-fan",
    action="store_true",
    default=False,
    help="scale fan"
)
parser.add_argument(
    "--first-layer-dense",
    action="store_true",
    help="First layer dense or sparse"
)
parser.add_argument(
    "--dense-training",
    action="store_true",
    help="Train the network without pruning, and checkpoint along the way"
)
parser.add_argument(
    "--last-layer-dense",
    action="store_true",
    help="Last layer dense or sparse"
)
parser.add_argument(
    "--label-smoothing",
    type=float,
    help="Label smoothing to use, default 0.0",
    default=None,
)
parser.add_argument(
    "--first-layer-type",
    type=str,
    default=None,
    help="Conv type of first layer"
)
parser.add_argument(
    "--trainer", type=str, default="default", help="cs, ss, or standard training"
)
parser.add_argument(
    "--score-init-constant",
    type=float,
    default=None,
    help="Sample Baseline Subnet Init",
)
# represents percentage of weights THAT REMAIN each time ep, global_ep
# whereas it represents number of weights TO PRUNE when calling prune()
parser.add_argument(
    "--prune-rate",
    default=0.5,
    help="Decides number of weights that REMAIN after sparse training.",
    type=float,
)
parser.add_argument(  # add for bottom K iterative pruning
    "--prune-type",
    default="FixThresholding",
    help="Type of prune - fix thresholding (FixTHresholding), or prune bottem k percent (BottomK)",
    type=str,
)
parser.add_argument(
    "--dataset",
    type=str,
    default="MNIST",
    help="Dataset to train the model with. Can be CIFAR10 or MNIST (default: MNIST)"
)
parser.add_argument(
    "--loss",
    type=str,
    default="cross-entropy-loss",
    help="which loss to use for pruning: cross-entropy-loss or zero-one-loss"
)
#    # Save/Load
#    parser.add_argument(
#        "--save-dir",
#        type=str,
#        default=False,
#        help="Directory to save the results of the experiment (default: \"./results\")"
#    )
parser.add_argument(
    "--save-plot-data",
    action="store_true",
    default=False,
    help="Boolean flag to indicate whether to save training data for plotting. The following data will be saved at every epoch: top-1 training set accuracy, top-1 validation set accuracy, epoch number (default: False)"
)
parser.add_argument(
    "--data-dir",
    type=str,
    default="../data",
    help="Directory of dataset to load in"
)
parser.add_argument(
    "--load-ckpt",
    type=str,
    default=None,
    help="Path to checkpoint to load from"
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=2,
    metavar="N",
    help="The number of batches to wait before logging training status. When pruning, this is the number of epochs before logging status (default: 100)"
)
parser.add_argument(
    "--ckpt-interval",
    type=int,
    default=-1,
    help="The number of epochs to train before saving the next checkpoint"
)
# Device settings
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    metavar="S",
    help="Random seed (default: 42)"
)
parser.add_argument(
    "--seed-fixed-init",
    type=int,
    default=24,
    metavar="S",
    help="Random seed when used for fixing weight/score init  (default:24)"
)
parser.add_argument(
    "--trial-num",
    type=int,
    default=1,
    help="Trial number (1,2, ...)"
)
parser.add_argument(
    "--fixed-init",
    action="store_true",
    default=False,
    help="fixed weight initialization"
)
parser.add_argument(
    "--mode-connect",
    action="store_true",
    default=False,
    help="Boolean flag to indicate whether to run mode connectivity"
)
parser.add_argument(
    '--mode-connect-filename',
    type=str,
    default=None,
    help='filename for state_dict used for mode connectivity'
)
parser.add_argument(
    '--how-to-connect',
    type=str,
    default='prob',
    help="procedure for interpolating the mask. Can be \"random\", which chooses a binary value for a mask coordinate with bern(alpha), \"score\" which uses the [0, 1] score for each coordinate for interpolation (the mask consists of continuous values rather than binary in this case), and \"round\" which is uses the mask obtained from the \"score\" option and applies naive rounding."
)
parser.add_argument(
    "--no-cuda",
    action="store_true",
    default=False,
    help="Disables CUDA training"
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    metavar="G",
    help="Override the default choice for a CUDA-enabled GPU by specifying the GPU\"s integer index (i.e. \"0\" for \"cuda:0\")"
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=4,
    metavar="W",
    help="Number of workers"
)
parser.add_argument(    
    "--shift",
    type=float,
    default=0.0,
    help="shift portion"
)   
parser.add_argument(    
    "--num-trial",
    type=int,
    default=1,
    help="number of trials for testing sharpness"
)
# WARNING: With DataParallel, this causes some issues
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    default=None,
    type=str,
    help="use pre-trained model",
)
parser.add_argument(
    "--pretrained2",
    dest="pretrained2",
    default=None,
    type=str,
    help="use pre-trained model 2",
)
parser.add_argument(
    "--save_every",
    default=-1,
    type=int,
    help="Save every ___ epochs"
)
'''
# @ksreenivasan: commenting this for now. I think EP uses it.
parser.add_argument(
    "--random-subnet",
    action="store_true",
    help="Whether or not to use a random subnet when fine tuning for lottery experiments",
)
'''
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--compare-rounding",
    dest="compare-rounding",
    action="store_true",
    help="compare different rounding schemes",
)
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
    )
parser.add_argument(
    "--width-mult",
    default=1.0,
    help="How much to vary the width of the network.",
    type=float,
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--warmup_length",
    default=0,
    type=int,
    help="Number of warmup iterations"
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    '--results-filename',
    type=str,
    default=None,
    help='csv results filename'
)
parser.add_argument(
    '--weight-training',
    action='store_true',
    default=False,
    help='flag that decides if we are doing pruning or weight training'
)
parser.add_argument(
    '--regularization',
    default=None,
    type=str,
    help='which regularizer to add : |var_red_1|var_red_2|bin_cross_entropy|'
)
"""
var_red_1: lmbda * p^(alpha) (1-p)^(alpha')
var_red_2: w^2 p(1-p)
bin_cross_entropy: -plog(1-p)?
"""
parser.add_argument(
    '--lmbda',
    type=float,
    default=0.001,
    help='regularization coefficient lambda'
)
parser.add_argument(
    "--alpha",
    default=1.0,
    type=float,
    help="first exponent in regularizer",
)
parser.add_argument(
    "--alpha_prime",
    default=1.0,
    type=float,
    help="second exponent in regularizer",
)

# Distributed training
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training"
)
parser.add_argument(
    "--rank",
    default=-1,
    type=int,
    help="node rank for distributed training"
)
parser.add_argument(
    "--dist-backend",
    default="nccl",
    type=str,
    help="distributed backend"
)
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
         "N processes per node, which has N GPUs. This is the "
         "fastest way to use PyTorch for either single node or "
         "multi node data parallel training"
)
parser.add_argument(
    "--random-subnet",
    action="store_true",
    default=False,
    help="Just initializes random subnetwork and then trains"
)
parser.add_argument(
    "--hc-quantized",
    action="store_true",
    default=False,
    help="round probablities in every iteration"
)
parser.add_argument(
    "--quantize-threshold",
    default=0.5,
    type=float,
    help="threhsold to use while quantizing scores in HC",
)
parser.add_argument(
    "--checkpoint-at-prune",
    action="store_true",
    default=False,
    help="save checkpoints every time we prune"
)
parser.add_argument(
    "--skip-sanity-checks",
    action="store_true",
    default=False,
    help="Enable this to skip sanity checks (save time)"
)
parser.add_argument(
    "--skip-fine-tune",
    action="store_true",
    default=False,
    help="Enable this to skip fine tuning (get pure pruned network)"
)
parser.add_argument(
    "--shuffle",
    action="store_true",
    default=False,
    help="shuffle weights/masks before sanity check"
)
parser.add_argument(
    "--reinit",
    action="store_true",
    default=False,
    help="reinit weights/masks before sanity check"
)
parser.add_argument(
    "--chg_mask",
    action="store_true",
    default=False,
    help="chg masks before sanity check"
)
parser.add_argument(
    "--chg_weight",
    action="store_true",
    default=False,
    help="chg weights before sanity check"
)
parser.add_argument(
    "--fine-tune-optimizer",
    type=str,
    default='sgd',
    help="optimizer option to use |sgd|adam| for fine-tuning weights"
)
parser.add_argument(
    "--fine-tune-lr-policy",
    type=str,
    default=None,
    help="Learning rate scheduler (for finetune)"
)
parser.add_argument(
    "--fine-tune-lr",
    type=float,
    default=0.01,
    metavar="LR",
    help="Learning rate for fine-tuning weights"
)
parser.add_argument(
    "--fine-tune-wd",
    type=float,
    default=0.0001,
    metavar="WD",
    help="Weight decay for fine-tuning weights"
)
#    parser.add_argument(
#        "--multigpu",
#        default=None,
#        type=str,
#        help="Which GPUs to use for multigpu training, comma separated"
#    )
parser.add_argument(
    "--rewind-score",
    action="store_true",
    default=False,
    help="if set True, every time when we prune, we set the score back to the initial state"
)
parser.add_argument(
    "--rewind-to-epoch",
    default=-1,
    type=int,
    help="to rewind to some epoch, you have to explicitly set the argument. Otherwise the code will never cache the rewinded score"
)
parser.add_argument(
    "--differentiate-clamp",
    action="store_true",
    default=False,
    help="if set True, we project to [0, 1] in forward and therefore differentiate clamp "
)
parser.add_argument(
    "--project-freq",
    default=1,
    type=int,
    help="project scores to [0, 1] every k gradient steps"
)
parser.add_argument(
     "--run_idx",
     default=None,
     help="index of run used for counting yml/log/save_folder"
)
parser.add_argument(
     "--subfolder",
     default=None,
     help="subfolder within the location for saving the results"
)
# added parser args for IMP
parser.add_argument(
    "--imp_rewind_iter", 
    default=1000, 
    type=int, 
    help="which iterations to rewind to"
)
parser.add_argument(
    "--imp-resume-round",
    default=-1,
    type=int, 
    help="which round to resume to"
)
parser.add_argument(
    "--imp-resume-epoch",
    default=-1,
    type=int, 
    help="which epoch to resume to"
)
parser.add_argument(
    "--imp-resume-iter",
    default=-1,
    type=int, 
    help="which iter to resume to"
)
parser.add_argument(
    "--imp-rewind-model", 
    default="short_imp/Liu_checkpoint_model_correct.pth"
)
parser.add_argument(
    "--imp-no-rewind",
    action="store_true",
    default=False,
    help="if set True, we run IMP algorithm without rewinding to previous states"
)
parser.add_argument(
    "--imp-rounds",
    type=int,
    default=-1,
    help="if set > 0, then ignore the epochs statement, and calculate epochs based on rounds * iter / round"
)
parser.add_argument(
    "--smart_ratio", 
    type=float,
    default=-1,
    help="the pruning weights in [0, 1]. E.g. smart_ratio = 0.98 will end up with a 2\% weight remaining model"
)
parser.add_argument(
    "--bottom-k-on-forward",
    action="store_true",
    default=False,
    help="Enable this to use bottomK on forward for HC"
)
parser.add_argument(
    "--target-sparsity",
    default=0.5,
    help="decides max percentage of weights that remain at the end of training",
    type=float,
)
parser.add_argument(
    "--lam_finetune_loss",
    type=float,
    default=-1,
    help="lambda for finetune loss "
)
parser.add_argument(
    "--num_step_finetune",
    type=int,
    default=10,
    help="number of steps to check finetune loss "
)
parser.add_argument(
    "--unflag-before-finetune",
    action="store_true",
    default=False,
    help="Enable this to unprune weights if possible, before fine-tune"
)
parser.add_argument(
    "--override-prune-rate",
    action="store_true",
    default=False,
    help="Enable this to specify prune-rate manually"
)
parser.add_argument(
    "--mixed_precision",
    type=int,
    default=0,
    help="Use mixed precision or not"
)
parser.add_argument('--transformer_emsize',
    type=int, default=200,
    help='size of word embeddings'
)
parser.add_argument('--transformer_nhid',
    type=int,
    default=200,
    help='number of hidden units per layer'
)
parser.add_argument('--transformer_nlayers',
    type=int,
    default=2,
    help='number of layers'
)
parser.add_argument('--transformer_clip',
    type=float,
    default=0.25,
    help='gradient clipping'
)
parser.add_argument('--transformer_bptt',
    type=int,
    default=35,
    help='sequence length'
)
parser.add_argument('--transformer_dropout',
    type=float,
    default=0.2,
    help='dropout applied to layers (0 = no dropout)'
)
parser.add_argument('--transformer_nhead',
    type=int,
    default=2,
    help='the number of heads in the encoder/decoder of the transformer model'
)
parser.add_argument(
    "--only-sanity",
    action="store_true",
    default=False,
    help="Only run sanity checks on the files in specific directory or subdirectories"
)
parser.add_argument(
    "--invert-sanity-check",
    action="store_true",
    default=False,
    help="Enable this to run the inverted sanity check (for HC)"
)
parser.add_argument(
    "--sanity-folder",
    default=None,
    type=str,
    metavar="PATH",
    help="directory(s) to access for only sanity check",
)
parser.add_argument(
    "--sr-version",
    default=1,
    type=int,
    help="smart ratio version number (1, 2, ...)",
)
parser.add_argument(
    "--use-full-data",
    action="store_true",
    default=False,
    help="Enable this use full train data and not leave anything for validation"
)
parser.add_argument(
    "--port",
    default=29500,
    type=int,
    help="Specify port to use for DDP",
)

parser_args = parser.parse_args()
# get commands from command line
override_args = _parser.argv_to_vars(sys.argv)

# load yaml file
yaml_txt = open(parser_args.config).read()

# override args
loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
for v in override_args:
    loaded_yaml[v] = getattr(parser_args, v)

print(f"=> Reading YAML config from {parser_args.config}")
parser_args.__dict__.update(loaded_yaml)


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
        mp.spawn(main_worker, args=(ngpus_per_node,parser_args), nprocs=ngpus_per_node, join=True)
    else:
        # Simply call main_worker function
        main_worker(parser_args.gpu, ngpus_per_node, parser_args)


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


def setup_distributed(rank, ngpus_per_node, parser_args):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '{}'.format(parser_args.port)

    dist.init_process_group("nccl", rank=rank, world_size=ngpus_per_node)

def main_worker(gpu, ngpus_per_node, parser_args):
    parser_args.gpu = gpu
    if parser_args.multiprocessing_distributed:
        parser_args.rank = parser_args.gpu
        setup_distributed(parser_args.rank, ngpus_per_node, parser_args)
        # if using ddp, divide batch size per gpu
        parser_args.batch_size = int(parser_args.batch_size / ngpus_per_node)


    #########################DATA LOADING CODE#########################
    from torchvision import datasets, transforms
    import torch.multiprocessing
    from torch.utils.data import random_split
    # torch.multiprocessing.set_sharing_strategy("file_system")
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

