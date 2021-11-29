# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import colors as mcolors
from pylab import *
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

import logging

import pdb
import time
import copy
plt.style.use('seaborn-whitegrid')

parser_args = None

logging.basicConfig()

# set seed for experiment
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # making sure GPU runs are deterministic even if they are slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logging.info("Seeded everything: {}".format(seed))

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, bias_scores, k):
        if parser_args.algo == 'pt_hack':
            # Get the supermask by normalizing scores and "sampling" by probability
            if parser_args.normalize_scores:
                # min-max normalization so that scores are in [0, 1]
                min_score = scores.min().item()
                max_score = scores.max().item()
                scores = (scores - min_score)/(max_score - min_score)

                # repeat for bias
                min_score = bias_scores.min().item()
                max_score = bias_scores.max().item()
                bias_scores = (bias_scores - min_score)/(max_score - min_score)

            # sample using scores as probability
            # by default the probabilities are too small. artificially
            # pushing them towards 1 helps!
            MULTIPLIER = 10
            scores = torch.clamp(MULTIPLIER*scores, 0, 1)
            bias_scores = torch.clamp(MULTIPLIER*bias_scores, 0, 1)
            out = torch.bernoulli(scores)
            bias_out = torch.bernoulli(bias_scores)

        elif parser_args.algo == 'ep':
            # Get the supermask by sorting the scores and using the top k%
            out = scores.clone()
            _, idx = scores.flatten().sort()
            j = int((1 - k) * scores.numel())
            # flat_out and out access the same memory.
            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1

            # repeat for bias
            # Get the supermask by sorting the scores and using the top k%
            bias_out = bias_scores.clone()
            _, idx = bias_scores.flatten().sort()
            j = int((1 - k) * bias_scores.numel())

            # flat_out and out access the same memory.
            bias_flat_out = bias_out.flatten()
            bias_flat_out[idx[:j]] = 0
            bias_flat_out[idx[j:]] = 1

        elif parser_args.algo == 'pt':
            # sample using scores as probability
            # by default the probabilities are too small. artificially
            # pushing them towards 1 helps!
            MULTIPLIER = 10
            scores = torch.clamp(MULTIPLIER*scores, 0, 1)
            bias_scores = torch.clamp(MULTIPLIER*bias_scores, 0, 1)
            out = torch.bernoulli(scores)
            bias_out = torch.bernoulli(bias_scores)

        else:
            logging.info("INVALID PRUNING ALGO")
            logging.info("EXITING")
            exit()

        return out, bias_out

    @staticmethod
    def backward(ctx, g_1, g_2):
        # send the gradient g straight-through on the backward pass.
        return g_1, g_2, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.bias:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_scores = nn.Parameter(torch.Tensor(1))
        if parser_args.algo in ('hc'):
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        # self.weight.data = 2*torch.bernoulli(0.5*torch.ones_like(self.weight)) - 1

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        if parser_args.bias:
            self.bias.requires_grad = False

    def forward(self, x):
        if parser_args.algo in ('hc'):
            # don't need a mask here. the scores are directly multiplied with weights
            self.scores.data = torch.clamp(self.scores.data, 0.0, 1.0)
            self.bias_scores.data = torch.clamp(self.bias_scores.data, 0.0, 1.0)
            subnet = self.scores
            bias_subnet = self.bias_scores
        elif parser_args.algo in ('pt', 'pt_hacky'):
            self.scores.data = self.scores.abs()
            self.bias_scores.data = self.bias_scores.abs()
            subnet, bias_subnet = GetSubnet.apply(self.scores, self.bias_scores, parser_args.sparsity)
        else:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), parser_args.sparsity)

        w = self.weight * subnet
        if parser_args.bias:
            b = self.bias * bias_subnet
        else:
            b = self.bias
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.bias:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_scores = nn.Parameter(torch.Tensor(1))
        if parser_args.algo in ('hc'):
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        # self.weight.data = 2*torch.bernoulli(0.5*torch.ones_like(self.weight)) - 1

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        if parser_args.bias:
            self.bias.requires_grad = False

    def forward(self, x):
        if parser_args.algo in ('hc'):
            # don't need a mask here. the scores are directly multiplied with weights
            self.scores.data = torch.clamp(self.scores.data, 0.0, 1.0)
            self.bias_scores.data = torch.clamp(self.bias_scores.data, 0.0, 1.0)
            subnet = self.scores
            bias_subnet = self.bias_scores
        elif parser_args.algo in ('pt', 'pt_hacky'):
            self.scores.data = self.scores.abs()
            self.bias_scores.data = self.bias_scores.abs()
            subnet, bias_subnet = GetSubnet.apply(self.scores, self.bias_scores, parser_args.sparsity)
        else:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), parser_args.sparsity)

        w = self.weight * subnet
        if parser_args.bias:
            b = self.bias * bias_subnet
        else:
            b = self.bias
        return F.linear(x, w, b)


# width d, depth l, precision prec
# width d, depth l, precision prec
class FCBinaryGadgetNet(nn.Module):
    def __init__(self, d=50, l=5, prec=4):
        super(FCBinaryGadgetNet, self).__init__()
        self.d = d
        self.l = l
        self.prec = prec
        self.layers = nn.ModuleList([
            SuperMaskLinear(784, 784*2*prec, bias=False),
            SuperMaskLinear(784*2*prec, 784*2*prec, bias=False),
            SuperMaskLinear(784*2*prec, d, bias=False),
        ])
        for i in range(l-1):
            self.layers.append(SuperMaskLinear(d, 2*prec*d, bias=False))
            self.layers.append(SuperMaskLinear(2*prec*d, 2*prec*d, bias=False))
            if i == l-1:
                self.layers.append(SuperMaskLinear(2*prec*d, 10, bias=False))
            else:
                self.layers.append(SuperMaskLinear(2*prec*d, d, bias=False))
        
    def initialize_weights(self):
        # initialize weights appropriately
        for layer_id, layer in enumerate(self.layers):
            if layer_id % 3 == 0:
                # first layer
                for idx, row in enumerate(layer.weight):
                    if idx%(2*self.prec) < self.prec:
                        layer.weight.data[idx] = torch.zeros_like(row)
                        layer.weight.data[idx][int(idx/(2*self.prec))] = 1
                    else:
                        layer.weight.data[idx] = torch.zeros_like(row)
                        layer.weight.data[idx][int(idx/(2*self.prec))] = -1
            elif layer_id % 3 == 1:
                # second layer
                for idx, row in enumerate(layer.weight):
                    layer.weight.data[idx] = torch.zeros_like(row)
                    p = 2**(idx%self.prec)
                    layer.weight.data[idx][idx] = p
            else:
                # third layer
                for idx, row in enumerate(layer.weight):
                    layer.weight.data[idx] = torch.ones_like(row)
                    layer.weight.data[idx][int(row.size(dim=0)/2):] = 0

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
            out = F.relu(out)
        output = F.log_softmax(x, dim=1)
        return output

