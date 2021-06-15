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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

#from utils.utils import set_seed

import pdb
import time
import copy
import random
plt.style.use('seaborn-whitegrid')





parser_args = None
num_neuron = 20 #100
print('num_neurons: ', num_neuron)

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
    print("Seeded everything: {}".format(seed))

def binarize_weights(orig_model, num_layer):

    #TODO: need to generalize for arbitrary network
    model = copy.deepcopy(orig_model)
    if num_layer >= 2:
        l = [model.fc1, model.fc2]
    elif num_layer >= 3:
        l.append(model.fc3)
    elif num_layer >= 4:
        l.append(model.fc4)
    elif num_layer >= 5:
        l.append(model.fc5)
    elif num_layer >= 6:
        l.append(model.fc6)

    for layer in l:
        layer.weight.data = layer.weight.data.sign()
        print(layer.weight.data)

    return model



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
            print("INVALID PRUNING ALGO")
            print("EXITING")
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
        '''
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        # self.weight.data = 2*torch.bernoulli(0.5*torch.ones_like(self.weight)) - 1
        '''
        fan = nn.init._calculate_correct_fan(self.weight, mode="fan_in") 
        # if args.scale_fan:
        #     fan = fan * (1 - args.prune_rate)
        gain = nn.init.calculate_gain(nonlinearity="relu")
        std = gain / math.sqrt(fan)
        self.weight.data = self.weight.data.sign() * std           
        print("We use signed constant in SupermaskLinear")
        print(self.weight.data)

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


# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


# Width overparam
class Net1(nn.Module):
    def __init__(self, ratio=1.0):
        super(Net1, self).__init__()
        self.fc1 = SupermaskLinear(784, (int)(num_neuron*ratio), bias=parser_args.bias)
        self.fc2 = SupermaskLinear((int)(num_neuron*ratio), 10, bias=parser_args.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net2(nn.Module):
    def __init__(self, ratio=2.0):
        super(Net2, self).__init__()
        w = (int)(num_neuron*ratio)
        self.fc1 = SupermaskLinear(784, w, bias=parser_args.bias)
        self.fc2 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc3 = SupermaskLinear(w, 10, bias=parser_args.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net3(nn.Module):
    def __init__(self, ratio=3.0):
        super(Net3, self).__init__()
        w = (int)(num_neuron*ratio)
        self.fc1 = SupermaskLinear(784, w, bias=parser_args.bias)
        self.fc2 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc3 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc4 = SupermaskLinear(w, 10, bias=parser_args.bias)


    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net4(nn.Module):
    def __init__(self, ratio=4.0):
        super(Net4, self).__init__()
        w = (int)(num_neuron*ratio)
        self.fc1 = SupermaskLinear(784, w, bias=parser_args.bias)
        self.fc2 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc3 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc4 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc5 = SupermaskLinear(w, 10, bias=parser_args.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net5(nn.Module):
    def __init__(self, ratio=5.0):
        super(Net5, self).__init__()
        w = (int)(num_neuron*ratio)
        self.fc1 = SupermaskLinear(784, w, bias=parser_args.bias)
        self.fc2 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc3 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc4 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc5 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc6 = SupermaskLinear(w, 10, bias=parser_args.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net10(nn.Module):
    def __init__(self, ratio=10.0):
        super(Net10, self).__init__()
        w = (int)(num_neuron*ratio)
        self.fc1 = SupermaskLinear(784, w, bias=parser_args.bias)
        self.fc2 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc3 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc4 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc5 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc6 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc7 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc8 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc9 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc10 = SupermaskLinear(w, w, bias=parser_args.bias)
        self.fc11 = SupermaskLinear(w, 10, bias=parser_args.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)
        x = F.relu(x)
        x = self.fc11(x)
        output = F.log_softmax(x, dim=1)
        return output

class NetNormal(nn.Module):
    # network for training
    def __init__(self):
        super(NetNormal, self).__init__()
        self.fc1 = nn.Linear(784, num_neuron, bias=parser_args.bias)
        self.fc2 = nn.Linear(num_neuron, 10, bias=parser_args.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        regularization_loss = 0
        if parser_args.regularization:
            regularization_loss =\
                get_regularization_loss(model, regularizer=parser_args.regularization,
                                        lmbda=parser_args.lmbda, alpha=parser_args.alpha,
                                        alpha_prime=parser_args.alpha_prime)

        # print("LOSS (before): {}".format(loss))
        loss += regularization_loss
        loss.backward()
        optimizer.step()
        if batch_idx % parser_args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc = 100. * correct/len(test_loader.dataset)
    return test_acc


# returns num_nonzero elements, total_num_elements so that it is easier to compute
# average sparsity in the end
def get_layer_sparsity(layer, threshold=0):
    # for algos where the score IS the mask
    if parser_args.algo in ['hc']:
        # assume the model is rounded
        num_middle = torch.sum(torch.gt(layer.scores,
                        torch.ones_like(layer.scores)*threshold) *\
                        torch.lt(layer.scores,
                        torch.ones_like(layer.scores.detach()*(1-threshold)).int()))
        if num_middle > 0:
            print("WARNING: Model scores are not binary. Sparsity number is unreliable.")
            raise ValueError
        w_numer, w_denom = layer.scores.detach().sum().item(), layer.scores.detach().flatten().numel()

        if parser_args.bias:
            b_numer, b_denom = layer.bias_scores.detach().sum().item(), layer.bias_scores.detach().flatten().numel()
        else:
            b_numer, b_denom = 0, 0
    else:
        # traditional pruning where we just check non-zero values in mask
        weight_mask, bias_mask = GetSubnet.apply(layer.scores.abs(), layer.bias_scores.abs(), parser_args.sparsity)
        w_numer, w_denom = weight_mask.sum().item(), weight_mask.flatten().numel()

        if parser_args.bias:
            b_numer, b_denom = bias_mask.sum().item(), bias_mask.flatten().numel()
            #bias_sparsity = 100.0 * bias_mask.sum().item() / bias_mask.flatten().numel()
        else:
            b_numer, b_denom = 0, 0
    return w_numer, w_denom, b_numer, b_denom


# returns avg_sparsity = number of non-zero weights!
def get_model_sparsity(model, threshold=0):
    numer = 0
    denom = 0

    # TODO: find a nicer way to do this (skip dropout)
    # TODO: Update: can't use .children() or .named_modules() because of the way things are wrapped in builder
    # TODO: for now, just write this code for each model
    for conv_layer in [model.conv1, model.conv2]:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(conv_layer, threshold)
        numer += w_numer
        denom += w_denom
        if parser_args.bias:
            numer += b_numer
            denom += b_denom

    for lin_layer in [model.fc1, model.fc2]:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(lin_layer, threshold)
        numer += w_numer
        denom += w_denom
        if parser_args.bias:
            numer += b_numer
            denom += b_denom
    # print('Overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))

    return 100*numer/denom


"""
@deprecated
# returns average sparsity of a layer -> not exactly correct
def get_model_sparsity(model, threshold=None):
    # compute mean sparsity of each layer
    # TODO: find a nicer way to do this (skip dropout)
    s1, bs1 = get_layer_sparsity(model.conv1, threshold)
    s2, bs2 = get_layer_sparsity(model.conv2, threshold)
    s3, bs3 = get_layer_sparsity(model.fc1, threshold)
    s4, bs4 = get_layer_sparsity(model.fc2, threshold)

    avg_sparsity = (s1 + s2 + s3 + s4)/4
    return avg_sparsity
"""


"""
# @deprecated
def get_model_sparsity_hc(model):
    # handle bias
    sparsity = []
    for name, params in model.named_parameters():
        if ".score" in name:
            num_middle = torch.gt(params, torch.ones_like(params)*0.01) * torch.lt(params, torch.ones_like(params)*0.99).int() # 0.25 / 0.75
            curr_sparsity = 100*torch.sum(num_middle).item()/num_middle.numel()
            sparsity.append(curr_sparsity)
            print(name, '{}/{} ({:.2f} %)'.format(torch.sum(num_middle).item(), num_middle.numel(), curr_sparsity))

    return sparsity
"""


def compute_loss(model, device, train_loader, criterion):
    model.eval()

    '''
    for name, params in model.named_parameters():
        if ".score" in name:
            print(params[0][0][0][0])
            break
    '''

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target).detach().item()
        break
    return loss


# derivative based rounding
def round_down(cp_model, params, device, train_loader, criterion):

    scores = params.data
    scores2 = torch.ones_like(scores) * -1      # initialize a dummy tensor
    sc2 = scores2.flatten()

    # check indices I that has score value of neither 0 nor 1
    sc = scores.flatten()
    flag_sc = torch.gt(sc, torch.zeros_like(sc)) * torch.lt(sc, torch.ones_like(sc)).int()

    # for i \in [n]/I, copy params values to dummy tensor
    sc2[flag_sc == 0] = sc[flag_sc == 0]

    start_time = time.time()
    # for i in I:
    # computes loss_1 & loss_0
    # depending on the difference, fill in a dummy tensor
    # temp = torch.clone(params.data.flatten())
    for idx in range(len(flag_sc)):

        if (idx+1) % 100 == 0:
            end_time = time.time()
            print(idx, end_time - start_time)

        if flag_sc[idx] == 1:
            # temp = torch.clone(params.data.flatten()[idx])
            # print(params.data[0][0][0][0])
            params.data.flatten()[idx] = 1
            # print(params.data[0][0][0][0])
            set_seed(idx)
            loss1 = compute_loss(cp_model, device, train_loader, criterion)

            params.data.flatten()[idx] = 0
            # print(params.data[0][0][0][0])
            set_seed(idx)
            loss0 = compute_loss(cp_model, device, train_loader, criterion)

            # print(loss1, loss0)

            if loss1 > loss0:   sc2[idx] = 0
            else:   sc2[idx] = 1

            params.data.flatten()[idx] = temp[idx]
            # print(params.data[0][0][0][0])
            # print(sum(scores2.flatten()))

    # print(scores2.flatten())

    return scores2


def plot_histogram_scores(model, epoch=0):
    # TODO: make this generalizable
    plt.rcParams.update({'font.size': 5})
    fig, axs = plt.subplots(2, 2)
    scores = model.conv1.scores.flatten().cpu().detach().numpy()
    axs[0, 0].hist(scores, facecolor='#2ab0ff', edgecolor='#169acf',
                   density=False, linewidth=0.5, bins=20)
    axs[0, 0].set_title('Conv1 Scores Distribution')

    scores = model.conv2.scores.flatten().cpu().detach().numpy()
    axs[0, 1].hist(scores, facecolor='#2ab0ff', edgecolor='#169acf',
                   density=False, linewidth=0.5, bins=20)
    axs[0, 1].set_title('Conv2 Scores Distribution')

    scores = model.fc1.scores.flatten().cpu().detach().numpy()
    axs[1, 0].hist(scores, facecolor='#2ab0ff', edgecolor='#169acf',
                   density=False, linewidth=0.5, bins=20)
    axs[1, 0].set_title('FC1 Scores Distribution')

    scores = model.fc2.scores.flatten().cpu().detach().numpy()
    axs[1, 1].hist(scores, facecolor='#2ab0ff', edgecolor='#169acf',
                   density=False, linewidth=0.5, bins=20)
    axs[1, 1].set_title('FC2 Scores Distribution')

    plt.tight_layout()
    algo = parser_args.algo
    reg = 'reg' if parser_args.regularization else 'noreg'
    opt = parser_args.optimizer
    filename = 'results/MNIST/weights_histogram_MNIST_{}_{}_{}_epoch_{}.pdf'.format(algo, reg, opt, epoch)
    plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.05)


def round_model(model, device, train_loader):
    cp_model = copy.deepcopy(model)
    for name, params in cp_model.named_parameters():
        if ".score" in name:
            if parser_args.round == 'naive':
                params.data = torch.gt(params, torch.ones_like(params)*0.5).int()
            elif parser_args.round == 'prob':
                params.data = torch.bernoulli(params)
            elif parser_args.round == 'pb':
                params.data = round_down(cp_model, params, device, train_loader, criterion)
                print(name, ' ended')
            else:
                print("INVALID ROUNDING")
                print("EXITING")
                exit()

    print("Rounding complete: Returning rounded model after {} rounding".format(parser_args.round))
    return cp_model


def get_regularization_loss(model, regularizer='var_red_1', lmbda=1, alpha=1, alpha_prime=1):
    def get_special_reg_sum(layer):
        # reg_loss =  \sum_{i} w_i^2 * p_i(1-p_i)
        # NOTE: alpha = alpha' = 1 here. Change if needed.
        reg_sum = 0
        w_i = layer.weight
        p_i = layer.scores
        reg_sum += torch.sum(torch.pow(w_i, 2) * torch.pow(p_i, 1) * torch.pow(1-p_i, 1))
        if parser_args.bias:
            b_i = layer.bias
            p_i = layer.bias_scores
            reg_sum += torch.sum(torch.pow(b_i, 2) * torch.pow(p_i, 1) * torch.pow(1-p_i, 1))
        return reg_sum


    regularization_loss = 0
    if regularizer == 'var_red_1':
        # reg_loss = lambda * p^{alpha} (1-p)^{alpha'}
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if parser_args.bias:
                    regularization_loss += torch.sum(torch.pow(params, alpha) * torch.pow(1-params, alpha_prime))

            elif ".score" in name:
                regularization_loss += torch.sum(torch.pow(params, alpha) * torch.pow(1-params, alpha_prime))

        regularization_loss = lmbda * regularization_loss

    elif regularizer == 'var_red_2':
        # reg_loss =  \sum_{i} w_i^2 * p_i(1-p_i)
        # NOTE: alpha = alpha' = 1 here. Change if needed.
        for conv_layer in [model.conv1, model.conv2]:
            layer = conv_layer
            regularization_loss += get_special_reg_sum(layer)

        for lin_layer in [model.fc1, model.fc2]:
            layer = lin_layer
            regularization_loss += get_special_reg_sum(layer)
        regularization_loss = lmbda * regularization_loss

    elif regularizer == 'bin_entropy':
        # reg_loss = -p \log(p) - (1-p) \log(1-p)
        # NOTE: This will be nan because log(0) = inf. therefore, replacing with 0
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if parser_args.bias:
                    regularization_loss +=\
                        torch.sum(-1.0 * params * torch.log(params).\
                            nan_to_num(posinf=0, neginf=0) - (1-params) * torch.log(params).\
                            nan_to_num(posinf=0, neginf=0))

            elif ".score" in name:
                regularization_loss +=\
                        torch.sum(-1.0 * params * torch.log(params).\
                            nan_to_num(posinf=0, neginf=0) - (1-params) * torch.log(params).\
                            nan_to_num(posinf=0, neginf=0))

        regularization_loss = lmbda * regularization_loss
    return regularization_loss


def round_and_evaluate(model, device, criterion, train_loader, test_loader):
    test(model, device, criterion, test_loader)
    # cp_model = Net().to(device)
    acc_list = []
    for itr in range(parser_args.num_test):
        cp_model = copy.deepcopy(model)
        # cp_model.load_state_dict(torch.load('model_checkpoints/mnist_pruned_model_{}_{}.pt'.format(parser_args.algo, parser_args.epochs)))
        print('Testing rounding technique of {}'.format(parser_args.round))
        for name, params in cp_model.named_parameters():
            if ".score" in name:
                if parser_args.round == 'naive':
                    params.data = torch.gt(params, torch.ones_like(params)*0.5).int()
                elif parser_args.round == 'prob':
                    params.data = torch.bernoulli(params)
                elif parser_args.round == 'pb':
                    params.data = round_down(cp_model, params, device, train_loader, criterion)
                    print(name, ' ended')
                else:
                    print("INVALID ROUNDING")
                    print("EXITING")
                    exit()

        acc = test(cp_model, device, criterion, test_loader)
        acc_list = np.append(acc_list, np.array([acc]))

    print("Rounding results: ")
    print('Mean Acc: {}, Std Dev: {}'.format(np.mean(acc_list), np.std(acc_list)))

    return np.mean(acc_list)


def main():
    global parser_args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='../data', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    parser.add_argument('--p-threshold', type=float, default=0.05,
                        help='probability threshold for pruning')
    parser.add_argument('--normalize-scores', action='store_true', default=False,
                        help='to normalize or not to normalize')
    parser.add_argument('--results-filename', type=str, default='results_acc_mnist.csv',
                        help='csv results filename')
    # ep: edge-popup, pt_hack: KS hacky probability pruning, pt_reg: probability pruning with regularization
    # hc: hypercube pruning
    parser.add_argument('--algo', type=str, default='ep',
                        help='pruning algo to use |ep|pt_hack|pt_reg|hc|')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer option to use |sgd|adam|')
    parser.add_argument('--evaluate-only', action='store_true', default=False,
                        help='just use rounding techniques to evaluate a saved model')
    parser.add_argument('--round', type=str, default='naive',
                        help='rounding technique to use |naive|prob|pb|')
    # naive: threshold(0.5), prob: probabilistic rounding, pb: pseudo-boolean paper's choice (RoundDown)
    parser.add_argument('--num-test', type=int, default=1,
                        help='number of different models testing in prob rounding')
    parser.add_argument('--mode', type=str, default="pruning",
                        help='can be used for either pruning | training.')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='can be used for either pruning | training.')
    parser.add_argument('--regularization', default=None, type=str,
                        help='which regularizer to add : |var_red_1|var_red_2|bin_cross_entropy|')
    # var_red_1: lmbda * p^(alpha) (1-p)^(alpha') | var_red_2: w^2 p(1-p) | bin_cross_entropy: -plog(1-p)?
    parser.add_argument('--lmbda', type=float, default=0.001,
                        help='regularization coefficient lambda')
    parser.add_argument("--alpha", default=1.0, type=float,
                        help="first exponent in regularizer")
    parser.add_argument("--alpha_prime", default=1.0, type=float,
                        help="second exponent in regularizer",)
    parser.add_argument("--ratio", default=1.0, type=float,
                        help="overparam ratio")
    parser.add_argument('--width-only', action='store_true', default=False,
                        help='just use width overparam')

    epoch_list = []
    test_acc_list = []
    model_sparsity_list = []

    parser_args = parser.parse_args()
    use_cuda = not parser_args.no_cuda and torch.cuda.is_available()


    results_dir = 'results/MNIST/'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    set_seed(parser_args.seed)

    device = torch.device("cuda:2" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(parser_args.data, 'mnist'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=parser_args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(parser_args.data, 'mnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=parser_args.test_batch_size, shuffle=True, **kwargs)

    if parser_args.mode == "pruning":
        if not parser_args.width_only:
            # width & depth overparam
            list_model = [Net1(), Net2(), Net3(), Net4(), Net5(), None, None, None, None, Net10()]
            model = list_model[(int)(parser_args.ratio)-1].to(device)
        else:
            # (only) width overparam
            model = Net1(parser_args.ratio).to(device)

    elif parser_args.mode == "training":
        model = NetNormal().to(device)
    else:
        raise NotImplementedError("Non-supported mode ...")
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    if parser_args.optimizer == 'sgd':
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=parser_args.lr,
            momentum=parser_args.momentum,
            weight_decay=parser_args.wd,
        )

    elif parser_args.optimizer == 'adam':
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                     lr=parser_args.lr,
                                     weight_decay=parser_args.wd,
                                     amsgrad=False,
                                     )
    else:
        print("INVALID OPTIMIZER")
        print("EXITING")
        exit()

    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=parser_args.epochs)

    if not parser_args.evaluate_only:
        for epoch in range(1, parser_args.epochs + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            if parser_args.algo in ['hc']:
                test_acc = round_and_evaluate(model, device, criterion, train_loader, test_loader)
            else:
                test_acc = test(model, device, criterion, test_loader)
            scheduler.step()
            epoch_list.append(epoch)
            test_acc_list.append(test_acc)
            if parser_args.mode != "training":
                if parser_args.algo == 'hc':
                    cp_model = round_model(model, device, train_loader)
                    model_sparsity = get_model_sparsity(cp_model)
                else:
                    model_sparsity = -1
                    #model_sparsity = get_model_sparsity(model)
                '''
                if epoch % 10 == 1:
                    plot_histogram_scores(model, epoch)
                '''
            else:
                model_sparsity = (sum([p.numel() for p in model.parameters()]))

            model_sparsity_list.append(model_sparsity)
            print("Test Acc: {:.2f}%\n".format(test_acc))
            print("---------------------------------------------------------")
            results_df = pd.DataFrame({'epoch': epoch_list, 'test_acc': test_acc_list, 'model_sparsity': model_sparsity_list})
            results_df.to_csv('results/MNIST/{}'.format(parser_args.results_filename), index=False)

        print('After binarizing')
        cp_model = binarize_weights(model, parser_args.ratio+1)
        test_acc = test(cp_model, device, criterion, test_loader)


        '''
        if parser_args.mode != "training":
            # gotta plot the final histogram as well
            plot_histogram_scores(model, epoch)
        '''

        if parser_args.save_model:
            if parser_args.mode != 'training':
                model_filename = "model_checkpoints/mnist_pruned_model_{}_{}.pt".format(parser_args.algo, parser_args.epochs)
            else:
                model_filename = "model_checkpoints/mnist_trained_model_{}.pt".format(parser_args.epochs)
            torch.save(model.state_dict(), model_filename)

    if parser_args.algo in ('hc'):
        # irrespective of evaluate_only, add an evaluate_only step
        model.load_state_dict(torch.load('model_checkpoints/mnist_pruned_model_{}_{}.pt'.format(parser_args.algo, parser_args.epochs)))
        round_acc_list = round_and_evaluate(model, device, criterion, train_loader, test_loader)

        print("Test Acc: {:.2f}%\n".format(test_acc))

    print("Experiment donezo")


if __name__ == '__main__':
    main()
