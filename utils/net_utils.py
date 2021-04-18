from args import args as parser_args
from functools import partial
import os
import pdb
import pathlib
import shutil
import math

import torch
import torch.nn as nn

from utils.mask_layers import MaskLinear, MaskConv


def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None


def freeze_model_subnet(model):
    print("=> Freezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True


def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")

    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(prune_rate)
            print(f"==> Setting prune rate of {n} to {prune_rate}")


def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SubnetL1RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, temperature=1.0):
        l1_accum = 0.0
        for n, p in model.named_parameters():
            if n.endswith("scores"):
                l1_accum += (p*temperature).sigmoid().sum()

        return l1_accum


        
#### Functions used for hypercube (HC) ####
def hc_round(model, round_scheme, noise=False, ratio=0.0):

    for name, params in model.named_parameters():
        if ".score" in name:
            if round_scheme == 'naive':
                params.data = torch.gt(params.data, torch.ones_like(params.data)*0.5).int().float()
                #params.data = torch.gt(params.detach(), torch.ones_like(params.data)*0.5).int().float()
            elif round_scheme == 'prob':
                params.data = torch.clamp(params.data, 0.0, 1.0)
                params.data = torch.bernoulli(params.data).float()
            else:
                print("INVALID ROUNDING")
                print("EXITING")  

            if noise:
                delta = torch.bernoulli(torch.ones_like(params.data)*ratio)
                params.data = (params.data + delta) % 2


# @deprecated
def get_score_sparsity_hc(model):
    sparsity = []
    numer = 0
    denom = 0
    for name, scores in model.named_parameters():
        if ".score" in name:
            num_ones = torch.sum(scores.detach().flatten())
            numer += num_ones.item()
            denom += scores.numel()
            curr_sparsity = 100 * num_ones.item()/scores.numel()
            sparsity.append(curr_sparsity)
            print(name, '{}/{} ({:.2f} %)'.format((int)(num_ones.item()), scores.numel(), curr_sparsity))
    print('overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))

    return 100*numer/denom


def get_layer_sparsity(layer, threshold=0):
    # for algos where the score IS the mask
    if parser_args.algo in ['hc']:
        # num_elements \in [threshold, 1-threshold]
        num_middle = torch.gt(layer.scores,
            torch.ones_like(layer.scores)*threshold) *\
             torch.lt(layer.scores, torch.ones_like(layer.scores)*(1-threshold)).int()
        weight_sparsity = 100*torch.sum(num_middle).item()/num_middle.numel()

        if parser_args.bias:
            num_middle = torch.gt(layer.bias_scores,
                torch.ones_like(layer.bias_scores)*threshold) *\
                 torch.lt(layer.bias_scores, torch.ones_like(layer.bias_scores)*(1-threshold)).int()
            bias_sparsity = 100*torch.sum(num_middle).item()/num_middle.numel()
        else:
            bias_sparsity = 0
    else:
        # traditional pruning where we just check non-zero values in mask
        weight_mask, bias_mask = GetSubnet.apply(layer.scores.abs(), layer.bias_scores.abs(), parser_args.sparsity)
        weight_sparsity = 100.0 * weight_mask.sum().item() / weight_mask.flatten().numel()
        if parser_args.bias:
            bias_sparsity = 100.0 * bias_mask.sum().item() / bias_mask.flatten().numel()
        else:
            bias_sparsity = 0
    # TODO: handle bias sparsity also
    return weight_sparsity, bias_sparsity


# returns avg_sparsity = number of non-zero weights!
def get_model_sparsity(model, threshold=0):
    # compute mean sparsity of each layer
    # TODO: find a nicer way to do this (skip dropout)
    # TODO: Update: can't use .children() or .named_modules() because of the way things are wrapped in builder
    # TODO: for now, just write this code for each model
    s_conv = []
    bs_conv = []
    s_linear = []
    bs_linear = []
    for conv_layer in [0, 2, 5, 7]:
        s, bs = get_layer_sparsity(model.module.convs[conv_layer], threshold)
        s_conv.append(s)
        bs_conv.append(bs)

    for lin_layer in [0, 2, 4]:
        s, bs = get_layer_sparsity(model.module.linear[lin_layer], threshold)
        s_linear.append(s)
        bs_linear.append(bs)

    avg_sparsity = (sum(s_conv) + sum(s_linear))/(len(s_conv) + len(s_linear))
    return avg_sparsity


#### Functions used for greedy pruning ####
def get_sparsity(model):
    # ONLY FOR GREEDY
    if "Baseline" in model.__class__.__name__:
        return 0
    else:
        total_num_params = 0
        total_num_kept = 0
        for child in model.children():
            if isinstance(child, MaskLinear) or isinstance(child, MaskConv):
                total_num_params += torch.numel(child.mask_weight)
                total_num_kept += torch.sum(child.mask_weight * child.fixed_weight)

        return 1 - (total_num_kept / total_num_params)      # The sparsity of a network is the ratio of weights that have been removed (pruned) to all the weights in the network


def flip(model):
    params_dict = {}
    idx_dict = {}
    idx_start = 0   # Starting index for a particular flattened binary mask in the network. 

    for name, param in model.named_parameters():
        if "mask" in name:
            flat_param = param.flatten()
            params_dict[name] = flat_param
            idx_dict[name] = idx_start
            idx_start += len(flat_param)

    param_names = list(params_dict.keys())

    rand_idx = random.sample(range(idx_start), k=round(0.1 * idx_start))

    for i in range(len(rand_idx)): 
        param_idx = rand_idx[i]
        for name in reversed(param_names):
            if idx_dict[name] <= param_idx:
                params_dict[name][param_idx - idx_dict[name]] = 1 - params_dict[name][param_idx - idx_dict[name]]
                break


def step(x):
#    return 2 * (x >= 0).float() - 1
    return (x >= 0).float()


class Step(nn.Module):
    def __init__(self):
        super(Step, self).__init__()

    def forward(self, x):
        return step(x)


def zero_one_loss(output, target):
    maxk = 1
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    zero_one_loss_instance = ~pred.eq(target.view(1, -1).expand_as(pred))
    return torch.mean(zero_one_loss_instance.to(torch.float32))
