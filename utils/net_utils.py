from args import args as parser_args
from functools import partial
import os
import pdb
import pathlib
import shutil
import math
import copy

import torch
import torch.nn as nn

from utils.mask_layers import MaskLinear, MaskConv
from utils.conv_type import GetSubnet as GetSubnetConv


# return layer objects of conv layers and linear layers so we can parse them
# efficiently
def get_layers(arch='Conv4', model=None):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    if arch == 'Conv4':
        conv_layers = [model.convs[0], model.convs[2], model.convs[5], model.convs[7]]
        linear_layers = [model.linear[0], model.linear[2], model.linear[4]]
    elif arch == 'resnet20':
        conv_layers = [model.conv1]
        for layer in [model.layer1, model.layer2, model.layer3]:
            for basic_block_id in [0, 1]:
                conv_layers.append(layer[basic_block_id].conv1)
                conv_layers.append(layer[basic_block_id].conv2)
                '''
                # handle shortcut
                if len(layer[basic_block_id].shortcut) > 0:
                    conv_layers.append(layer[basic_block_id].shortcut[0])
                '''
        linear_layers = [model.fc]
    elif arch == 'cResNet18':
        conv_layers = [model.conv1]
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for basic_block_id in [0, 1]:
                conv_layers.append(layer[basic_block_id].conv1)
                conv_layers.append(layer[basic_block_id].conv2)
                # handle shortcut
                if len(layer[basic_block_id].shortcut) > 0:
                    conv_layers.append(layer[basic_block_id].shortcut[0])
        linear_layers = [model.fc]
    elif arch == 'ResNet50':
        conv_layers = [model.conv1]
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for basic_block_id in [i for i in range(len(layer))]:
                conv_layers.append(layer[basic_block_id].conv1)
                conv_layers.append(layer[basic_block_id].conv2)
                conv_layers.append(layer[basic_block_id].conv3)
                # handle shortcut
#                if len(layer[basic_block_id].shortcut) > 0:
#                    conv_layers.append(layer[basic_block_id].shortcut[0])

        linear_layers = [model.fc]
    return (conv_layers, linear_layers)


def redraw(model, shuffle=False, reinit=False, invert=False, chg_mask=False, chg_weight=False):
    cp_model = copy.deepcopy(model)
    conv_layers, linear_layers = get_layers(parser_args.arch, cp_model)
    for layer in [*conv_layers, *linear_layers]:
        #print(layer)
        #print(layer.weight)
        if shuffle:
            if chg_mask:
                idx = torch.randperm(layer.flag.data.nelement())
                layer.flag.data = layer.flag.data.view(-1)[idx].view(layer.flag.data.size())
                if parser_args.bias:
                    idx = torch.randperm(layer.bias_flag.data.nelement())
                    layer.bias_flag.data = layer.bias_flag.data.view(-1)[idx].view(layer.bias_flag.data.size())

            if chg_weight:
                raise NotImplementedError
                '''
                idx = torch.randperm(layer.weight.data.nelement())
                layer.weight.data = layer.weight.data.view(-1)[idx].view(layer.weight.data.size())
                if parser_args.bias:
                    idx = torch.randperm(layer.bias.data.nelement())
                    layer.bias.data = layer.bias.data.view(-1)[idx].view(layer.bias.data.size())
                '''
        if reinit:
            if chg_weight:
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                if parser_args.bias:
                    nn.init.kaiming_normal_(layer.bias, mode="fan_in", nonlinearity="relu")
            else:
                raise NotImplementedError
        
        if invert:
            if chg_mask:
                layer.flag.data = 1 - layer.flag.data
                if parser_args.bias:
                    layer.flag_bias.data = 1 - layer.flag_bias.data
            else:
                raise NotImplementedError

        #print(layer.weight)


    return cp_model





def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False, parser_args=None):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        # added for mode connectivity
        if parser_args is not None and parser_args.mode_connect:
            print('We are saving stat_dict for checking mode connectivity: {}'.format(parser_args.mode_connect_filename))
            shutil.copyfile(filename, parser_args.mode_connect_filename)

#        if not save:
#            os.remove(filename)


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


        
# rounds model by round_scheme and returns the rounded model
def round_model(model, round_scheme, noise=False, ratio=0.0, rank=None):
    print("Rounding model with scheme: {}".format(round_scheme))
    if isinstance(model, nn.parallel.DistributedDataParallel):
        cp_model = copy.deepcopy(model.module)
    else:
        cp_model = copy.deepcopy(model)
    for name, params in cp_model.named_parameters():
        if ".score" in name:
            if noise:
                delta = torch.randn_like(params.data)*ratio
                params.data += delta

            if round_scheme == 'naive':
                params.data = torch.gt(params.data, torch.ones_like(params.data)*0.5).int().float()
                #params.data = torch.gt(params.detach(), torch.ones_like(params.data)*0.5).int().float()
            elif round_scheme == 'prob':
                params.data = torch.clamp(params.data, 0.0, 1.0)
                params.data = torch.bernoulli(params.data).float()
            elif round_scheme == 'naive_prob':
                if name == 'linear.0.scores':
                    print("I am applying prob. rounding to {}".format(name))
                    params.data = torch.clamp(params.data, 0.0, 1.0)
                    params.data = torch.bernoulli(params.data).float()
                else:
                    print("I am applying naive rounding to {}".format(name))
                    params.data = torch.gt(params.data, torch.ones_like(params.data)*0.5).int().float()

            else:
                print("INVALID ROUNDING")
                print("EXITING")  
            '''    
            if noise:
                delta = torch.bernoulli(torch.ones_like(params.data)*ratio)
                params.data = (params.data + delta) % 2
            '''

    if isinstance(model, nn.parallel.DistributedDataParallel):
        cp_model = nn.parallel.DistributedDataParallel(cp_model, device_ids=[rank], find_unused_parameters=True)

    return cp_model


"""
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
"""

"""
def prune(model):

    if parser_args.algo != 'hc_iter':
        print('not appropriate to use prune() in the current parser_args.algo')
        raise ValueError

    print('We are in prune function')
    conv_layers, linear_layers = get_layers(parser_args.arch, model)
    for layer in [*conv_layers, *linear_layers]:
        #print(layer)
        layer.flag.data = (layer.flag.data + torch.gt(layer.scores, torch.ones_like(layer.scores)*0.5).int() == 2).int()
    return 

"""

def prune(model):  # update prune() for bottom K pruning

    if parser_args.algo != 'hc_iter':
        print('not appropriate to use prune() in the current parser_args.algo')
        raise ValueError

    print('We are in prune function')
    conv_layers, linear_layers = get_layers(parser_args.arch, model)
    if parser_args.prune_type == 'FixThresholding':
        for layer in [*conv_layers, *linear_layers]:
            #print(layer)
            layer.flag.data = (layer.flag.data + torch.gt(layer.scores, torch.ones_like(layer.scores)*0.5).int() == 2).int()
    
    elif parser_args.prune_type == 'BottomK':
        # print("hi!!!!!!!!!!!!!!!!!!!!")
        import numpy as np
        n_non_zeros = 0
        for layer in [*conv_layers, *linear_layers]:
            n_non_zeros += layer.flag.data.clone().detach().cpu().numpy().sum()

        number_of_weights_to_prune = np.ceil(parser_args.prune_rate * n_non_zeros).astype(int)

        scores_weights = [layer.scores.data.clone().cpu().detach().numpy()[layer.flag.data.clone().detach().cpu().numpy().astype(bool)]
                   for layer in [*conv_layers, *linear_layers]]
        weight_vector = np.concatenate(scores_weights)
        threshold = np.sort(np.abs(weight_vector))[number_of_weights_to_prune]

        for layer in [*conv_layers, *linear_layers]:
            layer.flag.data = (layer.flag.data + torch.gt(layer.scores, torch.ones_like(layer.scores)*threshold).int() == 2).int()

        if parser_args.rewind_score and layer.saved_score is not None:
            # if we go into this branch, we will load the rewinded states of the scores
            with torch.no_grad():
                layer.score.data = copy.deepcopy(layer.saved_score.data)
                # for sanity check: the score rewind back
                print(layer.score.data)
        else:  # if we do not explicitly specify rewind_score, we will keep the score same
            pass

    return


# returns avg_sparsity = number of non-zero weights!
def get_model_sparsity(model, threshold=0):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    conv_layers, linear_layers = get_layers(parser_args.arch, model)
    numer = 0
    denom = 0

    # TODO: find a nicer way to do this (skip dropout)
    # TODO: Update: can't use .children() or .named_modules() because of the way things are wrapped in builder
    for conv_layer in conv_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(conv_layer, threshold)
        numer += w_numer
        denom += w_denom
        if parser_args.bias:
            numer += b_numer
            denom += b_denom

    for lin_layer in linear_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(lin_layer, threshold)
        numer += w_numer
        denom += w_denom
        if parser_args.bias:
            numer += b_numer
            denom += b_denom
    # print('Overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))
    return 100*numer/denom


# returns num_nonzero elements, total_num_elements so that it is easier to compute
# average sparsity in the end
def get_layer_sparsity(layer, threshold=0):
    if parser_args.algo in ['hc', 'hc_iter']:
        # assume the model is rounded, compute effective scores
        eff_scores = layer.scores * layer.flag
        eff_bias = layer.bias * layer.bias_flag
        num_middle = torch.sum(torch.gt(eff_scores,
                               torch.ones_like(eff_scores)*threshold) *
                               torch.lt(eff_scores,
                               torch.ones_like(eff_scores.detach()*(1-threshold)).int()))
        if num_middle > 0:
            print("WARNING: Model scores are not binary. Sparsity number is unreliable.")
            raise ValueError
        w_numer, w_denom = eff_scores.detach().sum().item(), eff_scores.detach().flatten().numel()

        if parser_args.bias:
            b_numer, b_denom = eff_bias_scores.detach().sum().item(), eff_bias_scores.detach().flatten().numel()
        else:
            b_numer, b_denom = 0, 0

    elif parser_args.algo in ['ep']:
        # NOTE: hard-coded
        w_numer, w_denom = parser_args.prune_rate * 100, 100
        b_numer, b_denom = 0, 0
    else:
        # traditional pruning where we just check non-zero values in mask
        weight_mask, bias_mask = GetSubnetConv.apply(layer.scores.abs(), layer.scores_bias.abs(), parser_args.prune_rate)
        w_numer, w_denom = weight_mask.sum().item(), weight_mask.flatten().numel()

        if parser_args.bias:
            b_numer, b_denom = bias_mask.sum().item(), bias_mask.flatten().numel()
            # bias_sparsity = 100.0 * bias_mask.sum().item() / bias_mask.flatten().numel()
        else:
            b_numer, b_denom = 0, 0
    return w_numer, w_denom, b_numer, b_denom


def get_regularization_loss(model, regularizer='var_red_1', lmbda=1, alpha=1, alpha_prime=1):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    conv_layers, linear_layers = get_layers(parser_args.arch, model)
    def get_special_reg_sum(layer):
        # reg_loss =  \sum_{i} w_i^2 * p_i(1-p_i)
        # NOTE: alpha = alpha' = 1 here. Change if needed.
        reg_sum = torch.tensor(0.).cuda()
        w_i = layer.weight
        p_i = layer.scores
        reg_sum += torch.sum(torch.pow(w_i, 2) * torch.pow(p_i, 1) * torch.pow(1-p_i, 1))
        if parser_args.bias:
            b_i = layer.bias
            p_i = layer.bias_scores
            reg_sum += torch.sum(torch.pow(b_i, 2) * torch.pow(p_i, 1) * torch.pow(1-p_i, 1))
        return reg_sum

    regularization_loss = torch.tensor(0.).cuda()
    if regularizer == 'var_red_1':
        # reg_loss = lambda * p^{alpha} (1-p)^{alpha'}
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if parser_args.bias:
                    regularization_loss += torch.sum(torch.pow(params, alpha) * torch.pow(1-params, alpha_prime))

            elif ".score" in name:
                #import pdb; pdb.set_trace()
                regularization_loss += torch.sum(torch.pow(params, alpha) * torch.pow(1-params, alpha_prime))

        regularization_loss = lmbda * regularization_loss

    elif regularizer == 'var_red_2':
        # reg_loss =  \sum_{i} w_i^2 * p_i(1-p_i)
        # NOTE: alpha = alpha' = 1 here. Change if needed.
        for conv_layer in conv_layers:
            regularization_loss += get_special_reg_sum(conv_layer)

        for lin_layer in linear_layers:
            regularization_loss += get_special_reg_sum(lin_layer)
        regularization_loss = lmbda * regularization_loss

    elif regularizer == 'bin_entropy':
        # reg_loss = -p \log(p) - (1-p) \log(1-p)
        # NOTE: This will be nan because log(0) = inf. therefore, ignoring the end points
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if parser_args.bias:
                    params_filt = params[(params > 0) & (params < 1)]
                    regularization_loss +=\
                        torch.sum(-1.0 * params_filt * torch.log(params_filt)
                            - (1-params_filt) * torch.log(1-params_filt))

            elif ".score" in name:
                params_filt = params[(params > 0) & (params < 1)]
                regularization_loss +=\
                        torch.sum(-1.0 * params_filt * torch.log(params_filt)
                            - (1-params_filt) * torch.log(1-params_filt))

        regularization_loss = lmbda * regularization_loss

    return regularization_loss


#### Functions used for greedy pruning ####
def get_sparsity(model):
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
