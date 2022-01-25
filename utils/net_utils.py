from args_helper import parser_args
from functools import partial
import os
import pdb
import pathlib
import shutil
import math
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from utils.mask_layers import MaskLinear, MaskConv
from utils.conv_type import GetSubnet as GetSubnetConv
from utils.conv_type import SubnetConv


# return layer objects of conv layers and linear layers so we can parse them
# efficiently
def get_layers(arch='Conv4', model=None):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    if arch == 'Conv4':
        conv_layers = [model.convs[0], model.convs[2],
                       model.convs[5], model.convs[7]]
        linear_layers = [model.linear[0], model.linear[2], model.linear[4]]

    elif arch == 'MobileNetV2':
        conv_layers = [model.conv1]
        for i in range(len(model.layers)):
            conv_layers.append(model.layers[i].conv1)
            conv_layers.append(model.layers[i].conv2)
            conv_layers.append(model.layers[i].conv3)
            if len(model.layers[i].shortcut) != 0:
                conv_layers.append(model.layers[i].shortcut[0])
        conv_layers.append(model.conv2)
        linear_layers = [model.linear]
        
    elif arch == 'resnet20':
        conv_layers = [model.conv1]
        for layer in [model.layer1, model.layer2, model.layer3]:
            for basic_block_id in [0, 1, 2]:
                conv_layers.append(layer[basic_block_id].conv1)
                conv_layers.append(layer[basic_block_id].conv2)
                '''
                # handle shortcut
                if len(layer[basic_block_id].shortcut) > 0:
                    conv_layers.append(layer[basic_block_id].shortcut[0])
                '''
        linear_layers = [model.fc]
    elif arch in ['resnet32', 'resnet32_double']:
        conv_layers = [model.conv1]
        for layer in [model.layer1, model.layer2, model.layer3]:
            for basic_block_id in [0, 1, 2, 3, 4]:
                conv_layers.append(layer[basic_block_id].conv1)
                conv_layers.append(layer[basic_block_id].conv2)
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

    elif arch == 'TinyResNet18':
        conv_layers = [model.conv1]
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for basic_block_id in [0, 1]:
                conv_layers.append(layer[basic_block_id].conv1)
                conv_layers.append(layer[basic_block_id].conv2)
        linear_layers = [model.fc]

    elif arch == 'ResNet50':
        conv_layers = [model.conv1]
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for basic_block_id in [i for i in range(len(layer))]:
                conv_layers.append(layer[basic_block_id].conv1)
                conv_layers.append(layer[basic_block_id].conv2)
                conv_layers.append(layer[basic_block_id].conv3)
                # handle shortcut
                # if len(layer[basic_block_id].shortcut) > 0:
                #     conv_layers.append(layer[basic_block_id].shortcut[0])
        linear_layers = [model.fc]
        if parser_args.uv_decomp:
            linear_layers.append(model.fc2)

    elif arch == 'vgg16':
        conv_layers = []
        for i in range(len(model.features)):
            if isinstance(model.features[i], SubnetConv):
                conv_layers.append(model.features[i])
        # check how to see how the model.features object works and if this is correct
        linear_layers = [model.classifier]

    elif arch == 'WideResNet28':
        conv_layers = [model.conv1]
        for block in [model.block1, model.block2, model.block3]:
            layer = block.layer
            for basic_block_id in range(len(layer)):
                conv_layers.append(layer[basic_block_id].conv1)
                conv_layers.append(layer[basic_block_id].conv2)
                # handle shortcut. this will pick up the conv layer in layer0
                if layer[basic_block_id].convShortcut:
                    conv_layers.append(layer[basic_block_id].convShortcut)
        linear_layers = [model.fc]
    
    elif arch == 'transformer':
        conv_layers = []
        linear_layers = []
        for layer in model.transformer_encoder.layers:
            linear_layers.append(layer.attn.query)
            linear_layers.append(layer.attn.key)
            linear_layers.append(layer.attn.value)
            linear_layers.append(layer.mlp.fc1)
            linear_layers.append(layer.mlp.fc2)
<<<<<<< HEAD
        linear_layers.append(model.decoder)
=======
        # linear_layers.append(model.decoder)
>>>>>>> SmartRatio_liu
    return (conv_layers, linear_layers)



def get_bn_layers(arch='ResNet50', model=None):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module

    if arch == 'ResNet50':
        bn_layers = [model.bn1]
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for basic_block_id in [i for i in range(len(layer))]:
                bn_layers.append(layer[basic_block_id].bn1)
                bn_layers.append(layer[basic_block_id].bn2)
                bn_layers.append(layer[basic_block_id].bn3)

    return bn_layers






def redraw(model, shuffle=False, reinit=False, invert=False, chg_mask=False, chg_weight=False):
    cp_model = copy.deepcopy(model)
    conv_layers, linear_layers = get_layers(parser_args.arch, cp_model)
    for layer in (conv_layers + linear_layers):
        if shuffle:
            if chg_mask:
                idx = torch.randperm(layer.flag.data.nelement())
                layer.flag.data = layer.flag.data.view(
                    -1)[idx].view(layer.flag.data.size())
                if parser_args.bias:
                    idx = torch.randperm(layer.bias_flag.data.nelement())
                    layer.bias_flag.data = layer.bias_flag.data.view(
                        -1)[idx].view(layer.bias_flag.data.size())

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
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="relu")
                if parser_args.bias:
                    nn.init.kaiming_normal_(
                        layer.bias, mode="fan_in", nonlinearity="relu")
            else:
                raise NotImplementedError

        if invert:
            if chg_mask:
                layer.flag.data = 1 - layer.flag.data
                if parser_args.bias:
                    layer.flag_bias.data = 1 - layer.flag_bias.data
            else:
                raise NotImplementedError

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
            print('We are saving stat_dict for checking mode connectivity: {}'.format(
                parser_args.mode_connect_filename))
            shutil.copyfile(filename, parser_args.mode_connect_filename)

#        if not save:
#            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            # print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                # print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                # print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    # print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None


def freeze_model_subnet(model):
    print("=> Freezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            # print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                # print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            # print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                # print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            # print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True


def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")

    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(prune_rate)
            # print(f"==> Setting prune rate of {n} to {prune_rate}")


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
                params.data = torch.gt(params.data, torch.ones_like(
                    params.data)*parser_args.quantize_threshold).int().float()
            elif round_scheme == 'prob':
                params.data = torch.clamp(params.data, 0.0, 1.0)
                params.data = torch.bernoulli(params.data).float()
            elif round_scheme == 'naive_prob':
                if name == 'linear.0.scores':
                    # print("Applying prob. rounding to {}".format(name))
                    params.data = torch.clamp(params.data, 0.0, 1.0)
                    params.data = torch.bernoulli(params.data).float()
                else:
                    # print("Applying naive rounding to {}".format(name))
                    params.data = torch.gt(params.data, torch.ones_like(
                        params.data)*0.5).int().float()
            elif round_scheme == 'all_ones':
                params.data = torch.ones_like(params.data)
            else:
                print("INVALID ROUNDING")
                print("EXITING")
            '''    
            if noise:
                delta = torch.bernoulli(torch.ones_like(params.data)*ratio)
                params.data = (params.data + delta) % 2
            '''

    if isinstance(model, nn.parallel.DistributedDataParallel):
        cp_model = nn.parallel.DistributedDataParallel(
            cp_model, device_ids=[rank], find_unused_parameters=True)

    return cp_model


"""
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


def prune(model, update_thresholds_only=False, update_scores=False):
    if update_thresholds_only:
        pass
        #print("Updating prune thresholds")
    else:
        print("Pruning Model:")

    scores_threshold = bias_scores_threshold = -np.inf
    if parser_args.algo not in ['hc_iter', 'global_ep', 'global_ep_iter']:
        print('Warning: Using prune() in {}. Are you sure?'.format(parser_args.algo))

    conv_layers, linear_layers = get_layers(parser_args.arch, model)

    if parser_args.prune_type == 'FixThresholding':
        if parser_args.algo == 'hc_iter':  # and update_thresholds_only == False:
            # prune weights that would be rounded to 0
            for layer in (conv_layers + linear_layers):
                layer.flag.data = (layer.flag.data + torch.gt(layer.scores,
                                   torch.ones_like(layer.scores)*0.5).int() == 2).int()
        else:
            raise NotImplementedError

    # prune the bottom k of scores.abs()
    elif parser_args.prune_type == 'BottomK':
        num_active_weights = 0
        num_active_biases = 0
        active_scores_list = []
        active_bias_scores_list = []
        for layer in (conv_layers + linear_layers):
            num_active_weights += layer.flag.data.sum().item()
            active_scores = (layer.scores.data[layer.flag.data == 1]).clone()
            active_scores_list.append(active_scores)
            if parser_args.bias:
                num_active_biases += layer.bias_flag.data.sum().item()
                active_biases = (
                    layer.bias_scores.data[layer.bias_flag.data == 1]).clone()
                active_bias_scores_list.append(active_biases)

        number_of_weights_to_prune = np.ceil(
            parser_args.prune_rate * num_active_weights).astype(int)
        number_of_biases_to_prune = np.ceil(
            parser_args.prune_rate * num_active_biases).astype(int)

        agg_scores = torch.cat(active_scores_list)
        agg_bias_scores = torch.cat(
            active_bias_scores_list) if parser_args.bias else torch.tensor([])

        # if invert_sanity_check, then threshold is based on sorted scores in descending order, and we prune all scores ABOVE it
        scores_threshold = torch.sort(
            torch.abs(agg_scores), descending=parser_args.invert_sanity_check).values[number_of_weights_to_prune-1].item()
<<<<<<< HEAD
=======
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", scores_threshold)
>>>>>>> SmartRatio_liu

        if parser_args.bias:
            bias_scores_threshold = torch.sort(
                torch.abs(agg_bias_scores), descending=parser_args.invert_sanity_check).values[number_of_biases_to_prune-1].item()
        else:
            bias_scores_threshold = -1

        if update_thresholds_only:
            for layer in (conv_layers + linear_layers):
                layer.scores_prune_threshold = scores_threshold
            if parser_args.bias:
                layer.bias_scores_prune_threshold = bias_scores_threshold

        else:
            for layer in (conv_layers + linear_layers):
                if parser_args.invert_sanity_check:
                    layer.flag.data = (layer.flag.data + torch.lt(layer.scores,
                                       torch.ones_like(layer.scores)*scores_threshold).int() == 2).int()
                else:
                    layer.flag.data = (layer.flag.data + torch.gt(layer.scores,
                                       torch.ones_like(layer.scores)*scores_threshold).int() == 2).int()
                if update_scores:
                    layer.scores.data = layer.scores.data * layer.flag.data
                if parser_args.bias:
                    if parser_args.invert_sanity_check:
                        layer.bias_flag.data = (layer.bias_flag.data + torch.lt(layer.bias_scores, torch.ones_like(
                            layer.bias_scores)*bias_scores_threshold).int() == 2).int()
                    else:
                        layer.bias_flag.data = (layer.bias_flag.data + torch.gt(layer.bias_scores, torch.ones_like(
                            layer.bias_scores)*bias_scores_threshold).int() == 2).int()
                    if update_scores:
                        layer.bias_scores.data = layer.bias_scores.data * layer.bias_flag.data

            if parser_args.rewind_score and layer.saved_scores is not None:
                # if we go into this branch, we will load the rewinded states of the scores
                with torch.no_grad():
                    layer.scores.data = copy.deepcopy(layer.saved_scores.data)
                    if parser_args.bias:
                        # TODO: this will probably break
                        layer.bias_scores.data = copy.deepcopy(
                            layer.saved_bias_scores.data)
                    # for sanity check: the score rewind back
                    # print(layer.scores.data)  # yes, it always rewind back to the same score, the saved score does not change
            else:  # if we do not explicitly specify rewind_score, we will keep the score same
                pass

    return scores_threshold, bias_scores_threshold


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
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(
            conv_layer, threshold)
        numer += w_numer
        denom += w_denom
        if parser_args.bias:
            numer += b_numer
            denom += b_denom

    for lin_layer in linear_layers:
        w_numer, w_denom, b_numer, b_denom = get_layer_sparsity(
            lin_layer, threshold)
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
    if parser_args.algo in ['hc', 'hc_iter'] and not parser_args.bottom_k_on_forward:
        # assume the model is rounded, compute effective scores
        eff_scores = layer.scores * layer.flag
        if parser_args.bias:
            eff_bias_scores = layer.bias_scores * layer.bias_flag
        num_middle = torch.sum(torch.gt(eff_scores,
                               torch.ones_like(eff_scores)*threshold) *
                               torch.lt(eff_scores,
                               torch.ones_like(eff_scores.detach()*(1-threshold)).int()))
        if num_middle > 0:
            print("WARNING: Model scores are not binary. Sparsity number is unreliable.")
            raise ValueError
        w_numer, w_denom = eff_scores.detach().sum(
        ).item(), eff_scores.detach().flatten().numel()

        if parser_args.bias:
            b_numer, b_denom = eff_bias_scores.detach().sum(
            ).item(), eff_bias_scores.detach().flatten().numel()
        else:
            b_numer, b_denom = 0, 0

    elif parser_args.algo in ['global_ep', 'ep', 'global_ep_iter'] or parser_args.bottom_k_on_forward:
        if parser_args.algo == 'ep':
            weight_mask, bias_mask = GetSubnetConv.apply(
                layer.scores.abs(), layer.bias_scores.abs(), parser_args.prune_rate)
        else:
            weight_mask, bias_mask = GetSubnetConv.apply(layer.scores.abs(), layer.bias_scores.abs(
            ), 0, layer.scores_prune_threshold, layer.bias_scores_prune_threshold)
        w_numer, w_denom = weight_mask.sum().item(), weight_mask.flatten().numel()

        if parser_args.bias:
            b_numer, b_denom = bias_mask.sum().item(), bias_mask.flatten().numel()
        else:
            b_numer, b_denom = 0, 0
    else:
        # traditional pruning where we just check non-zero values in mask
        weight_mask, bias_mask = GetSubnetConv.apply(
            layer.scores.abs(), layer.bias_scores.abs(), parser_args.prune_rate)
        w_numer, w_denom = weight_mask.sum().item(), weight_mask.flatten().numel()

        if parser_args.bias:
            b_numer, b_denom = bias_mask.sum().item(), bias_mask.flatten().numel()
            # bias_sparsity = 100.0 * bias_mask.sum().item() / bias_mask.flatten().numel()
        else:
            b_numer, b_denom = 0, 0
    return w_numer, w_denom, b_numer, b_denom


def get_regularization_loss(model, regularizer='L2', lmbda=1, alpha=1, alpha_prime=1):
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model = model.module
    conv_layers, linear_layers = get_layers(parser_args.arch, model)

    def get_special_reg_sum(layer):
        # reg_loss =  \sum_{i} w_i^2 * p_i(1-p_i)
        # NOTE: alpha = alpha' = 1 here. Change if needed.
        reg_sum = torch.tensor(0.).cuda()
        w_i = layer.weight
        p_i = layer.scores
        reg_sum += torch.sum(torch.pow(w_i, 2) *
                             torch.pow(p_i, 1) * torch.pow(1-p_i, 1))
        if parser_args.bias:
            b_i = layer.bias
            p_i = layer.bias_scores
            reg_sum += torch.sum(torch.pow(b_i, 2) *
                                 torch.pow(p_i, 1) * torch.pow(1-p_i, 1))
        return reg_sum

    #pdb.set_trace()
    regularization_loss = torch.tensor(0.).cuda()
    if regularizer == 'L2':
        # reg_loss =  ||p||_2^2
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if parser_args.bias:
                    regularization_loss += torch.norm(params, p=2)**2

            elif ".score" in name:
                regularization_loss += torch.norm(params, p=2)**2
        regularization_loss = lmbda * regularization_loss

    elif regularizer == 'L1':
        # reg_loss =  ||p||_1
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if parser_args.bias:
                    regularization_loss += torch.norm(params, p=1)

            elif ".score" in name:
                regularization_loss += torch.norm(params, p=1)
        regularization_loss = lmbda * regularization_loss

    elif regularizer == 'L1_L2':
        # reg_loss =  ||p||_1 + ||p||_2^2
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if parser_args.bias:
                    regularization_loss += torch.norm(params, p=1)
                    regularization_loss += torch.norm(params, p=2)**2

            elif ".score" in name:
                regularization_loss += torch.norm(params, p=1)
                regularization_loss += torch.norm(params, p=2)**2
        regularization_loss = lmbda * regularization_loss

    elif regularizer == 'var_red_1':
        # reg_loss = lambda * p^{alpha} (1-p)^{alpha'}
        for name, params in model.named_parameters():
            if ".bias_score" in name:
                if parser_args.bias:
                    regularization_loss += torch.sum(
                        torch.pow(params, alpha) * torch.pow(1-params, alpha_prime))

            elif ".score" in name:
                #import pdb; pdb.set_trace()
                regularization_loss += torch.sum(
                    torch.pow(params, alpha) * torch.pow(1-params, alpha_prime))

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

    #print('red loss: ', regularization_loss)

    return regularization_loss

# note target sparsity is MAX PERCENTAGE of weights remaining at the end of training
# parser_args.prune_rate is a fraction i.e; 1/100*percentage


def get_prune_rate(target_sparsity=0.5, iter_period=5):
    print("Computing prune_rate for target_sparsity {} with iter_period {}".format(
        target_sparsity, iter_period))
    max_epochs = parser_args.epochs
    num_prune_iterations = np.floor((max_epochs-1)/iter_period)
    # if algo is HC, iter_HC or anything that uses prune() then, prune_rate represents number of weights to prune
    prune_rate = 1 - np.exp(np.log(target_sparsity/100)/num_prune_iterations)
    return prune_rate

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
                total_num_kept += torch.sum(child.mask_weight *
                                            child.fixed_weight)

        # The sparsity of a network is the ratio of weights that have been removed (pruned) to all the weights in the network
        return 1 - (total_num_kept / total_num_params)


def flip(model):
    params_dict = {}
    idx_dict = {}
    # Starting index for a particular flattened binary mask in the network.
    idx_start = 0

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
                params_dict[name][param_idx - idx_dict[name]] = 1 - \
                    params_dict[name][param_idx - idx_dict[name]]
                break


def step(x):
    # return 2 * (x >= 0).float() - 1
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


def load_pretrained_imagenet(model, dataloader):


    pretrained = imagenet_ResNet50(pretrained=True).cuda()
    model_s = pretrained.model # source model
    #model_s = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet') # source model
    #model_s = model_s.cuda()

    #for param_tensor in model_s.state_dict():
    #    print(param_tensor, "\t", model_s.state_dict()[param_tensor].size())
    PATH = 'pretrained_model_imagenet.pth'
    torch.save(model_s.state_dict(), PATH)
    model.load_state_dict(torch.load(PATH), strict=False)

    # test the consistency of model and pretrained model
    model = model.cuda()
    x = torch.rand(16,3,224,224).cuda() # random dataset
    z1 = model.forward(x, hidden=True)
    z2 = pretrained(x, hidden=True)
    #z2 = model_s.features(x)
    print('Compare hidden feature: ', (z1 == z2).all())


    # load the final layer
    #num_classes = pretrained.l0.weight.shape[0]
    #model.fc.weight.data = pretrained.l0.weight.data.view(num_classes, -1, 1, 1)
    #model.fc.bias.data = pretrained.l0.bias.data
    '''
    y1 = model(x)
    y2 = pretrained(x)
    print('Compare prediction: ', torch.norm(y1 - y2))
    print('Note: this is small only if we turn off dropout at both loaded/our models')
    #pdb.set_trace()

    print('pretrained model on transfer task')
    val_loss, val_accuracy = validate(model_s, dataloader)
    '''

    return model
    

class imagenet_ResNet50(nn.Module):
    def __init__(self, pretrained):
        super(imagenet_ResNet50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained = None)
        # change the classification layer
        self.l0= nn.Linear(2048, 101)
        self.dropout = nn.Dropout2d(0.4)
        
    def forward(self, x, hidden=False):
        # get the batch size only, ignore(c, h, w)
        batch, _, _, _ = x.size()
        x = self.model.features(x)
        if hidden:
            return x
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0


'''
def test_and_load_pretrained_imagenet(model, dataloader):
    model = model.cuda()
    x = torch.rand(16,3,224,224).cuda() # random dataset
    z1 = model.forward(x, hidden=True)

    # load pytorch pretrained model (imagenet)
   # imagenet_model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
   # imagenet_model = imagenet_model.cuda()
    imagenet_model = imagenet_ResNet50(pretrained=True).cuda()
    #z2 = imagenet_model.forward(x, hidden=True)

    # check initial model
    print('our initial model on transfer task')
    val_loss, val_accuracy = validate(model, dataloader)
    
    print('pretrained model on transfer task')
    val_loss, val_accuracy = validate(imagenet_model, dataloader)

    # copy weights from imagenet_model to model
    #import pdb; pdb.set_trace()
    conv, lin = get_layers('ResNet50', model)
    layers = [*conv, *lin]
    conv2, lin2 = get_layers('ResNet50', imagenet_model.model)
    layers2 = [*conv2, *lin2]

    for target_layer, source_layer in zip(layers, layers2):
        if source_layer is None:
            continue
        #print(target_layer, source_layer)
        assert(target_layer.weight.data.shape == source_layer.weight.data.shape)
        target_layer.weight.data = source_layer.weight.data    

    bn = get_bn_layers('ResNet50', model)
    bn2 = get_bn_layers('ResNet50', imagenet_model.model)
    
    for target_layer, source_layer in zip(bn, bn2):
        assert(target_layer.weight.data.shape == source_layer.weight.data.shape)
        target_layer.weight.data = source_layer.weight.data    
        target_layer.bias.data = source_layer.bias.data    
    z3 = model.forward(x, hidden=True)

    # check updated model
    print('our updated model on transfer task')
    val_loss, val_accuracy = validate(model, dataloader)

    print((z1 == z2).all())
    print((z3 == z2).all())
    print((z3 == z2).all())
    import pdb; pdb.set_trace()

    return model
'''

#validation function
def validate(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            #loss = criterion(outputs, torch.max(target, 1)[1])
            loss = criterion(outputs, target)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            #running_correct += (preds == torch.max(target, 1)[1]).sum().item()
            running_correct += (preds == target).sum().item()

        loss = running_loss/len(dataloader.dataset)
        accuracy = 100. * running_correct/len(dataloader.dataset)
        print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}')
        
        return loss, accuracy

