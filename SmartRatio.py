import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pandas as pd
import copy
import types
import math

from utils.net_utils import get_layers

# def count_total_parameters(net):
#     total = 0
#     for m in net.modules():
#         if isinstance(m, (nn.Linear, nn.Conv2d)):
#             total += m.weight.numel()
#     return total

# def count_fc_parameters(net):
#     total = 0
#     for m in net.modules():
#         if isinstance(m, (nn.Linear)):
#             total += m.weight.numel()
#     return total



def SmartRatio(model, sr_args, parser_args):
    ## TODO: current method assumes we are not using bias. We need to add lines for bias_scores, bias_flag, bias... 

    if 'vgg' in parser_args.arch.lower():
        resnet_flag = False
    elif 'resnet' in parser_args.arch.lower():
        resnet_flag = True
    elif 'transformer' in parser_args.arch.lower() or 'mobile' in parser_args.arch.lower():
        resnet_flag = True  # NOTE: hard code
    else:
        raise NotImplementedError("Smart Ratio only works for vgg and resnet")

    keep_ratio = 1-parser_args.smart_ratio
    linear_keep_ratio = sr_args.linear_keep_ratio

    model = copy.deepcopy(model)  # .eval()
    model.zero_grad()

    # 1. Compute the number of weights to be retrained
    conv_layers, linear_layers = get_layers(parser_args.arch, model)
    m_arr = []
    layer_num = 0
    for layer in [*conv_layers, *linear_layers]:
        layer.scores.data = torch.ones_like(layer.scores.data)
        m_arr.append(layer.weight.data.view(-1).size()[0])
        #m_dict[layer_num] = layer.weight.data.view(-1).size()[0]
        layer_num += 1
        print(layer_num, layer)

    linear_layer_num = len(linear_layers)
    if parser_args.arch == 'transformer':
        linear_layer_num = 1

    num_remain_weights = keep_ratio * sum(m_arr)
    num_layers = layer_num

    print(m_arr)
    print(sum(m_arr)) #print(sum(m_dict.values()))
    print(num_layers, ' layers')
    

    # 2. set p_l = (L-l+1)^2 + (L-l+1)
    p_arr = []
    for l in range(1, num_layers+1):
        if l > num_layers - linear_layer_num:  # hacky way applicable for resnet_kaiming.py models
            p_arr.append(linear_keep_ratio)
        elif resnet_flag:
            p_arr.append((num_layers-l+1)**2 + (num_layers-l+1))
        else:
            p_arr.append( ((num_layers-l+1)**2 + (num_layers-l+1)) / (l * l) )
   
    # 3. Find gamma such that p = 1 - \frac{ \sum_l m_l gamma p_l  }{ \sum_l m_l }

    conv_term = np.multiply(np.array(m_arr[:-linear_layer_num]), np.array(p_arr[:-linear_layer_num])).sum()
    # lin_term = m_arr[-1] * p_arr[-1]
    lin_term = np.multiply(np.array(m_arr[-linear_layer_num:]), np.array(p_arr[-linear_layer_num:])).sum()
    num_weights = sum(m_arr)
    print("conv_term:", conv_term, "lin_term:", lin_term)
    scale = (num_weights * keep_ratio - lin_term) / conv_term
    p_arr[:-1] = scale * np.array(p_arr[:-1])

    # sometimes, if the prune_ratio is too small, some layer's keep ratio may be larger than 1
    ExtraNum = 0
    for i in range(num_layers):
        size = m_arr[i]
        if i < num_layers - 1:
            if p_arr[i] >= 1:
                ExtraNum = ExtraNum + int((p_arr[i]-1) * size)
                p_arr[i] = 1
            else:
                RestNum = int((1-p_arr[i])*m_arr[i])
                if RestNum >= ExtraNum:
                    p_arr[i] = p_arr[i] + ExtraNum / m_arr[i]
                    ExtraNum = 0
                else:
                    ExtraNum = ExtraNum - RestNum
                    p_arr[i] = 1
        if ExtraNum == 0:
            break

    # if we use modified version of smart ratio
    if parser_args.sr_version >= 2:    
        if parser_args.arch.lower() != 'resnet20':
            raise NotImplementedError

        root = 'per_layer_sparsity_resnet20/'
        hc = pd.read_csv(root + 'hc_iter.csv')
        if parser_args.sr_version == 2:
            if parser_args.smart_ratio == 0.9856: # 1.44% sparsity 
                p_arr[0], p_arr[-1] = hc['1_4'].array[0]/100, hc['1_4'].array[-1]/100
            elif parser_args.smart_ratio == 0.9628: # 3.72% sparsity
                p_arr[0], p_arr[-1] = hc['3_72'].array[0]/100, hc['3_72'].array[-1]/100
            else:
                raise NotImplementedError
        elif parser_args.sr_version == 3:
            print("Check whether we are using desired csv file")
            import pdb; pdb.set_trace()
            srV3 = pd.read_csv(root + 'smart_ratio_v3_lr1e-8_manual.csv')
            if parser_args.smart_ratio == 0.9856: # 1.44% sparsity
                #import pdb; pdb.set_trace() 
                p_arr = srV3['{}.0'.format(parser_args.srV3_epoch)].tolist() 
            else:
                raise NotImplementedError
        elif parser_args.sr_version == 4:
            if parser_args.smart_ratio == 0.9856: # 1.44% sparsity 
                p_arr[0], p_arr[-1] = 1, 1
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError
    


    print("p_arr", p_arr)
    #print(sum(p_arr))

    # 4. Randomly set the mask of each layer 
    """
    layer_idx = 0
    for layer in [*conv_layers, *linear_layers]: # hacky way since linear layer is at the last part for resnet_kaiming.py models
        p_curr = p_arr[layer_idx]
        layer.flag.data = torch.bernoulli(p_curr * torch.ones_like(layer.flag.data)) 
        layer_idx += 1

        print(layer_idx, torch.sum(layer.flag.data)/layer.flag.data.view(-1).size()[0])
    """
    # This part is modified to have exact number per layer (instead of Bernoulli sampling)
    layer_idx = 0
    conv_layers, linear_layers = get_layers(arch=parser_args.arch, model=model)
    for layer in conv_layers:
        N = np.prod(layer.weight.shape)
        K = int(p_arr[layer_idx] * N)
        tmp_array = np.array([0] * (N-K) + [1] * K)
        np.random.shuffle(tmp_array)
        layer.flag.data = torch.nn.Parameter(torch.from_numpy(tmp_array).float().reshape(layer.weight.shape))
        layer.scores.data = torch.nn.Parameter(torch.ones(layer.weight.shape))
        layer_idx += 1
    for layer in linear_layers:
        N = np.prod(layer.weight.shape)
        K = int(p_arr[layer_idx] * N)
        tmp_array = np.array([0] * (N-K) + [1] * K)
        np.random.shuffle(tmp_array)
        layer.flag.data = torch.nn.Parameter(torch.from_numpy(tmp_array).float().reshape(layer.weight.shape))
        layer.scores.data = torch.nn.Parameter(torch.ones(layer.weight.shape))
        layer_idx += 1

    return model


