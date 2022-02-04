import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
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
    print("p_arr", p_arr)
    print(sum(p_arr))

    # if we use modified version of smart ratio
    if parser_args.sr_version >= 2:    
        if parser_args.arch.lower() != 'resnet20':
            raise NotImplementedError
        if parser_args.sr_version == 2:
            # followed the result in https://github.com/ksreenivasan/results_repo_pruning/blob/master/per_layer_sparsity_resnet20/hc_iter.csv
            if parser_args.smart_ratio == 0.9856: # 1.44% sparsity
                p_arr[0], p_arr[-1] = 0.3449074074074074, 0.259375 
            elif parser_args.smart_ratio == 0.9628: # 3.72% sparsity
                p_arr[0], p_arr[-1] = 0.4143518518518518, 0.546875
            else:
                raise NotImplementedError
        elif parser_args.sr_version == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError
    

    print("p_arr", p_arr)
    print(sum(p_arr))

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

    print("p_arr", p_arr)
    print(sum(p_arr))

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


    '''
    # ========== calculate the sparsity using order statistics ============
    CNT = 0
    Num = []
    # ========== calculate the number of layers and the corresponding number of weights ============
    for idx, m in enumerate(net.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
            Num.append(m.weight.data.view(-1).size()[0])
            CNT = CNT + 1
                
    Num = torch.from_numpy(np.array(Num)).float()
        
    # ========== set ratio ============
    n = CNT
    Ratio = torch.rand(1,CNT)
    for i in range(CNT):
        k = i + 1 # 1~CNT
        Ratio[0][n-k] = (k)**2 + k
        if args.linear_decay != 0:
            Ratio[0][n-k] = k
        if args.ascend != 0:
            Ratio[0][n-k] = (n-k+1)**2 + (n-k+1)
        if args.cubic != 0:
            Ratio[0][n-k] = (k)**3

    Ratio = Ratio[0]

    num_now = 0
    total_num = 0
    linear_num = 0
    
    
    # ========== calculation and scaling ============
    i = 0
    TEST = 0
    for m in net.modules():
        if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
            if not isinstance(m,nn.Linear):
                num_now = num_now + int((Ratio[i])*Num[i])
                if args.arch != 'resnet':
                    TEST = TEST + int(Num[i]*Ratio[i]/(i+1)**2)
                else:
                    TEST = TEST + int(Num[i]*Ratio[i])
            else:
                linear_num = linear_num + Num[i]
            total_num = total_num + Num[i]
            i = i + 1

    goal_num = int(total_num * (1-args.init_prune_ratio)) - int(linear_num*linear_keep_ratio)
    # ========== since the #linear_num is much lesser than that of total_num ============
    # ========== one can just easily set balance_ratio = 1 - init_prune_ratio without hurting the performance ============
    balance_ratio = goal_num / (total_num - linear_num)
    # TEST
    k = (goal_num) / TEST
    i = 0
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            if args.arch != 'resnet':
                Ratio[i] = Ratio[i] * k / (i+1)**2
            else:
                Ratio[i] = Ratio[i] * k
            i = i + 1     
    
    
    # ========== if the prune-ratio is too small, then some keep_ratio will > 1 ============
    # ========== the easy modification ============
    ExtraNum = 0
    i = 0
    for m in net.modules():
        size = Num[i]
        if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
            if not isinstance(m,nn.Linear):
                if Ratio[i] >= 1:
                    ExtraNum = ExtraNum + int((Ratio[i]-1)*size)
                    Ratio[i] = 1
                else:
                    RestNum = int((1-Ratio[i])*Num[i])
                    if RestNum >= ExtraNum:
                        Ratio[i] = Ratio[i] + ExtraNum/Num[i]
                        ExtraNum = 0
                    else:
                        ExtraNum = ExtraNum - RestNum
                        Ratio[i] = 1
            if ExtraNum == 0:
                break
            i = i + 1
    
    # ========== set the smart-ratio masks ============
    keep_masks = []
    CNT = 0

    for m in net.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
            mask = m.weight.data.abs().clone().float().cuda()
            Size = mask.size()
            mask = mask.view(-1)
            keep_ratio = Ratio[CNT]
            num_keep = int((keep_ratio)*Num[CNT])
            if Ratio[CNT] >= 1:
                num_keep = int(Num[CNT])
            if args.uniform != 0:
                Ratio[CNT] = balance_ratio
                num_keep = int(Ratio[CNT]*Num[CNT])
            if isinstance(m,nn.Linear):
                num_keep = int(linear_keep_ratio*Num[CNT])
            # ========== this judgement is for our hybrid ticket ============
            # ========== if specify the hybrid method, our smart ratio will combine the magnitude-based pruning ============
            if args.hybrid != 0:
                print("################### DEBUG PRINT : USING HYBRID TICKET ###################")
                value,idx = torch.topk(mask,num_keep)
                temp = torch.zeros(int(Num[CNT]))
                temp[idx] = 1.0
                mask = temp.clone().float().cuda()
            
            else:
                temp = torch.ones(1,num_keep)
                mask[0:num_keep] = temp
                temp = torch.zeros(1,int(Num[CNT].item()-num_keep))
                mask[num_keep:] = temp
                mask = mask.view(-1)[torch.randperm(mask.nelement())].view(mask.size())
            
            
            
            CNT = CNT + 1
            keep_masks.append(mask.view(Size))


    return keep_masks
    '''
