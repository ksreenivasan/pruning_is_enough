import sys
import copy
import math
import random

import torch

from utils.net_utils import get_sparsity, zero_one_loss
from utils.logging import log_batch

# amp in pytorch
from torch.cuda.amp import autocast


def compute_batch_loss(model, device, data, target, criterion, topk):
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = model(data)
        batch_loss = criterion(output, target).item()

    batch_acc = accuracy(output, target, topk=topk)

    return batch_loss, batch_acc


def prune_weights(model, device, train_loader, criterion, args, topk=(1, 5)):
    model.eval()
    loss = 0
    acc = [0] * len(topk)
    
    prev_masks = {name: copy.deepcopy(param) for name, param in model.named_parameters() if "mask" in name}
    
    if "random" in args.how_to_prune:
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

        if args.how_to_prune == "random_with_replacement":
            rand_idx = random.choices(range(idx_start), k=len(train_loader))
        else:
            rand_idx = random.sample(range(idx_start), k=len(train_loader))

        for batch_idx, (data, target) in enumerate(train_loader):
            param_idx = rand_idx[batch_idx]
            for name in reversed(param_names):
                if idx_dict[name] <= param_idx:
                    params_dict[name][param_idx - idx_dict[name]] = 1
                    batch_loss_keep, batch_acc_keep = compute_batch_loss(model, device, data, target, criterion, topk)

                    params_dict[name][param_idx - idx_dict[name]] = 0
                    batch_loss_prune, batch_acc_prune = compute_batch_loss(model, device, data, target, criterion, topk)
                    
                    if args.metric == "loss":
                        batch_metric_keep = batch_loss_keep
                        batch_metric_prune = batch_loss_prune
                    elif args.metric == "accuracy":
                        batch_metric_keep = batch_acc_keep[0]
                        batch_metric_prune = batch_acc_prune[0]

                    if batch_metric_prune > batch_metric_keep:
                        params_dict[name][param_idx - idx_dict[name]] = 1
                        batch_loss = batch_metric_keep
                        batch_acc = batch_acc_keep
                    else:
                        batch_loss = batch_metric_prune
                        batch_acc = batch_acc_prune
                
                    loss += batch_loss
                    acc = [sum(x) for x in zip(acc, batch_acc)]

                    break
                    
            if ((batch_idx+1) % args.log_interval) == 0:
                log_batch(batch_idx+1, len(train_loader), batch_acc[0], batch_acc[1], batch_loss, get_sparsity(model))

        avg_loss = loss / (batch_idx+1)
        avg_acc = [x / (batch_idx+1) for x in acc]

    elif "layer" in args.how_to_prune:
        net_size = sum([param.numel() for name, param in model.named_parameters() if "mask" not in name])
        if args.how_to_prune == "layer_reversed":
            named_params = reversed(list(model.named_parameters()))
        else:
            named_params = model.named_parameters()

        total_iter = 0
        if args.pruning_strategy == "activations_and_weights":
            for child in model.children():
                if hasattr(child, "mask_weight"):
                    activations_kept = len(child.pruned_activation) - sum(child.pruned_activation)
                total_iter += (activations_kept * child.in_channels) 
        else:
            total_iter = math.ceil(net_size / args.submask_size)
        
        num_iter = 0
        submask = 1
        it = iter(train_loader)
        for name, param in named_params:
            if "mask" in name:
                flat_param = param.flatten()
                module = getattr(model, name.split('.')[0])
                for i in range(math.ceil(len(flat_param) / args.submask_size)):
                    can_prune = not module.pruned_activation[math.floor(i / module.in_channels)]
                    if (args.pruning_strategy == "activations_and_weights" and can_prune) or args.pruning_strategy != "activations_and_weights":      # TODO: this if statement only works for linear layers...
                        try:
                            data, target = it.next()
                        except StopIteration:
                            it = iter(train_loader)
                            data, target = it.next()

                        start_idx = i * args.submask_size
                        end_idx = min((i+1) * args.submask_size, len(flat_param))

                        if args.submask_size > 1:
                            submask = torch.round(torch.rand(end_idx - start_idx))

                        flat_param[start_idx:end_idx] = submask
                        batch_loss_keep, batch_acc_keep = compute_batch_loss(model, device, data, target, criterion, topk)

                        flat_param[start_idx:end_idx] = 1 - submask
                        batch_loss_prune, batch_acc_prune = compute_batch_loss(model, device, data, target, criterion, topk)

                        if args.metric == "loss":
                            batch_metric_keep = batch_loss_keep
                            batch_metric_prune = batch_loss_prune
                        elif args.metric == "accuracy":
                            batch_metric_keep = batch_acc_keep[0]
                            batch_metric_prune = batch_acc_prune[0]

                        if batch_metric_prune > batch_metric_keep:
                            flat_param[start_idx:end_idx] = submask
                            batch_loss = batch_metric_keep
                            batch_acc = batch_acc_keep
                        else:
                            batch_loss = batch_metric_prune
                            batch_acc = batch_acc_prune
                            
                        loss += batch_loss
                        acc = [sum(x) for x in zip(acc, batch_acc)]

                        num_iter += 1

                        if (num_iter % args.log_interval) == 0:
                            log_batch(num_iter, total_iter, batch_acc[0], batch_acc[1], batch_loss, get_sparsity(model))
        
        avg_loss = loss / num_iter
        avg_acc = [x / num_iter for x in acc]

    else:
        print("Please correctly specify the how-to-prune argument. Can be either \"random,\" \"random_with_replacement,\" \"layer,\" or \"layer_reversed\"")
        print("Exiting ...")
        sys.exit()

    curr_masks = {name: copy.deepcopy(param) for name, param in model.named_parameters() if "mask" in name}

    hamming_dist = 0        # determine the hamming distance between previous and current masks

    for key in list(curr_masks.keys()):
        hamming_dist += (curr_masks[key].numel() - torch.sum(prev_masks[key] == curr_masks[key]))

    return avg_loss, avg_acc, hamming_dist


def prune_activations(model, device, train_loader, criterion, args, topk=(1, 5)):
    model.eval()
    loss = 0
    acc = [0] * len(topk) 
    net_size = sum([param.numel() for name, param in model.named_parameters() if "mask" not in name])
    prev_masks = {name: copy.deepcopy(param) for name, param in model.named_parameters() if "mask" in name}
    
    num_iter = 0
    it = iter(train_loader)
    for layer in model.children():
        if hasattr(layer, "mask_weight"):
            for i in range(layer.mask_weight.size()[0]):
                try:
                    data, target = it.next()
                except StopIteration:
                    it = iter(train_loader)
                    data, target = it.next()

                layer.mask_weight[i, :] = 1
                if hasattr(layer, "mask_bias"):
                    layer.mask_bias[i] = 1
                batch_loss_keep, batch_acc_keep = compute_batch_loss(model, device, data, target, criterion, topk)

                layer.mask_weight[i, :] = 0
                if hasattr(layer, "mask_bias"):
                    layer.mask_bias[i] = 0
                batch_loss_prune, batch_acc_prune = compute_batch_loss(model, device, data, target, criterion, topk)

                if args.metric == "loss":
                    batch_metric_keep = batch_loss_keep
                    batch_metric_prune = batch_loss_prune
                elif args.metric == "accuracy":
                    batch_metric_keep = batch_acc_keep[0]
                    batch_metric_prune = batch_acc_prune[0]

                if batch_metric_prune > batch_metric_keep:
                    layer.mask_weight[i, :] = 1
                    if hasattr(layer, "mask_bias"):
                        layer.mask_bias[i] = 1
                    batch_loss = batch_metric_keep
                    batch_acc = batch_acc_keep
                else:
                    batch_loss = batch_metric_prune
                    batch_acc = batch_acc_prune
                    
                loss += batch_loss
                acc = [sum(x) for x in zip(acc, batch_acc)]

                num_iter += 1

                if (num_iter % args.log_interval) == 0:
                    log_batch(num_iter, model.num_activations, batch_acc[0], batch_acc[1], batch_loss, get_sparsity(model))

    avg_loss = loss / num_iter
    avg_acc = [x / num_iter for x in acc]

    curr_masks = {name: copy.deepcopy(param) for name, param in model.named_parameters() if "mask" in name}

    hamming_dist = 0        # determine the hamming distance between previous and current masks

    for key in list(curr_masks.keys()):
        hamming_dist += (curr_masks[key].numel() - torch.sum(prev_masks[key] == curr_masks[key]))

    return avg_loss, avg_acc, hamming_dist


def simulated_annealing(model, device, train_loader, criterion, args, topk=(1, 5)):
    model.eval()
    net_size = sum([param.numel() for name, param in model.named_parameters() if "mask" not in name])
    loss = 0
    acc = [0] * len(topk)
    T = args.temp

    if args.metric == "loss":
        batch_metric_best = float("inf")
    elif args.metric == "accuracy":
        batch_metric_best = 0;

    flat_params = []
    for name, param in model.named_parameters():
        if "mask" in name:
            flat_params.append(param.flatten())
    
    num_mask_params = sum([x.numel() for x in flat_params])

    num_iter = 0
    it = iter(train_loader)

    params_dict = {}
    idx_dict = {}
    idx_start = 0   # Starting index for a particular flattened binary mask in the network. 

    for name, param in model.named_parameters():
        if "mask" in name:
            flat_param = param.flatten()
            params_dict[name] = flat_param
            idx_dict[name] = idx_start
            idx_start += len(flat_param)
            for i in range(len(flat_param)):    # initialize the mask randomly. it may be helpful to initialize in a "smarter" way
                if random.uniform(0, 1) > 0.5:
                    flat_param[i] = 1 - flat_param[i]

    param_names = list(params_dict.keys())
    rand_idx = random.choices(range(idx_start), k=args.max_iter)
  
    state_best = model.state_dict()

    num_iter = 0
    it = iter(train_loader)
    for i in range(args.max_iter):
        try:
            data, target = it.next()
        except StopIteration:
            it = iter(train_loader)
            data, target = it.next()

        param_idx = rand_idx[i]
        
        for name in reversed(param_names):
            if idx_dict[name] <= param_idx:
                batch_loss_curr, batch_acc_curr = compute_batch_loss(model, device, data, target, criterion, topk)

                params_dict[name][param_idx - idx_dict[name]] = 1 - params_dict[name][param_idx - idx_dict[name]]
                batch_loss_new, batch_acc_new = compute_batch_loss(model, device, data, target, criterion, topk)
                
                if args.metric == "loss":
                    batch_metric_curr = batch_loss_curr
                    batch_metric_new = batch_loss_new
                elif args.metric == "accuracy":
                    batch_metric_curr = batch_acc_curr[0]
                    batch_metric_new = batch_acc_new[0]

                if batch_metric_new <= batch_metric_curr:       # keep the mask changes if the new mask performs better than the current mask
                    batch_acc = batch_acc_new
                    batch_loss = batch_loss_new
                    if batch_metric_new <= batch_metric_best:
                        state_best = model.state_dict()
                        batch_metric_best = batch_metric_new
                elif math.exp((batch_metric_curr - batch_metric_new) / T) > random.uniform(0, 1):   # keep the new mask with higher loss with some probability
                    batch_acc = batch_acc_new
                    batch_loss = batch_loss_new
                else:                                                                               # revert back to the current mask otherwise
                    params_dict[name][param_idx - idx_dict[name]] = 1 - params_dict[name][param_idx - idx_dict[name]]
                    batch_acc = batch_acc_curr
                    batch_loss = batch_loss_curr

                loss += batch_loss
                acc = [sum(x) for x in zip(acc, batch_acc)]

                T *= args.td    # decrease temperature

                break
            
        num_iter += 1
        if (num_iter % args.log_interval) == 0:
            log_batch(num_iter, args.max_iter, batch_acc[0], batch_acc[1], batch_loss, get_sparsity(model))

    avg_loss = loss / num_iter
    avg_acc = [x / num_iter for x in acc]

    return avg_loss, avg_acc, float("nan")


def inference(model, device, data_loader, num_classes, criterion, batch_size, topk=(1, 5), name=""):
    model.eval()
    loss = 0
    set_size = len(data_loader.dataset)
    all_target = torch.zeros(set_size)
    all_output = torch.zeros(set_size, num_classes)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target)
            all_target[batch_idx*batch_size:min(set_size, (batch_idx+1)*batch_size)] = target
            all_output[batch_idx*batch_size:min(set_size, (batch_idx+1)*batch_size)] = output

    avg_loss = loss / math.ceil(set_size / batch_size)
    acc = accuracy(all_output, all_target, topk=topk)

    if name:
        name = name + ": "
    
    print(name + "Average Loss: {:0.4f}, Top 1 Accuracy: {:0.2f}%".format(avg_loss, acc[0]))

    return avg_loss, acc


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train(model, device, train_loader, optimizer, criterion, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % log_interval == 0:
            batch_acc = accuracy(output, target, topk=(1, 5))
            log_batch(batch_idx+1, len(train_loader), batch_acc[0], batch_acc[1], loss, 0)


def train_amp(model, device, train_loader, optimizer, criterion, epoch, log_interval, scaler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        optimizer.zero_grad()
        #loss.backward()
        scaler.scale(loss).backward()

        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        if (batch_idx+1) % log_interval == 0:
            batch_acc = accuracy(output, target, topk=(1, 5))
            log_batch(batch_idx+1, len(train_loader), batch_acc[0], batch_acc[1], loss, 0)


