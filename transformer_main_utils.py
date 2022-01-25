# coding: utf-8
import argparse
import time
import math
import os
import re
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.onnx
import pandas as pd

import transformer_data as data
import transformer_model

from utils.utils import set_seed
from utils.builder import get_builder
from utils.schedulers import get_scheduler
from args_helper import parser_args
from main_utils import switch_to_wt, set_gpu, get_optimizer, dotdict
from utils.net_utils import prune, get_prune_rate, round_model, get_regularization_loss
from SmartRatio import SmartRatio


def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def print_nonzeros(model):
    nonzero = 0
    total = 0
    for name, p in model.named_parameters():
        # if (re.match('.*\.flag', name) or re.match('.*\.bias_flag', name)) and 'decoder' not in name:
        if re.match('.*.flag', name) and not re.match('.*.bias_flag', name)  and 'decoder' not in name:
            tensor = p.data.detach().cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, ({100 * nonzero / total:6.2f}% remained)')
    return nonzero / total


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(parser_args.transformer_bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(parser_args, model, ntokens, criterion, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, parser_args.transformer_bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def export_onnx(batch_size, seq_len):
    onnx_path = os.path.join("results", parser_args.subfolder, "onnx_output.onnx")
    print('The model is also exported in ONNX format at {}'.format(onnx_path))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), onnx_path)


def train(parser_args, epoch, ntokens, train_data, model, optimizer, criterion, mask=None, mask_bias=None):
    # this should be incorporated into trainer/default
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, parser_args.transformer_bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)

        regularization_loss = torch.tensor(0)
        if parser_args.regularization:
            regularization_loss = get_regularization_loss(model, regularizer=parser_args.regularization,
                                        		  lmbda=parser_args.lmbda, alpha=parser_args.alpha,
                                        		  alpha_prime=parser_args.alpha_prime)
        loss += regularization_loss
        loss.backward()

        # GOOD
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), parser_args.transformer_clip)
       
        # lr = 5.
        # import ipdb; ipdb.set_trace()
        # for p in model.parameters():
        #     if p.requires_grad:
        #         p.data.add_(p.grad, alpha=-lr)
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]

        total_loss += loss.item()

        if mask is not None:
            for name, param in model.named_parameters():
                if name in model.prunable_layer_names:
                    tensor = param.data.detach()
                    param.data = tensor * mask[name].to(param.device).float()
        if mask_bias is not None:
            for name, param in model.named_parameters():
                if name in model.prunable_biases:
                    tensor = param.data.detach()
                    param.data = tensor * mask[name].to(param.device).float()

        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            # cur_loss = total_loss / log_interval
            cur_loss = loss.item()
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // parser_args.transformer_bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def finetune(parser_args, ntokens, model, criterion, train_data, val_data, test_data, old_epoch_list, old_val_acc_list, old_test_acc_list, old_model_sparsity_list):

    model = switch_to_wt(model)
    parser_args.regularization = False
    best_val_loss = None

    epoch_list = copy.deepcopy(old_epoch_list)
    val_acc_list = copy.deepcopy(old_val_acc_list)
    model_sparsity_list = copy.deepcopy(old_model_sparsity_list)
    test_acc_list = copy.deepcopy(old_test_acc_list)

    if parser_args.unflag_before_finetune:
        model = round_model(model, round_scheme="all_ones", noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu, name_prefix=None)

    optimizer = get_optimizer(parser_args, model, finetune_flag=True)
    scheduler = get_scheduler(optimizer, parser_args.fine_tune_lr_policy) 

    for epoch in range(parser_args.epochs, parser_args.epochs * 2):
        epoch_list.append(epoch)
        epoch_start_time = time.time()
        train(parser_args, epoch, ntokens, train_data, model, optimizer, criterion)
        scheduler.step()

        val_loss = evaluate(parser_args, model, ntokens, criterion, val_data)
        val_acc_list.append(val_loss)
        test_loss = evaluate(parser_args, model, ntokens, criterion, test_data)
        test_acc_list.append(test_loss)
        avg_sparsity = -1
        model_sparsity_list.append(avg_sparsity)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join("results", parser_args.subfolder, "finetune_model.pt"), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            pass
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] /= 4.0

        result_df = pd.DataFrame({'val': val_acc_list, 'nact': model_sparsity_list, "test": test_acc_list})
        if name_prefix is None:
            result_df.to_csv("results/{}/acc_and_sparsity.csv".format(parser_args.subfolder), index=False)
        else:
            result_df.to_csv("results/{}/acc_and_sparsity_{}.csv".format(parser_args.subfolder, name_prefix), index=False)

    with open(os.path.join("results", parser_args.subfolder, "finetune_model.pt"), 'rb') as f:
        model = torch.load(f)

        # Run on test data.
        test_loss = evaluate(parser_args, model, ntokens, criterion, test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
                                            test_loss, math.exp(test_loss)))
        print('=' * 89)
    return model


def test_random_subnet(parser_args, ntokens, model, criterion, train_data, val_data, test_data):
    # get a randomly pruned model with SmartRatio
    smart_ratio_args = {'linear_keep_ratio': 0.3, }
    smart_ratio_args = dotdict(smart_ratio_args)
    model = SmartRatio(model, smart_ratio_args, parser_args)
    model = set_gpu(parser_args, model)
    # this model modify `flag` to represent the sparsity,
    # and `score` are all ones.
    old_epoch_list, old_val_acc_list, old_model_sparsity_list, old_test_acc_list = [], [], [], []

    finetune(parser_args, ntokens, model, criterion, train_data, val_data, test_data, old_epoch_list, old_val_acc_list, old_test_acc_list, old_model_sparsity_list)


def do_sanity_checks(parser_args, ntokens, model, criterion, train_data, val_data, test_data, epoch_list, val_acc_list, test_acc_list, model_sparsity_list):

    print("Beginning Sanity Checks:")
    # do the sanity check for shuffled mask/weights, reinit weights
    print("Sanity Check 1: Weight Reinit")
    cp_model = copy.deepcopy(model)
    cp_model = redraw(model, shuffle=False, reinit=True, invert=False, chg_mask=False, chg_weight=True)
    finetune(parser_args, ntokens, cp_model, criterion, train_data, val_data, test_data, epoch_list, val_acc_list, test_acc_list, model_sparsity_list, name_prefix='weight_reinit')

    print("Sanity Check 2: Mask Reshuffle")
    cp_model = copy.deepcopy(model)
    cp_model = redraw(model, shuffle=False, reinit=True, invert=False, chg_mask=True, chg_weight=False)
    finetune(parser_args, ntokens, cp_model, criterion, train_data, val_data, test_data, epoch_list, val_acc_list, test_acc_list, model_sparsity_list, name_prefix='mask_shuffle')


