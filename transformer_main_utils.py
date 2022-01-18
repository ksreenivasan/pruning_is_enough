# coding: utf-8
import argparse
import time
import math
import os
import re
import torch
import torch.nn as nn
import numpy as np
import torch.onnx

import transformer_data as data
import transformer_model as model

from utils.utils import set_seed
from utils.builder import get_builder
from args_helper import parser_args
from main_utils import switch_to_wt, set_gpu
from utils.net_utils import prune, get_prune_rate, round_model, get_regularization_loss


def batchify(data, bsz):
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
        if re.match('.*\.flag', name) or re.match('.*\.bias_flag', name):
            tensor = p.data.detach().cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, ({100 * nonzero / total:6.2f}% remained)')
    return (round((nonzero/total)*100, 1))


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


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
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


def train(parser_args, ntokens, train_data, model, optimizer, criterion):
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
        if args.regularization:
            regularization_loss = get_regularization_loss(model, regularizer=args.regularization,
                                        				  lmbda=args.lmbda, alpha=args.alpha,
                                        				  alpha_prime=args.alpha_prime)
        loss += regularization_loss
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), parser_args.transformer_clip)
        
        # for p in model.parameters():
        #     if p.requires_grad:
        #         p.data.add_(p.grad, alpha=-lr)
        optimizer.step()

        total_loss += loss.item()

        log_interval = 500
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // parser_args.transformer_bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()



def finetine(parser_args, old_epoch_list, old_test_acc_list, old_model_sparsity_list):

	model = switch_to_wt(model)
	lr = parser_args.fine_tune_lr
	parser_args.regularization = False

	epoch_list = copy.deepcopy(old_epoch_list)
    test_acc_list = copy.deepcopy(old_test_acc_list)
    model_sparsity_list = copy.deepcopy(old_model_sparsity_list)

    if parser_args.unflag_before_finetune:
        model = round_model(model, round_scheme="all_ones", noise=parser_args.noise, ratio=parser_args.noise_ratio, rank=parser_args.gpu)

    avg_sparsity = print_nonzeros(cp_model)

    for epoch in range(parser_args.epochs, parser_args.epochs * 2):
    	epoch_list.append(epoch)
        epoch_start_time = time.time()
        train(parser_args, ntokens, train_data, model, optimizer, criterion)
        
        val_loss = evaluate(val_data)
        val_acc_list.append(val_loss)
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
            lr /= 4.0


