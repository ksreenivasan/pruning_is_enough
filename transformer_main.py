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
from main_utils import switch_to_wt
from utils.net_utils import prune, get_prune_rate, round_model

# parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
if parser_args.subfolder is not None:
        if not os.path.isdir('results/'):
            os.mkdir('results/')
        result_subroot = 'results/' + parser_args.subfolder
        if not os.path.isdir(result_subroot):
            os.mkdir(result_subroot)

# Set the random seed manually for reproducibility.
set_seed(parser_args.seed * parser_args.trial_num)
device = torch.device("cuda:{}".format(parser_args.gpu))

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(parser_args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, parser_args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model.TransformerModel(get_builder(), ntokens, parser_args.transformer_emsize, parser_args.transformer_nhead, parser_args.transformer_nhid, parser_args.transformer_nlayers, parser_args.transformer_dropout).to(device)
model = switch_to_wt(model).to(device)
print(model)
criterion = nn.NLLLoss()

if not parser_args.override_prune_rate:
    parser_args.prune_rate = get_prune_rate(parser_args.target_sparsity, parser_args.iter_period)
    print("Setting prune_rate to {}".format(parser_args.prune_rate))
else:
    print("Overriding prune_rate to {}".format(parser_args.prune_rate))


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

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

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


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, parser_args.transformer_bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(data)
        output = output.view(-1, ntokens)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), parser_args.transformer_clip)
        
        for p in model.parameters():
            if p.requires_grad:
                p.data.add_(p.grad, alpha=-lr)
                # print(p.grad.norm())

        total_loss += loss.item()

        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // parser_args.transformer_bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        # if args.dry_run:
        #     break


def export_onnx(batch_size, seq_len):
    onnx_path = os.path.join("results", parser_args.subfolder, "onnx_output.onnx")
    print('The model is also exported in ONNX format at {}'.format(onnx_path))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), onnx_path)


# Loop over epochs.
lr = parser_args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, parser_args.epochs + 1):
        epoch_start_time = time.time()
        train()
        if not parser_args.weight_training and parser_args.algo in ['hc_iter', 'global_ep_iter'] and epoch % (parser_args.iter_period) == 0 and epoch != 1:
            prune(model)
            cp_model = round_model(model, parser_args.round, noise=parser_args.noise,
                                   ratio=parser_args.noise_ratio, rank=parser_args.gpu)
            # avg_sparsity = get_model_sparsity(cp_model)
            avg_sparsity = print_nonzeros(cp_model)
            print('Model avg sparsity: {}'.format(avg_sparsity))

        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join("results", parser_args.subfolder, "model.pt"), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(os.path.join("results", parser_args.subfolder, "model.pt"), 'rb') as f:
    model = torch.load(f)


# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


# Export the model in ONNX format.
# export_onnx(batch_size=1, seq_len=parser_args.transformer_bptt)
