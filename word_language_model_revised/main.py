# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

import pdb
import pandas as pd
import torch.nn.utils.prune as prune

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--algo', type=str, default='scratch',
                    help='type of algorithims (scratch, renda, retrain_renda)')
parser.add_argument('--prune_rate', type=float, default=0.2,
                    help='pruning rate for each round (for renda)')
parser.add_argument('--retrain_start_round', type=int, default=1,
                    help='starting round of retrain (renda)')
parser.add_argument('--rounds', type=int, default=1,
                    help='number of pruning rounds (for renda)')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--debug', action='store_true',
                    help='debug model')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

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
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

def set_parameters_to_prune():
    if args.model == 'LSTM' and args.tied:
        args.parameters_to_prune = (
            (model.encoder, 'weight'),
            (model.rnn, 'weight_ih_l0'),
            (model.rnn, 'weight_hh_l0'),
            (model.rnn, 'bias_ih_l0'),
            (model.rnn, 'bias_hh_l0'),
            (model.rnn, 'weight_ih_l1'),
            (model.rnn, 'weight_hh_l1'),
            (model.rnn, 'bias_ih_l1'),
            (model.rnn, 'bias_hh_l1'),
            (model.decoder, 'bias'),
        )
    else:
        raise NotImplementedError



ntokens = len(corpus.dictionary)
print('ntokens: ', ntokens)
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
set_parameters_to_prune()
criterion = nn.NLLLoss()
print(model)
for name, param in model.named_parameters():
    print(name, param.shape)

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
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    import pdb; pdb.set_trace()
    return data, target


def get_sparsity(verbose=True):
    print('='*30 + ' Getting sparsity ' + '='*30)
    if args.model == 'LSTM':
        numer, denom = 0, 0
        #pdb.set_trace()
        for name_m, module in model.named_modules():
            if name_m == '':
                continue
            print('module ', name_m)
            if list(module.named_buffers()) == []: # un-pruned case
                for name_p, param in module.named_parameters():
                    nonzero, nelement = int(torch.sum(param.data!=0)), param.nelement()
                    numer += nonzero
                    denom += nelement
                    if verbose:
                        print('{} sparsity: {}/{} ({:.2f}%)'.format(name_p, nonzero, nelement, 100*nonzero/nelement))
            else:
                for name_m, mask in module.named_buffers():
                    nonzero, nelement = int(torch.sum(mask.data!=0)), mask.nelement()
                    numer += nonzero
                    denom += nelement
                    if verbose:
                        print('{} sparsity: {}/{} ({:.2f}%)'.format(name_m, nonzero, nelement, 100*nonzero/nelement))




        # for name, param in model.named_parameters():
        #     if name.endswith('orig'): # when pruning is applied
        #         nonzero, nelement = int(torch.sum(param.data!=0)), param.nelement()
        #     else:
        #         nonzero, nelement = int(torch.sum(param.data!=0)), param.nelement()

        #     numer += nonzero
        #     denom += nelement
        #     if verbose:
        #         print('{} sparsity: {}/{} ({:.2f}%)'.format(name, nonzero, nelement, 100*nonzero/nelement))
        print('Overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))
    else:
        raise NotImplementedError

    return 100*numer/denom

def store_and_mag_prune(round=-1):

    # store mask & model with effective weights of current round
    #print(' Storing the state dict ')
    #print(model.state_dict().keys())
    ckpt_path = 'checkpoints/{}_{}_{}.pth'.format(args.algo, args.prune_rate, round)
    torch.save(model.state_dict(), ckpt_path)
    print('saved checkpoint in {}'.format(ckpt_path))

    # apply prune & get sparsity
    prune.global_unstructured(
                args.parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=args.prune_rate,
            )


def dummy_prune():
    # do not prune, but change the variables
    prune.global_unstructured(
            args.parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=0,
        )

def switch_to_wt():
    for (module, param_name) in args.parameters_to_prune:
        prune.remove(module, param_name)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break





def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

def save_result(epoch_list, val_ppl_list, val_loss_list, test_ppl_list, test_loss_list, sparsity_list, \
                epoch, val_loss, test_loss, sparsity, args):

    epoch_list.append(epoch)
    val_ppl_list.append(math.exp(val_loss))
    val_loss_list.append(val_loss)
    test_ppl_list.append(math.exp(test_loss))
    test_loss_list.append(test_loss)
    sparsity_list.append(sparsity)

    results_df = pd.DataFrame(
            {'epoch': epoch_list, 'val_ppl': val_ppl_list, 'val_loss': val_loss_list, 'test_ppl': test_ppl_list, 
            'test_loss': test_loss_list, 'sparsity': sparsity_list})
    if args.algo == 'retrain_renda':
        results_filename = 'results/' + 'ppl_sparsity_{}_pr{}_pre{}_rd{}_ep{}.csv'.format(args.algo, args.prune_rate, args.retrain_start_round, args.rounds, args.epochs)        
    else:
        results_filename = 'results/' + 'ppl_sparsity_{}_pr{}_rd{}_ep{}.csv'.format(args.algo, args.prune_rate, args.rounds, args.epochs)
    results_df.to_csv(results_filename, index=False)
    print('saved to: ', results_filename)

# load seed 
if args.algo == 'retrain_renda':
    
    #ckpt_path = 'checkpoints/test_ckpt_renda_4.pth'
    ckpt_path = 'checkpoints/renda_{}_{}_{}.pth'.format(args.algo, args.prune_rate, args.retrain_start_round)
    print("loading seed from {}".format(ckpt_path))
    dummy_prune()
    model.load_state_dict(torch.load(ckpt_path))
    switch_to_wt()

    # test the loaded model
    sparsity = get_sparsity()
    test_loss = evaluate(test_data)
    print('Loaded model: Sparsity {:.2f}%, Test ppl {:8.2f}'.format(sparsity, math.exp(test_loss)))

# Loop over rounds & epochs.
epoch_list, val_ppl_list, val_loss_list, test_ppl_list, test_loss_list, sparsity_list = [], [], [], [], [], []
if args.algo not in ['renda']:
    print('='*30 + ' We are doing single round ' + '='*30)
    args.rounds = 1

# check the model structure
# for name, param in model.named_parameters():
#     print(name, param.shape)
# print('='*30 + ' Printing the modules ' + '='*30)
# for name, module in model.named_modules():
#     print(name, module)


#get_sparsity()
for round in range(1, args.rounds+1):
    print("="*82 + '\n Round {}'.format(round) + '\n' + "="*82)
    #if args.debug:
    #    #exit() 
    #    store_and_mag_prune(round)

    best_val_loss = None
    lr = args.lr # lr setting (all), lr rewind (Renda)
    if args.prune_rate == 0.2:
        lr = lr / round
    for epoch in range(1, args.epochs+1):        
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)        
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            # with open(args.save, 'wb') as f:
            #     torch.save(model, f) 
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

        # test
        test_loss = evaluate(test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
        #print(epoch_list, val_ppl_list, val_loss_list, test_ppl_list, test_loss_list)
        #print(epoch, val_loss, test_loss)

        # get sparsity
        sparsity = get_sparsity(verbose=False)

        # save the results into csv
        save_result(epoch_list, val_ppl_list, val_loss_list, test_ppl_list, test_loss_list, sparsity_list, \
                    epoch, val_loss, test_loss, sparsity, args)

    # do magnitude pruning
    if args.algo in ['renda']:
        sparsity = get_sparsity()
        store_and_mag_prune(round)   
    
      

# Load the best saved model.
# with open(args.save, 'rb') as f:
#     model = torch.load(f)
#     # after load the rnn params are not a continuous chunk of memory
#     # this makes them a continuous chunk, and will speed up forward pass
#     # Currently, only rnn model supports flatten_parameters function.
#     if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
#         model.rnn.flatten_parameters()
#
# Run on test data.
# test_loss = evaluate(test_data)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)
# save_result(epoch_list, val_ppl_list, val_loss_list, test_ppl_list, test_loss_list, sparsity_list, \
#             epoch, best_val_loss, test_loss, sparsity, args)


if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)







# def old_prune_weights(round=-1):
#     print("This is no longer used")
#     raise NotImplementedError

#     #print("prune the model using magnitude of the weights")
#     if args.model == 'LSTM':
#         magnitude_list = []
#         nelement_list = []
#         shape_list = []
#         # measure the magnitude of each weight/bias
#         for name, param in model.named_parameters():
#             magnitude_list.append(param.data.flatten().abs())
#             nelement_list.append(param.nelement())
#             shape_list.append(param.shape)
#             print(name, param.nelement())
#         # sort the magnitudes & get threshold
#         mag = torch.cat(magnitude_list)
#         total_nelement = len(mag)
#         sorted_mag, indices_mag = torch.sort(mag) # ascending order
#         threshold = sorted_mag[int(total_nelement*args.prune_rate)]
#         print('threshold: ', threshold)

#         # define mask (which neuron will be pruned)
#         mask_list = []
#         for name, param in model.named_parameters():
#             mask_list.append((param.data.abs() > threshold).int())    
#         #get_sparsity(verbose=True) # get sparsity before pruning
            
#         # check the sparsity of each mask
#         # print('Checking mask sparsity')
#         # numer, denom = 0, 0
#         # for mask in mask_list:
#         #     nonzero, nelement = len(mask.nonzero()), mask.nelement()
#         #     numer += nonzero
#         #     denom += nelement
#         #     print('{}/{} : ({:.2f} %)'.format(nonzero, nelement, 100*nonzero/nelement))
#         # print('Overall sparsity: {}/{} ({:.2f} %)'.format((int)(numer), denom, 100*numer/denom))

#         # print('Check named_buffers before pruning')
#         # for module in model.modules():
#         #     print(list(module.named_buffers()))

#         # apply pruning
#         if not args.tied:
#             raise NotImplementedError
#         for name, module in model.named_modules():
#             if name in ['', 'drop']: # whole module
#                 pass
#             elif name == 'encoder':
#                 prune.custom_from_mask(module, name="weight", mask=mask_list[0])
#             elif name == 'rnn':
#                 prune.custom_from_mask(module, name="weight_ih_l0", mask=mask_list[1])
#                 prune.custom_from_mask(module, name="weight_hh_l0", mask=mask_list[2])
#                 prune.custom_from_mask(module, name="bias_ih_l0", mask=mask_list[3])
#                 prune.custom_from_mask(module, name="bias_hh_l0", mask=mask_list[4])
#                 prune.custom_from_mask(module, name="weight_ih_l1", mask=mask_list[5])
#                 prune.custom_from_mask(module, name="weight_hh_l1", mask=mask_list[6])
#                 prune.custom_from_mask(module, name="bias_ih_l1", mask=mask_list[7])
#                 prune.custom_from_mask(module, name="bias_hh_l1", mask=mask_list[8])
#             elif name == 'decoder':
#                 prune.custom_from_mask(module, name="bias", mask=mask_list[9])
#             else:
#                 raise ValueError
        
#         # check the sparsity after pruning
#         get_sparsity(verbose=True)
#         # print('Check named_buffers after pruning')
#         # for module in model.modules():
#         #     print(list(module.named_buffers()))

#         # store mask & model with effective weights of current round
#         print(' Storing the state dict ')
#         #print(model.state_dict().keys())
#         torch.save(model.state_dict(), 'checkpoints/{}_{}.pth'.format(args.algo, round))

#     else:
#         raise NotImplementedError
   