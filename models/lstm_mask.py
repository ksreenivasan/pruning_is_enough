import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from enum import IntEnum
from typing import Optional, Tuple

from args_helper import parser_args
import pdb

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class NaiveLSTM_Mask(nn.Module):
    def __init__(self, input_sz, hidden_sz, layer_index=0):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.define_variables()

        print("##### Sizes: W_ii {}, W_hi: {}, W_if: {}, W_hf: {}, W_ig: {}, W_hg: {}, W_io: {}, W_ho: {}".format(
                    self.W_ii.size(), self.W_hi.size(), self.W_if.size(), self.W_hf.size(), 
                    self.W_ig.size(), self.W_hg.size(), self.W_io.size(), self.W_ho.size()))

        self._layer_index = layer_index # for stacked LSTM
        
        self.init_weights()
        self.freeze_weights()
        # set a storage for layer score 
        if parser_args.rewind_score:
            self.saved_scores = None       
            
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    def define_variables(self):
        input_sz = self.input_size
        hidden_sz = self.hidden_size
        # input gate 
        ## weight
        self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hi = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = Parameter(torch.Tensor(hidden_sz))
        ## score
        self.W_ii_scores = Parameter(torch.Tensor(self.W_ii.size()))
        self.W_hi_scores = Parameter(torch.Tensor(self.W_hi.size()))
        self.b_i_scores = Parameter(torch.Tensor(self.b_i.size()))
        ## flag
        self.W_ii_flag = Parameter(torch.ones(self.W_ii.size()))
        self.W_hi_flag = Parameter(torch.ones(self.W_hi.size()))
        self.b_i_flag = Parameter(torch.ones(self.b_i.size()))

        # forget gate
        ## weight
        self.W_if = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hf = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = Parameter(torch.Tensor(hidden_sz))
        ## score
        self.W_if_scores = Parameter(torch.Tensor(self.W_if.size()))
        self.W_hf_scores = Parameter(torch.Tensor(self.W_hf.size()))
        self.b_f_scores = Parameter(torch.Tensor(self.b_f.size()))
        ## flag
        self.W_if_flag = Parameter(torch.ones(self.W_if.size()))
        self.W_hf_flag = Parameter(torch.ones(self.W_hf.size()))
        self.b_f_flag = Parameter(torch.ones(self.b_f.size()))

        # input modulation gate
        ## weight
        self.W_ig = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hg = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_g = Parameter(torch.Tensor(hidden_sz))
        ## score
        self.W_ig_scores = Parameter(torch.Tensor(self.W_if.size()))
        self.W_hg_scores = Parameter(torch.Tensor(self.W_hf.size()))
        self.b_g_scores = Parameter(torch.Tensor(self.b_f.size()))
        ## flag
        self.W_ig_flag = Parameter(torch.ones(self.W_ig.size()))
        self.W_hg_flag = Parameter(torch.ones(self.W_hg.size()))
        self.b_g_flag = Parameter(torch.ones(self.b_g.size()))

        # output gate
        ## weight
        self.W_io = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ho = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = Parameter(torch.Tensor(hidden_sz))
        ## score
        self.W_io_scores = Parameter(torch.Tensor(self.W_io.size()))
        self.W_ho_scores = Parameter(torch.Tensor(self.W_ho.size()))
        self.b_o_scores = Parameter(torch.Tensor(self.b_o.size()))
        ## flag
        self.W_io_flag = Parameter(torch.ones(self.W_io.size()))
        self.W_ho_flag = Parameter(torch.ones(self.W_ho.size()))
        self.b_o_flag = Parameter(torch.ones(self.b_o.size()))

    def init_weights(self):
        for name, p in self.named_parameters():
            #print('LSTM init: ', name, p.size())
            if name.endswith('scores'): # scores
                if parser_args.algo in ['hc_iter']:
                    if parser_args.random_subnet:
                        p.data = torch.bernoulli(parser_args.prune_rate * torch.ones_like(p.data))
                    elif parser_args.score_init in ['unif']:
                        nn.init.uniform_(p.data, a=0.0, b=1.0)
                    else:
                        raise NotImplementedError
                else:
                    if p.data.ndimension() >= 2:
                        nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
                    else:
                        nn.init.uniform_(p.data, a=-1.0, b=1.0)
            elif name.endswith('flag'):  # flag
                continue
            else: # weights                
                if p.data.ndimension() >= 2:
                    nn.init.xavier_uniform_(p.data)
                else:
                    nn.init.zeros_(p.data)
        
    def freeze_weights(self):
        if parser_args.freeze_weights:
            for name, p in self.named_parameters():
                if name.endswith('scores'): # scores
                    continue
                elif name.endswith('flag'):  # flag
                    p.requires_grad = False
                else:
                    p.requires_grad = False

    def get_subnet_list(self):

        for name, p in self.named_parameters():
            if name.endswith('ii_scores'):
                if parser_args.algo in ['hc_iter']
                    if parser_args.hc_quantized:
                        subnet = GetSubnet.apply(p.abs())
                        subnet_dict['ii'] = subnet * name.split('_')[:-2] + 
                    else:


            elif name.endswith('hi_scores'):                
                #TODO: add here...
                pdb.set_trace()
            else:
                continue    
            print(name)

        pdb.set_trace()
        raise NotImplementedError


        return subnet_dict

    def clamp_scores(self):
        for name, p in self.named_parameters():
            if name.endswith('scores'):
                p.data = torch.clamp(p.data, 0.0, 1.0)


    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        seq_sz, bs, _ = x.size()
        hidden_seq = []

        if parser_args.differentiate_clamp and parser_args.algo in ['hc_iter']:
            self.clamp_scores()

        if parser_args.algo in ['imp']:
            W_ii, W_hi, b_i = self.W_ii, self.W_hi, self.b_i
            W_if, W_hf, b_f = self.W_if, self.W_hf, self.b_f
            W_ig, W_hg, b_g = self.W_ig, self.W_hg, self.b_g
            W_io, W_ho, b_o = self.W_io, self.W_ho, self.b_o
        else:
            subnet_dict = self.get_subnet_list()
            W_ii, W_hi, b_i = self.W_ii * subnet_dict['ii'], self.W_hi * subnet_dict['hi'], self.b_i * subnet_dict['bi']
            W_if, W_hf, b_f = self.W_if * subnet_dict['if'], self.W_hf * subnet_dict['hf'], self.b_i * subnet_dict['bf']
            W_ig, W_hg, b_g = self.W_ig * subnet_dict['ig'], self.W_hg * subnet_dict['hg'], self.b_i * subnet_dict['bg']
            W_io, W_ho, b_o = self.W_io * subnet_dict['io'], self.W_ho * subnet_dict['ho'], self.b_i * subnet_dict['bo']

        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[t, :, :]

            i_t = torch.nn.functional.sigmoid(x_t @ W_ii + h_t @ W_hi + b_i)
            f_t = torch.nn.functional.sigmoid(x_t @ W_if + h_t @ W_hf + b_f)
            g_t = torch.nn.functional.tanh(x_t @ W_ig + h_t @ W_hg + b_g)
            o_t = torch.nn.functional.sigmoid(x_t @ W_io + h_t @ W_ho + b_o)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.nn.functional.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        #hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()

        return hidden_seq, (h_t, c_t)

    # def init_scores(self):
    #     scores_list = [self.W_ii_scores, self.W_hi_scores, self.b_i_scores,
    #                     self.W_if_scores, self.W_hf_scores, self.b_f_scores,
    #                     self.W_ig_scores, self.W_hg_scores, self.b_g_scores,
    #                     self.W_io_scores, self.W_ho_scores, self.b_o_scores]

    #     if parser_args.algo in ['hc_iter']:
    #         for scores in scores_list:
    #             if parser_args.random_subnet:
    #                 scores.data = torch.bernoulli(parser_args.prune_rate * torch.ones_like(scores.data))
    #             elif parser_args.score_init in ['unif']:
    #                 nn.init.uniform_(scores, a=0.0, b=1.0)
    #             else:
    #                 raise NotImplementedError
    #     else:
    #         for scores in scores_list:
    #             nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    #             #nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0) # can't do kaiming here. picking U[-1, 1] for no real reason

class StackedLSTM_Mask(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers

        # multi-layer LSTM
        if num_layers == 2:
            self.lstm1 = NaiveLSTM_Mask(input_sz=input_sz, hidden_sz=hidden_sz)#, layer_index=0)
            self.dropout = nn.Dropout(p=dropout)
            self.lstm2 = NaiveLSTM_Mask(input_sz=hidden_sz, hidden_sz=hidden_sz)#, layer_index=1)
        elif num_layers == 1:
            self.lstm1 = NaiveLSTM_Mask(input_sz=input_sz, hidden_sz=hidden_sz)#, layer_index=0)
        
    def forward(self, x, init_states):
        # """Assumes x is of shape (batch, sequence, feature)"""
        if self.num_layers == 2:
            output, hidden = self.lstm1(x=x, 
                                init_states=init_states)
            output = self.dropout(output)
            output, hidden = self.lstm2(x=output)
        else:
            output, hidden = self.lstm1(x=x, init_states=init_states)
        return output, hidden

class Embedding_Mask(nn.Embedding):
    def __init__(self, ntoken, ninp):
        super().__init__(ntoken, ninp) # self.weight is automatically made with size 33278 x 1500
        self.ntoken = ntoken
        self.ninp = ninp

        #pdb.set_trace() 

    def forward(self, x):
        output = None # get the result of lookup table? (but masked lookup table?)
        raise NotImplementedError

        return output



class Linear_Mask(nn.Linear):
    def __init__(self, nhid, ntoken):
        super().__init__(nhid, ntoken)
        self.nhid = nhid
        self.ntoken = ntoken

        pdb.set_trace()

        # TODO: add definition of weight/bias, following nn.Linear
        self.weight = None
        self.bias = None

        # TODO: add mask/flag for weight/bias

    def forward(self, x):
        output = None #self.weight * x
        raise NotImplementedError

        return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = Embedding_Mask(ntoken, ninp)  #nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = StackedLSTM_Mask(ninp, nhid, nlayers, dropout=dropout) # #
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = Linear_Mask(nhid, ntoken) #nn.Linear(nhid, ntoken) #

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        # weight = next(self.parameters())
        # if self.rnn_type == 'LSTM':
        #     return (weight.new_zeros(self.nlayers, bsz, self.nhid),
        #             weight.new_zeros(self.nlayers, bsz, self.nhid))
        # else:
        #     return weight.new_zeros(self.nlayers, bsz, self.nhid)
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(bsz, self.nhid),
                    weight.new_zeros(bsz, self.nhid))
        else:
            return weight.new_zeros(bsz, self.nhid)





# class OptimizedLSTM(nn.Module):
#     def __init__(self, input_sz: int, hidden_sz: int):
#         super().__init__()
#         self.input_sz = input_sz
#         self.hidden_size = hidden_sz
#         self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
#         self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
#         self.bias = Parameter(torch.Tensor(hidden_sz * 4))
#         self.init_weights()
    
#     def init_weights(self):
#         for p in self.parameters():
#             if p.data.ndimension() >= 2:
#                 nn.init.xavier_uniform_(p.data)
#             else:
#                 nn.init.zeros_(p.data)
        
#     def forward(self, x: torch.Tensor, 
#                 init_states: Optional[Tuple[torch.Tensor]]=None
#                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         """Assumes x is of shape (batch, sequence, feature)"""
#         seq_sz, bs, _ = x.size()
#         hidden_seq = []
#         if init_states is None:
#             h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
#                         torch.zeros(self.hidden_size).to(x.device))
#         else:
#             h_t, c_t = init_states
        
#         HS = self.hidden_size
#         for t in range(seq_sz):
#             x_t = x[t, :, :]
#             #import pdb; pdb.set_trace()
#             # batch the computations into a single matrix multiplication
#             gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
#             i_t, f_t, g_t, o_t = (
#                 torch.sigmoid(gates[:, :HS]), # input
#                 torch.sigmoid(gates[:, HS:HS*2]), # forget
#                 torch.tanh(gates[:, HS*2:HS*3]),
#                 torch.sigmoid(gates[:, HS*3:]), # output
#             )
#             c_t = f_t * c_t + i_t * g_t
#             h_t = o_t * torch.tanh(c_t)
#             hidden_seq.append(h_t.unsqueeze(Dim.batch))
#         hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
#         # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
#         #hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
#         return hidden_seq, (h_t, c_t)


# class RNNModel_Orig(nn.Module):
#     """Container module with an encoder, a recurrent module, and a decoder."""

#     def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
#         super(RNNModel_orig, self).__init__()
#         self.ntoken = ntoken
#         self.drop = nn.Dropout(dropout)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         if rnn_type in ['LSTM', 'GRU']:
#             #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
#             self.rnn = StackedLSTM(ninp, nhid, nlayers, dropout=dropout)
#         else:
#             try:
#                 nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
#             except KeyError:
#                 raise ValueError( """An invalid option for `--model` was supplied,
#                                  options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
#             self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
#         self.decoder = nn.Linear(nhid, ntoken)

#         # Optionally tie weights as in:
#         # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
#         # https://arxiv.org/abs/1608.05859
#         # and
#         # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
#         # https://arxiv.org/abs/1611.01462
#         if tie_weights:
#             if nhid != ninp:
#                 raise ValueError('When using the tied flag, nhid must be equal to emsize')
#             self.decoder.weight = self.encoder.weight

#         self.init_weights()

#         self.rnn_type = rnn_type
#         self.nhid = nhid
#         self.nlayers = nlayers

#     def init_weights(self):
#         initrange = 0.1
#         nn.init.uniform_(self.encoder.weight, -initrange, initrange)
#         nn.init.zeros_(self.decoder.weight)
#         nn.init.uniform_(self.decoder.weight, -initrange, initrange)

#     def forward(self, input, hidden):
#         emb = self.drop(self.encoder(input))
#         output, hidden = self.rnn(emb, hidden)
#         output = self.drop(output)
#         decoded = self.decoder(output)
#         decoded = decoded.view(-1, self.ntoken)
#         return F.log_softmax(decoded, dim=1), hidden

#     def init_hidden(self, bsz):
#         # weight = next(self.parameters())
#         # if self.rnn_type == 'LSTM':
#         #     return (weight.new_zeros(self.nlayers, bsz, self.nhid),
#         #             weight.new_zeros(self.nlayers, bsz, self.nhid))
#         # else:
#         #     return weight.new_zeros(self.nlayers, bsz, self.nhid)
#         weight = next(self.parameters())
#         if self.rnn_type == 'LSTM':
#             return (weight.new_zeros(bsz, self.nhid),
#                     weight.new_zeros(bsz, self.nhid))
#         else:
#             return weight.new_zeros(bsz, self.nhid)

