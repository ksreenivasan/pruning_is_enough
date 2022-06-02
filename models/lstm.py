import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from enum import IntEnum
from typing import Optional, Tuple

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class NaiveLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, layer_index=0):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        # input gate
        self.W_ii = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hi = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = Parameter(torch.Tensor(hidden_sz))
        # forget gate
        self.W_if = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hf = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = Parameter(torch.Tensor(hidden_sz))
        # ???
        self.W_ig = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_hg = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_g = Parameter(torch.Tensor(hidden_sz))
        # output gate
        self.W_io = Parameter(torch.Tensor(input_sz, hidden_sz))
        self.W_ho = Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = Parameter(torch.Tensor(hidden_sz))

        print("##### Sizes: W_ii {}, W_hi: {}, W_if: {}, W_hf: {}, W_ig: {}, W_hg: {}, W_io: {}, W_ho: {}".format(
                    self.W_ii.size(), self.W_hi.size(), self.W_if.size(), self.W_hf.size(), 
                    self.W_ig.size(), self.W_hg.size(), self.W_io.size(), self.W_ho.size()))

        self._layer_index = layer_index # for stacked LSTM
        
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        
    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        seq_sz, bs, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[t, :, :]

            i_t = torch.nn.functional.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t = torch.nn.functional.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            g_t = torch.nn.functional.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            o_t = torch.nn.functional.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.nn.functional.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        #hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()

        return hidden_seq, (h_t, c_t)

class StackedLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers

        # multi-layer LSTM
        if num_layers == 2:
            self.lstm1 = NaiveLSTM(input_sz=input_sz, hidden_sz=hidden_sz)#, layer_index=0)
            self.dropout = nn.Dropout(p=dropout)
            self.lstm2 = NaiveLSTM(input_sz=hidden_sz, hidden_sz=hidden_sz)#, layer_index=1)
        elif num_layers == 1:
            self.lstm1 = NaiveLSTM(input_sz=input_sz, hidden_sz=hidden_sz)#, layer_index=0)
        
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


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
            self.rnn = StackedLSTM(ninp, nhid, nlayers, dropout=dropout) # #
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken) #

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


