import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_model_utils import *
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
# from model_utils_torch import TransformerEncoderLayer
from args_helper import parser_args


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, builder, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        # try:
        #     from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # except:
        #     raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        # encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        encoder_layers = Block(builder, dim=ninp, num_heads=nhead, mlp_hidden_dim=nhid, drop=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.transformer_encoder = nn.Sequential(*[encoder_layers for i in range(nlayers)])
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # self.decoder = builder.conv1x1(ninp, ntoken)
        # self.decoder = nn.Linear(ninp, ntoken)
        self.decoder = builder.linear(ninp, ntoken)

        self.init_weights()

        self.prunable_layer_names, self.prunable_biases = self.get_prunable_param_names()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def get_prunable_param_names(model):
        prunable_weights = [name + '.weight' for name, module in model.named_modules() if
                (isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)) and
                'decoder' not in name]
        if parser_args.bias:
            prunable_biases = [name + '.bias' for name, module in model.named_modules() if
                (isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)) and
                'decoder' not in name]
        else:
            prunable_biases = [""]

        return prunable_weights, prunable_biases

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=self.src_mask)
        N, C, H = output.shape
        # output = output.view(N * C, H, 1, 1)  # .contiguous()
        output = self.decoder(output)
        # output = output.view(N, C, -1)  # .contiguous()
        return F.log_softmax(output, dim=-1)
