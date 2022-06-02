import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np

import math

from args_helper import parser_args

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k, scores_prune_threshold=-np.inf):
        if parser_args.algo == 'ep':
            # Get the supermask by sorting the scores and using the top k%
            out = scores.clone()
            _, idx = scores.flatten().sort()
            j = int((1 - k) * scores.numel())
            # flat_out and out access the same memory.
            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1

        elif parser_args.algo in ['hc', 'hc_iter']:
            # round scores to {0, 1}
            # NOTE: doing this EP style where the scores are unchanged, but mask is computed
            # can also try a variant where we actually round the scores
            if parser_args.bottom_k_on_forward:
                out = torch.gt(scores, torch.ones_like(scores)*scores_prune_threshold).float()
            else:
                out = torch.gt(scores, torch.ones_like(scores)*parser_args.quantize_threshold).float()

        else:
            raise NotImplementedError

        return out




class SubnetEmbedding(nn.Embedding):
    def __init__(self, ntoken, ninp):
        super().__init__(ntoken, ninp) # self.weight is automatically made with size 33278 x 1500

        # scores
        self.scores = Parameter(torch.Tensor(self.weight.size()))
        if parser_args.algo in ['hc_iter']:
            if parser_args.random_subnet:
                self.scores.data = torch.bernoulli(parser_args.prune_rate * torch.ones_like(self.scores.data))
            elif parser_args.score_init in ['unif']:
                nn.init.uniform_(self.scores, a=0.0, b=1.0)    
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # flag
        self.flag = Parameter(torch.ones(self.weight.size()))

        # True by default: turn the gradient on the weights off
        if parser_args.freeze_weights:
            self.weight.requires_grad = False
            self.flag.requires_grad = False

    def forward(self, x):
        if parser_args.algo in ['hc_iter']:
            # don't need a mask here. the scores are directly multiplied with weights
            if parser_args.differentiate_clamp:
                self.scores.data = torch.clamp(self.scores.data, 0.0, 1.0)
                
            if parser_args.hc_quantized:
                subnet = GetSubnet.apply(self.scores, parser_args.prune_rate)
                subnet = subnet * self.flag.data.float()                
            else:
                subnet = self.scores * self.flag.data.float()
                
        elif parser_args.algo in ['imp']: # no STE, no subnet. Mask is handled outside
            pass
        elif parser_args.algo in ['ep']:
            subnet = GetSubnet.apply(self.scores.abs(), parser_args.prune_rate)
        else:
            raise NotImplementedError
            
        if parser_args.algo in ['imp']: # no STE, no subnet. Mask is handled outside
            w = self.weight
        else:
            w = self.weight * subnet

        return F.embedding(
            x, w, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
