import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math

from args_helper import parser_args


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, bias_scores, k, scores_prune_threshold=-np.inf, bias_scores_prune_threshold=-np.inf):
        if parser_args.algo == 'pt_hack':
            # Get the supermask by normalizing scores and "sampling" by probability
            if parser_args.normalize_scores:
                # min-max normalization so that scores are in [0, 1]
                min_score = scores.min().item()
                max_score = scores.max().item()
                scores = (scores - min_score)/(max_score - min_score)

                # repeat for bias
                min_score = bias_scores.min().item()
                max_score = bias_scores.max().item()
                bias_scores = (bias_scores - min_score)/(max_score - min_score)

            # sample using scores as probability
            # by default the probabilities are too small. artificially
            # pushing them towards 1 helps!
            MULTIPLIER = 10
            scores = torch.clamp(MULTIPLIER*scores, 0, 1)
            bias_scores = torch.clamp(MULTIPLIER*bias_scores, 0, 1)
            out = torch.bernoulli(scores)
            bias_out = torch.bernoulli(bias_scores)

        elif parser_args.algo == 'ep' or parser_args.algo == 'ep+greedy':
            # Get the supermask by sorting the scores and using the top k%
            out = scores.clone()
            _, idx = scores.flatten().sort()
            j = int((1 - k) * scores.numel())
            # flat_out and out access the same memory.
            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1

            # repeat for bias
            # Get the supermask by sorting the scores and using the top k%
            bias_out = bias_scores.clone()
            _, idx = bias_scores.flatten().sort()
            j = int((1 - k) * bias_scores.numel())

            # flat_out and out access the same memory.
            bias_flat_out = bias_out.flatten()
            bias_flat_out[idx[:j]] = 0
            bias_flat_out[idx[j:]] = 1

        elif parser_args.algo in ['global_ep', 'global_ep_iter']:
            # define out, bias_out based on the layer's prune_threshold, bias_threshold
            out = torch.gt(scores, torch.ones_like(scores)*scores_prune_threshold).float()
            bias_out = torch.gt(bias_scores, torch.ones_like(bias_scores)*bias_scores_prune_threshold).float()

        elif parser_args.algo == 'pt':
            scores = torch.clamp(MULTIPLIER*scores, 0, 1)
            bias_scores = torch.clamp(MULTIPLIER*bias_scores, 0, 1)
            out = torch.bernoulli(scores)
            bias_out = torch.bernoulli(bias_scores)

        elif parser_args.algo in ['hc', 'hc_iter']:
            # round scores to {0, 1}
            # NOTE: doing this EP style where the scores are unchanged, but mask is computed
            # can also try a variant where we actually round the scores
            if parser_args.bottom_k_on_forward:
                out = torch.gt(scores, torch.ones_like(scores)*scores_prune_threshold).float()
                bias_out = torch.gt(bias_scores, torch.ones_like(bias_scores)*bias_scores_prune_threshold).float()
            else:
                out = torch.gt(scores, torch.ones_like(scores)*parser_args.quantize_threshold).float()
                bias_out = torch.gt(bias_scores, torch.ones_like(bias_scores)*parser_args.quantize_threshold).float()

        else:
            print("Pruning Algo = {}".format(parser_args.algo))
            print("Are you sure you should be here?")
            print("GetSubnet")

        return out, bias_out

    @staticmethod
    def backward(ctx, g_1, g_2):
        # send the gradient g straight-through on the backward pass.
        return g_1, g_2, None, None, None


# Not learning weights, finding subnet
class SubnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize flag (representing the pruned weights)
        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        if self.bias is not None:
            self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_flag = nn.Parameter(torch.Tensor(1))

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if self.bias is not None:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_scores = nn.Parameter(torch.Tensor(1))
        
        # prune scores below this for global EP in bottom-k
        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf
        
        if parser_args.algo in ['hc', 'hc_iter']:
            if parser_args.random_subnet:
                self.scores.data = torch.bernoulli(parser_args.prune_rate * torch.ones_like(self.scores.data))
                self.bias_scores.data = torch.bernoulli(parser_args.prune_rate * torch.ones_like(self.bias_scores.data))
            elif parser_args.score_init in ['half']:
                self.scores.data = 0.5 * torch.ones_like(self.scores.data)
                self.bias_scores.data = 0.5 * torch.ones_like(self.bias_scores.data)
            elif parser_args.score_init in ['bern']:
                self.scores.data = torch.bernoulli(0.5 * torch.ones_like(self.scores.data))                                                                                                                     
                self.bias_scores.data = torch.bernoulli(0.5 * torch.ones_like(self.bias_scores.data))    
            elif parser_args.score_init in ['unif']:
                nn.init.uniform_(self.scores, a=0.0, b=1.0)
                nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
            elif parser_args.score_init in ['bimodal', 'skew']:
                Beta = torch.distributions.beta.Beta
                if parser_args.score_init == 'bimodal':
                    alpha, beta = 0.1, 0.1
                elif parser_args.score_init == 'skew':
                    alpha, beta = 1, 5
                m = Beta(torch.ones_like(self.scores.data)*alpha, torch.ones_like(self.scores.data)*beta)
                self.scores.data = m.sample()
                m = Beta(torch.ones_like(self.bias_scores.data)*alpha, torch.ones_like(self.bias_scores.data)*beta)
                self.bias_scores.data = m.sample()
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0) # can't do kaiming here. picking U[-1, 1] for no real reason

        # True by default
        if parser_args.freeze_weights:
            # NOTE: turn the gradient on the weights off
            self.weight.requires_grad = False
            self.flag.requires_grad = False
            self.bias_flag.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        # set a storage for layer score 
        if parser_args.rewind_score:
            self.saved_scores = None

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        if parser_args.algo in ['hc', 'hc_iter']:
            # don't need a mask here. the scores are directly multiplied with weights
            if parser_args.differentiate_clamp:
                self.scores.data = torch.clamp(self.scores.data, 0.0, 1.0)
                self.bias_scores.data = torch.clamp(self.bias_scores.data, 0.0, 1.0)

            if parser_args.hc_quantized:
                subnet, bias_subnet = GetSubnet.apply(self.scores, self.bias_scores, parser_args.prune_rate)
                subnet = subnet * self.flag.data.float()
                bias_subnet = subnet * self.bias_flag.data.float()
            else:
                subnet = self.scores * self.flag.data.float()
                bias_subnet = self.bias_scores * self.bias_flag.data.float()
        elif parser_args.algo in ['imp']:
            # no STE, no subnet. Mask is handled outside
            pass
        elif parser_args.algo in ['global_ep', 'global_ep_iter']:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0, self.scores_prune_threshold, self.bias_scores_prune_threshold)
        else:
            # ep, global_ep, global_ep_iter, pt etc
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), parser_args.prune_rate)
        
        if parser_args.algo in ['imp']:
            # no STE, no subnet. Mask is handled outside
            w = self.weight
            b = self.bias
        else:
            w = self.weight * subnet
            if self.bias is not None:
                b = self.bias * bias_subnet
            else:
                b = self.bias

        x = F.linear(x, w, b)
        return x
