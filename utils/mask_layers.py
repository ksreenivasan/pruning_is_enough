import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = args[0]          # TODO: get rid of this and use fc.in_features instead!
        self.out_channels = args[1]
        self.mask_weight = nn.Parameter(torch.ones(self.weight.size()))
        
        if kwargs['bias']:
            self.mask_bias = nn.Parameter(torch.ones(self.bias.size()))

        # create a list of boolean flags to indicate whether each activation has been pruned. This is used when
        # args.pruning_strategy is set to "activations_and_weights." Once activations are pruned, we do not went to check
        # whether or not to prune a weight that belongs to that activation, so we check this list first before checking.
        self.pruned_activation = [False] * len(self.weight)

    def update_mask_weight(self, out_idx, in_idx, value):
        self.mask_weight[out_idx, in_idx] = value

    def update_mask_bias(self, out_idx, value):
        self.mask_bias[out_idx] = value

    def forward(self, x):
        w = self.weight * self.mask_weight
        b = self.bias * self.mask_bias
        return F.linear(x, w, b)


class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = args[0]
        self.out_channels = args[1]
        self.mask_weight = nn.Parameter(torch.ones(self.weight.size()))

        if kwargs['bias']:
            self.mask_bias = nn.Parameter(torch.ones(self.bias.size()))

        # create a list of boolean flags to indicate whether each activation has been pruned. This is used when
        # args.pruning_strategy is set to "activations_and_weights." Once activations are pruned, we do not went to check
        # whether or not to prune a weight that belongs to that activation, so we check this list first before checking.
        self.pruned_activation = [False] * len(self.weight)

    def forward(self, x):
        w = self.weight * self.mask_weight
        b = self.bias * self.mask_bias
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
