import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_weight = nn.Parameter(torch.ones(self.weight.size()))
        
        if kwargs['bias']:
            self.mask_bias = nn.Parameter(torch.ones(self.bias.size()))
            self.fixed_bias = nn.Parameter(torch.ones(self.bias.size()))

        # create a list of boolean flags to indicate whether each activation has been pruned. This is used when
        # args.pruning_strategy is set to "activations_and_weights." Once activations are pruned, we do not went to check
        # whether or not to prune a weight that belongs to that activation, so we check this list first before checking.
        self.pruned_activation = [False] * len(self.weight)    # TODO: delete this! make sure that you replace all instances of it with the fixed matrix below

        # create a second mask weight that captures permanent changes to the pruned network. This matrix is relevant 
        # when the network is pruned by activation first and then weights, or when EP is used, followed by greedy pruning.
        # this fixed mask is checked before pruning (or replacing) a weight from self.mask_weight
        self.fixed_weight = nn.Parameter(torch.ones(self.weight.size()))

    def update_mask_weight(self, out_idx, in_idx, value):
        self.mask_weight[out_idx, in_idx] = value

    def update_mask_bias(self, out_idx, value):
        self.mask_bias[out_idx] = value

    def set_fixed_mask(self, mask_weight, mask_bias):
        # argument mask is some mask learned from a different pruning algorithm. must have the same shape as self.mask_weight
        self.fixed_weight = nn.Parameter(mask_weight.detach().clone())
        if mask_bias is not None:
            self.fixed_bias = nn.Parameter(mask_bias.detach().clone())

    def forward(self, x):
        w = self.weight * self.mask_weight * self.fixed_weight

        if self.bias == None:
            b = None
        else:
            b = self.bias * self.mask_bias * self.fixed_bias

        return F.linear(x, w, b)


class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = args[0]
        self.out_channels = args[1]
        self.mask_weight = nn.Parameter(torch.ones(self.weight.size()))

        if kwargs['bias']:
            self.mask_bias = nn.Parameter(torch.ones(self.bias.size()))
            self.fixed_bias = nn.Parameter(torch.ones(self.bias.size()))

        # create a list of boolean flags to indicate whether each activation has been pruned. This is used when
        # args.pruning_strategy is set to "activations_and_weights." Once activations are pruned, we do not went to check
        # whether or not to prune a weight that belongs to that activation, so we check this list first before checking.
        self.pruned_activation = [False] * len(self.weight)

        # create a second mask weight that captures permanent changes to the pruned network. This matrix is relevant 
        # when the network is pruned by activation first and then weights, or when EP is used, followed by greedy pruning.
        # this fixed mask is checked before pruning (or replacing) a weight from self.mask_weight
        self.fixed_weight = nn.Parameter(torch.ones(self.weight.size()))

    def set_fixed_mask(self, mask_weight, mask_bias):
        # argument mask is some mask learned from a different pruning algorithm. must have the same shape as self.mask_weight
        self.fixed_weight = nn.Parameter(mask_weight.detach().clone())
        if mask_bias is not None:
            self.fixed_bias = nn.Parameter(mask_bias.detach().clone())

    def forward(self, x):
        w = self.weight * self.mask_weight * self.fixed_weight

        if self.bias == None:
            b = None
        else:
            b = self.bias * self.mask_bias * self.fixed_bias

        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
