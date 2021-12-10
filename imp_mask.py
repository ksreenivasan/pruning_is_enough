# this code is modified based on
# https://github.com/facebookresearch/open_lth/blob/main/pruning/mask.py


import numpy as np
import torch

from args_helper import parser_args

class Mask(dict):
    def __init__(self, other_dict=None):  # IN USE
        super(Mask, self).__init__()
        if other_dict is not None:
            for k, v in other_dict.items(): self[k] = v

    def __setitem__(self, key, value):
        if not isinstance(key, str) or len(key) == 0:
            raise ValueError('Invalid tensor name: {}'.format(key))
        if isinstance(value, np.ndarray):
            value = torch.as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise ValueError('value for key {} must be torch Tensor or numpy ndarray.'.format(key))
        if ((value != 0) & (value != 1)).any(): raise ValueError('All entries must be 0 or 1.')

        super(Mask, self).__setitem__(key, value)

    @staticmethod
    def ones_like(model) -> 'Mask':  # IN USE
        mask = Mask()
        for name in model.prunable_layer_names:
            mask[name] = torch.ones(list(model.state_dict()[name].shape))
        if parser_args.bias:
            mask_bias = Mask()
            for name in model.prunable_biases:
                mask_bias[name] = torch.ones(list(model.state_dict()[name].shape))
        else:
            mask_bias = None
        return mask, mask_bias

    def numpy(self):  # IN USE
        return {k: v.cpu().numpy() for k, v in self.items()}

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in self.values()]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in self.values()]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity