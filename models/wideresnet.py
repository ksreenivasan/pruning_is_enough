"""
    Wideresnet for Cifar10
    Source: https://github.com/xternalz/WideResNet-pytorch/
    (Official implementation is in torch and lua but this should work)
    Paper: https://arxiv.org/pdf/1605.07146.pdf
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.builder import get_builder
from args_helper import parser_args
from utils.net_utils import prune


class BasicBlock(nn.Module):
    def __init__(self, builder, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.builder = builder

        self.bn1 = builder.batchnorm(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = builder.conv3x3(in_planes, out_planes, stride=stride)
        self.bn2 = builder.batchnorm(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = builder.conv3x3(out_planes, out_planes, stride=1)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and builder.conv1x1(in_planes, out_planes,
                             stride=stride) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, builder, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(builder, block, in_planes, out_planes, nb_layers, stride, dropRate)
        self.builder = builder

    def _make_layer(self, builder, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(builder, i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, builder, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        self.builder = builder
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = builder.conv3x3(3, nChannels[0], stride=1)
        # 1st block
        self.block1 = NetworkBlock(builder, n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(builder, n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(builder, n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = builder.batchnorm(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = builder.conv1x1(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # TODO: for IMP. might break right now
        self.prunable_layer_names, self.prunable_biases = self.get_prunable_param_names()

        """
        # TODO: deleting weight init because it should be handled by subnetconv
        # let's hope it works well.
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
        """

    def get_prunable_param_names(model):
        prunable_weights = [name + '.weight' for name, module in model.named_modules() if
                isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)]
        if parser_args.bias:
            prunable_biases = [name + '.bias' for name, module in model.named_modules() if
                isinstance(module, torch.nn.modules.conv.Conv2d) or
                isinstance(module, torch.nn.modules.linear.Linear)]
        else:
            prunable_biases = [""]

        return prunable_weights, prunable_biases

    def forward(self, x):
        # update score thresholds for global ep
        if parser_args.algo in ['global_ep', 'global_ep_iter'] or parser_args.bottom_k_on_forward:
            prune(self, update_thresholds_only=True)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

"""
    TODO: for now, hardcoding widen_factor to be 2,
          and dropout_probablity to be 0.
          Will add models where these things are different as we go on.
"""
def WideResNet28():
    return WideResNet(get_builder(), depth=28, num_classes=10, widen_factor=2,
                      dropRate=0)
