# https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
# https://towardsdatascience.com/resnets-for-cifar-10-e63e900524e0

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from utils.builder import get_builder
from args_helper import parser_args
from utils.net_utils import prune

# def _weights_init(m):
#     classname = m.__class__.__name__
#     #print(classname)
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(in_planes, planes, stride=stride)
        self.bn1 = builder.batchnorm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=1)
        self.bn2 = builder.batchnorm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, builder, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.builder = builder

        self.conv1 = builder.conv3x3(3, 16, stride=1, first_layer=True)
        self.bn1 = builder.batchnorm(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = builder.conv1x1(64 * block.expansion, 10) # 10 = num_classes for cifar10

        self.prunable_layer_names, self.prunable_biases = self.get_prunable_param_names()


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = self.fc(out)
        return out.flatten(1)


def resnet20():
    return ResNet(get_builder(), BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(get_builder(), BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(get_builder(), BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(get_builder(), BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(get_builder(), BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(get_builder(), BasicBlock, [200, 200, 200])
