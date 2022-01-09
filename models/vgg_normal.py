#copied and edited from: 
# 1. https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html#vgg16 
# 2. https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py

import torch
import torch.nn as nn

from args_helper import parser_args
from utils.builder import get_builder
from utils.net_utils import prune

class VGG16_Normal(nn.Module):
    def __init__(self, btn):
        super(VGG16_Normal, self).__init__()
        self.features = self.make_layers()
        self.classifier = nn.Linear(512, 10) 
    def forward(self, x):  
        x = self.features(x)
        x = self.classifier(x) 
        x = x.view(x.size(0), -1)
        return x  #.flatten(1)

    def make_layers(self, batch_norm=False):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace-True)]
                in_channels = v
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, builder, btn):
        super(VGG16, self).__init__()
        self.builder = builder
        self.features = self.make_layers()
        self.classifier = builder.conv1x1(512, 10) 
    def forward(self, x):  
        x = self.features(x)
        x = self.classifier(x) 
        x = x.view(x.size(0), -1)
        return x  #.flatten(1)

    def make_layers(self, batch_norm=False):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = self.builder.conv3x3(in_channels, v)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, self.builder.activation()]
                in_channels = v
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def vgg16_normal(pretrained=False):
    return VGG16_Normal(get_builder(), False)

def vgg16(pretrained=False):
    return VGG16(get_builder(), False)
