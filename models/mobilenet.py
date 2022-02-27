#mobilenet.py

'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.builder import get_builder
from args_helper import parser_args


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, builder, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = builder.conv1x1(in_planes, planes, stride=1)
        self.bn1 = builder.batchnorm(planes)
        #import pdb; pdb.set_trace()
        self.conv2 = builder.conv3x3(planes, planes, stride=stride, groups=planes)
        self.bn2 = builder.batchnorm(planes)
        self.conv3 = builder.conv1x1(planes, out_planes, stride=1)
        self.bn3 = builder.batchnorm(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, out_planes, stride=1),
                builder.batchnorm(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNet_base(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, builder, num_classes=10):
        super(MobileNet_base, self).__init__()

        self.builder = builder
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = builder.conv3x3(3, 32, stride=1)
        self.bn1 = builder.batchnorm(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = builder.conv1x1(320, 1280, stride=1)
        self.bn2 = builder.batchnorm(1280)
        self.linear = builder.conv1x1(1280, num_classes) # original model: bias=True

        self.prunable_layer_names, self.prunable_biases = self.get_prunable_param_names()

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(self.builder, in_planes, out_planes, expansion, stride))
                in_planes = out_planes
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
        #print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        #print(out.shape)
        out = self.layers(out)
        #print(out.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        #print(out.shape)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        #print(out.shape)
        out = self.linear(out)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        return out

def MobileNetV2():
    return MobileNet_base(get_builder())

def tinyMobileNetV2(num_classes=200):
    return MobileNet_base(get_builder(), num_classes=num_classes)

'''
class BlockNormal(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(BlockNormal, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2Normal(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2Normal, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(BlockNormal(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        #print(out.shape)
        out = self.layers(out)
        #print(out.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        #print(out.shape)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.linear(out)
        #print(out.shape)
        return out


def test():
    net = MobileNetV2Normal()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
'''
