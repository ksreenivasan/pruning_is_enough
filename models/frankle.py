"""
Replications of models from Frankle et al. Lottery Ticket Hypothesis
"""

import torch
import torch.nn as nn
from utils.builder import get_builder

from args_helper import parser_args
from utils.net_utils import prune

class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(64 * 16 * 16, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        # update score thresholds for global ep
        if parser_args.algo in ['global_ep', 'global_ep_iter'] or parser_args.bottom_k_on_forward:
            prune(self, update_thresholds_only=True)
        out = self.convs(x)
        out = out.view(out.size(0), 64 * 16 * 16, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv4(nn.Module):
    def __init__(self, width=1.0):
        super(Conv4, self).__init__()
        builder = get_builder()
        print(width)
        self.width = width
        self.n0 = (int)(8*width)
        self.n1 = (int)(64*width)
        self.n2 = (int)(128*width)
        self.n3 = (int)(256*width)

        self.convs = nn.Sequential(
            builder.conv3x3(3, self.n1, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(self.n1, self.n1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(self.n1, self.n2),
            nn.ReLU(),
            builder.conv3x3(self.n2, self.n2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(32 * 32 * self.n0, self.n3),
            nn.ReLU(),
            builder.conv1x1(self.n3, self.n3),
            nn.ReLU(),
            builder.conv1x1(self.n3, 10),
        )

    def forward(self, x):
        # update score thresholds for global ep
        if parser_args.algo in ['global_ep', 'global_ep_iter'] or parser_args.bottom_k_on_forward:
            prune(self, update_thresholds_only=True)
        out = self.convs(x)
        out = out.view(out.size(0), (int)(8192 * self.width), 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv4Normal(nn.Module):
    def __init__(self, width=1.0):
        super(Conv4Normal, self).__init__()
        print(width)
        self.width = width
        self.n0 = (int)(8*width)
        self.n1 = (int)(64*width)
        self.n2 = (int)(128*width)
        self.n3 = (int)(256*width)

        self.convs = nn.Sequential(
            nn.Conv2d(3, self.n1, 3, 1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.n1, self.n1, 3, 1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(self.n1, self.n2, 3, 1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.n2, self.n2, 3, 1, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            nn.Conv2d(32 * 32 * self.n0, self.n3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.n3, self.n3, 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.n3, 10, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), (int)(8192 * self.width), 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(256 * 4 * 4, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        # update score thresholds for global ep
        if parser_args.algo in ['global_ep', 'global_ep_iter'] or parser_args.bottom_k_on_forward:
            prune(self, update_thresholds_only=True)
        out = self.convs(x)
        out = out.view(out.size(0), 256 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()

class Conv8(nn.Module):
    def __init__(self):
        super(Conv8, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(256, 512),
            nn.ReLU(),
            builder.conv3x3(512, 512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 2 * 2, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        # update score thresholds for global ep
        if parser_args.algo in ['global_ep', 'global_ep_iter']:
            prune(self, update_thresholds_only=True)
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        builder = get_builder()
        self.linear = nn.Sequential(
            builder.conv1x1(28 * 28, 300, first_layer=True),
            nn.ReLU(),
            builder.conv1x1(300, 100),
            nn.ReLU(),
            builder.conv1x1(100, 10),
        )

    def forward(self, x):
        # update score thresholds for global ep
        if parser_args.algo in ['global_ep', 'global_ep_iter'] or parser_args.bottom_k_on_forward:
            prune(self, update_thresholds_only=True)
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = self.linear(out)
        return out.squeeze()


def scale(n):
    return int(n * args.width_mult)


class Conv4Wide(nn.Module):
    def __init__(self):
        super(Conv4Wide, self).__init__()
        builder = get_builder()

        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            nn.ReLU(),
            builder.conv3x3(scale(128), scale(128)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(128)*8*8, scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        # update score thresholds for global ep
        if parser_args.algo in ['global_ep', 'global_ep_iter'] or parser_args.bottom_k_on_forward:
            prune(self, update_thresholds_only=True)
        out = self.convs(x)
        out = out.view(out.size(0), scale(128)*8*8, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6Wide(nn.Module):
    def __init__(self):
        super(Conv6Wide, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            nn.ReLU(),
            builder.conv3x3(scale(128), scale(128)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(128), scale(256)),
            nn.ReLU(),
            builder.conv3x3(scale(256), scale(256)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(256) * 4 * 4, scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        # update score thresholds for global ep
        if parser_args.algo in ['global_ep', 'global_ep_iter'] or parser_args.bottom_k_on_forward:
            prune(self, update_thresholds_only=True)
        out = self.convs(x)
        out = out.view(out.size(0), scale(256) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()
