import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mask_layers import MaskLinear, MaskConv


class LeNet5(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(LeNet5, self).__init__()
        self.conv1 = MaskConv(1, 6, kernel_size=(5, 5), padding=2, bias=args.bias)
        self.conv2 = MaskConv(6, 16, kernel_size=(5, 5), bias=args.bias)
        self.fc1 = MaskLinear(400, 120, bias=args.bias)
        self.fc2 = MaskLinear(120, 84, bias=args.bias)
        self.fc3 = MaskLinear(84, num_classes, bias=args.bias)

        self.num_activations = len(self.conv1.weight) + len(self.conv2.weight) + \
                len(self.fc1.weight) + len(self.fc2.weight) + len(self.fc3.weight)

    def forward(self, x): 
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = x.view(-1, 16*5*5)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=-1)

        return output


class BaselineLeNet5(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(BaselineLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2, bias=args.bias)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), bias=args.bias)
        self.fc1 = nn.Linear(400, 120, bias=args.bias)
        self.fc2 = nn.Linear(120, 84, bias=args.bias)
        self.fc3 = nn.Linear(84, num_classes, bias=args.bias)

    def forward(self, x): 
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        x = x.view(-1, 16*5*5)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=-1)

        return output


