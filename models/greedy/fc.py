import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mask_layers import MaskLinear
from utils.net_utils import step

class TwoLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(TwoLayerFC, self).__init__()
        self.fc1 = MaskLinear(input_size, args.hidden_size, bias=args.bias)
        self.fc2 = MaskLinear(args.hidden_size, num_classes, bias=args.bias)

        self.num_activations = len(self.fc1.weight) + len(self.fc2.weight)
    
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output


class FourLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(FourLayerFC, self).__init__()
        self.fc1 = MaskLinear(input_size, args.hidden_size, bias=args.bias)
        self.fc2 = MaskLinear(args.hidden_size, args.hidden_size, bias=args.bias)
        self.fc3 = MaskLinear(args.hidden_size, args.hidden_size, bias=args.bias)
        self.fc4 = MaskLinear(args.hidden_size, num_classes, bias=args.bias)

        self.num_activations = len(self.fc1.weight) + len(self.fc2.weight) + \
                len(self.fc3.weight) + len(self.fc4.weight)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)

        return output


class BaselineTwoLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(BaselineTwoLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, args.hidden_size, bias=args.bias)
        self.fc2 = nn.Linear(args.hidden_size, num_classes, bias=args.bias)
    
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)

        return output


class BaselineFourLayerFC(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(BaselineFourLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, args.hidden_size, bias=args.bias)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size, bias=args.bias)
        self.fc3 = nn.Linear(args.hidden_size, args.hidden_size, bias=args.bias)
        self.fc4 = nn.Linear(args.hidden_size, num_classes, bias=args.bias)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)

        return output


