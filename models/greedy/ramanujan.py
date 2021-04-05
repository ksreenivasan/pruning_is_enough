import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mask_layers import MaskLinear, MaskConv


class Net(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(Net, self).__init__()
        self.conv1 = MaskConv(1, 32, kernel_size=(3, 3), stride=1, bias=args.bias)
        self.conv2 = MaskConv(32, 64, kernel_size=(3, 3), stride=1, bias=args.bias)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = MaskLinear(9216, 128, bias=args.bias)
        self.fc2 = MaskLinear(128, num_classes, bias=args.bias)

        self.num_activations = len(self.conv1.weight) + len(self.conv2.weight) + \
                len(self.fc1.weight) + len(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)

        return out


class Conv4(nn.Module):
    def __init__(self, input_size, num_classes, args):
        super(Conv4, self).__init__()
        self.conv1 = MaskConv(3, 64, kernel_size=(3, 3), stride=1, bias=args.bias)
        self.conv2 = MaskConv(64, 64, kernel_size=(3, 3), stride=1, bias=args.bias)
        self.conv3 = MaskConv(64, 128, kernel_size=(3, 3), stride=1, bias=args.bias)
        self.conv4 = MaskConv(128, 128, kernel_size=(3, 3), stride=1, bias=args.bias)
        self.fc1 = MaskLinear(8192, 256, bias=args.bias)
        self.fc2 = MaskLinear(256, 256, bias=args.bias)
        self.fc3 = MaskLinear(256, num_classes, bias=args.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = self.linear(out)
        return out.squeeze()

