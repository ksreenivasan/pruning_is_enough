import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math

# BasicBlock {{{
class BasicBlock(nn.Module):
    M = 2
    expansion = 1

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(BasicBlock, self).__init__()
        if base_width / 64 > 1:
            raise ValueError("Base width >64 does not work for BasicBlock")

        self.conv1 = builder.conv3x3(inplanes, planes, stride)
        self.bn1 = builder.batchnorm(planes)
        self.relu = builder.activation()
        self.conv2 = builder.conv3x3(planes, planes)
        self.bn2 = builder.batchnorm(planes, last_bn=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.bn1 is not None:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    M = 3
    expansion = 4

    def __init__(self, builder, inplanes, planes, stride=1, downsample=None, base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * base_width / 64)
        self.conv1 = builder.conv1x1(inplanes, width)
        self.bn1 = builder.batchnorm(width)
        self.conv2 = builder.conv3x3(width, width, stride=stride)
        self.bn2 = builder.batchnorm(width)
        self.conv3 = builder.conv1x1(width, planes * self.expansion)
        self.bn3 = builder.batchnorm(planes * self.expansion, last_bn=True)
        self.relu = builder.activation()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


# Bottleneck }}}

# ResNet {{{
class ResNet(nn.Module):
    def __init__(self, builder, block, layers, num_classes=1000, base_width=64):
        self.args_first_layer_dense = False
        self.args_last_layer_dense = False
        self.args_bias = False

        self.inplanes = 64
        super(ResNet, self).__init__()

        self.base_width = base_width
        if self.base_width // 64 > 1:
            print(f"==> Using {self.base_width // 64}x wide model")

        if self.args_first_layer_dense:
            self.conv1 = nn.Conv2d(
                3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = builder.conv7x7(3, 64, stride=2, first_layer=True)

        self.bn1 = builder.batchnorm(64)
        self.relu = builder.activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(builder, block, 64, layers[0])
        self.layer2 = self._make_layer(builder, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(builder, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(builder, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        if self.args_last_layer_dense:
            self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)
        else:
            # self.fc = builder.conv1x1(512 * block.expansion, num_classes)
            self.fc = builder.linear(512 * block.expansion, num_classes)
        
        self.prunable_layer_names, self.prunable_biases = self.get_prunable_param_names()

    def _make_layer(self, builder, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            dconv = builder.conv1x1(
                self.inplanes, planes * block.expansion, stride=stride
            )
            dbn = builder.batchnorm(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        layers = []
        layers.append(block(builder, self.inplanes, planes, stride, downsample, base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, base_width=self.base_width))

        return nn.Sequential(*layers)

    def get_prunable_param_names(self):
        prunable_weights = [name + '.weight' for name, module in self.named_modules() if
                isinstance(module, nn.modules.conv.Conv2d) or
                isinstance(module, nn.modules.linear.Linear)]
        if self.args_bias:
            prunable_biases = [name + '.bias' for name, module in self.named_modules() if
                isinstance(module, nn.modules.conv.Conv2d) or
                isinstance(module, nn.modules.linear.Linear)]
        else:
            prunable_biases = [""]

        return prunable_weights, prunable_biases

    def forward(self, x):
        # update score thresholds for global ep
        # TODO: Don't need this for now
        # if parser_args.algo in ['global_ep', 'global_ep_iter'] or parser_args.bottom_k_on_forward:
        #     prune(self, update_thresholds_only=True)
        x = self.conv1(x)

        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.view(x.size(0), -1)

        return x


# ResNet }}}
def ResNet18(pretrained=False):
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2], 1000)


def ResNet50(pretrained=False):
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3], 1000)


def ResNet101(pretrained=False):
    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3], 200)
    #return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3], 1000)


def WideResNet50_2(pretrained=False):
    return ResNet(
        get_builder(), Bottleneck, [3, 4, 6, 3], num_classes=1000, base_width=64 * 2
    )


def WideResNet101_2(pretrained=False):
    return ResNet(
        get_builder(), Bottleneck, [3, 4, 23, 3], num_classes=1000, base_width=64 * 2
    )


class Builder(object):
    def __init__(self, conv_layer, bn_layer, first_layer=None, weight_init="signed_constant"):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        # self.first_layer = first_layer or conv_layer
        self.first_layer = conv_layer
        self.weight_init = weight_init

    def conv(self, kernel_size, in_planes, out_planes, stride=1, first_layer=False, groups=1):
        # conv_layer = self.first_layer if first_layer else self.conv_layer
        conv_layer = self.conv_layer

        if first_layer:
            print(f"==> Building first layer")

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True,
                groups=groups
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes,
                kernel_size=1,
                stride=stride,
                bias=True,
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=True,
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=True,
            )
        else:
            return None

        self._init_conv(conv)

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1, first_layer=False, groups=1):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, first_layer=first_layer, groups=groups)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, first_layer=False):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def linear(self, in_planes, out_planes):
        l = SubnetLinear(in_planes, out_planes, bias=True)
        self._init_conv(l)
        return l

    def conv7x7(self, in_planes, out_planes, stride=1, first_layer=False):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, first_layer=False):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, first_layer=first_layer)
        return c

    def batchnorm(self, planes, last_bn=False, first_layer=False):
        return self.bn_layer(planes)

    def activation(self):
        # always ReLU
        return (lambda: nn.ReLU(inplace=True))()

    def _init_conv(self, conv):
        if self.weight_init == "signed_constant":
            fan = nn.init._calculate_correct_fan(conv.weight, "fan_in")
            # scale_fan = False always because this isn't EP
            # if scale_fan:
            #     fan = fan * (1 - parser_args.prune_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            conv.weight.data = conv.weight.data.sign() * std

        elif self.weight_init == "unsigned_constant":
            fan = nn.init._calculate_correct_fan(conv.weight, "fan_in")
            # scale_fan = False always because this isn't EP
            # if parser_args.scale_fan:
            #     fan = fan * (1 - parser_args.prune_rate)
            gain = nn.init.calculate_gain("relu")
            std = gain / math.sqrt(fan)
            conv.weight.data = torch.ones_like(conv.weight.data) * std

        elif self.weight_init == "kaiming_normal":
            # scale_fan = False always because this isn't EP
            # if parser_args.scale_fan:
            #     fan = nn.init._calculate_correct_fan(conv.weight, parser_args.mode)
            #     fan = fan * (1 - parser_args.prune_rate)
            #     gain = nn.init.calculate_gain(parser_args.nonlinearity)
            #     std = gain / math.sqrt(fan)
            #     with torch.no_grad():
            #         conv.weight.data.normal_(0, std)
            # else:
            #     nn.init.kaiming_normal_(
            #         conv.weight, mode=parser_args.mode, nonlinearity=parser_args.nonlinearity
            #     )
            nn.init.kaiming_normal_(
                    conv.weight, mode="fan_in", nonlinearity="relu"
                )

        elif self.weight_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                conv.weight, mode="fan_in", nonlinearity="relu"
            )

        elif self.weight_init == "xavier_normal":
            nn.init.xavier_normal_(conv.weight)

        elif self.weight_init == "xavier_constant":
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(conv.weight)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            conv.weight.data = conv.weight.data.sign() * std

        elif self.weight_init == "standard":
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))

        else:
            raise ValueError(f"{weight_init} is not an initialization option!")


def get_builder():
    conv_type = "SubnetConv"
    bn_type = "AffineBatchNorm" # TODO: might change this if it causes problems later
    first_layer_type = None # TODO: I think
    weight_init = "kaiming_normal" # TODO: this is only for debugging wt training

    print("==> Conv Type: {}".format(conv_type))
    print("==> BN Type: {}".format(bn_type))

    # need to fix this
    # conv_layer = getattr(utils.conv_type, parser_args.conv_type)
    # bn_layer = getattr(utils.bn_type, parser_args.bn_type)
    conv_layer = SubnetConv
    bn_layer = AffineBatchNorm

    if first_layer_type is not None:
        first_layer = getattr(utils.conv_type, first_layer_type)
        print(f"==> First Layer Type: {first_layer_type}")
    else:
        first_layer = None

    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer, first_layer=first_layer, weight_init=weight_init)

    return builder

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, bias_scores, k, scores_prune_threshold=-np.inf, bias_scores_prune_threshold=-np.inf):
        algo = 'hc_iter'
        quantize_threshold = 0.5
        if algo == 'ep':
            # Get the supermask by sorting the scores and using the top k%
            out = scores.clone()
            _, idx = scores.flatten().sort()
            j = int((1 - k) * scores.numel())
            # flat_out and out access the same memory.
            flat_out = out.flatten()
            flat_out[idx[:j]] = 0
            flat_out[idx[j:]] = 1

            # repeat for bias
            # Get the supermask by sorting the scores and using the top k%
            bias_out = bias_scores.clone()
            _, idx = bias_scores.flatten().sort()
            j = int((1 - k) * bias_scores.numel())

            # flat_out and out access the same memory.
            bias_flat_out = bias_out.flatten()
            bias_flat_out[idx[:j]] = 0
            bias_flat_out[idx[j:]] = 1

        elif algo in ['global_ep', 'global_ep_iter']:
            # define out, bias_out based on the layer's prune_threshold, bias_threshold
            out = torch.gt(scores, torch.ones_like(scores)*scores_prune_threshold).float()
            bias_out = torch.gt(bias_scores, torch.ones_like(bias_scores)*bias_scores_prune_threshold).float()

        elif algo in ['hc', 'hc_iter']:
            # round scores to {0, 1}
            out = torch.gt(scores, torch.ones_like(scores)*quantize_threshold).float()
            bias_out = torch.gt(bias_scores, torch.ones_like(bias_scores)*quantize_threshold).float()

        else:
            print("INVALID PRUNING ALGO")
            print("EXITING")
            exit()

        return out, bias_out

    @staticmethod
    def backward(ctx, g_1, g_2):
        # send the gradient g straight-through on the backward pass.
        return g_1, g_2, None, None, None


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args_bias = False
        self.algo = 'hc'
        self.prune_rate = 0.5
        # resnet50 has bias=False because of BN layers
        self.bias = None

        # initialize flag (representing the pruned weights)
        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        if self.args_bias:
            self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_flag = nn.Parameter(torch.Tensor(1))

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if self.args_bias:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_scores = nn.Parameter(torch.Tensor(1))
        
        # prune scores below this for global EP in bottom-k
        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf
        
        if self.algo in ['hc', 'hc_iter']:
            # score init is always uniform
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0) # can't do kaiming here. picking U[-1, 1] for no real reason

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.flag.requires_grad = False
        self.bias_flag.requires_grad = False

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        if self.algo in ['hc', 'hc_iter', 'transformer']:
            subnet, bias_subnet = GetSubnet.apply(self.scores, self.bias_scores, self.prune_rate)
            subnet = subnet * self.flag.data.float()
            bias_subnet = subnet * self.bias_flag.data.float()
        elif self.algo in ['imp']:
            # no STE, no subnet. Mask is handled outside
            pass
        elif self.algo in ['global_ep', 'global_ep_iter']:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0, self.scores_prune_threshold, self.bias_scores_prune_threshold)
        else:
            # ep, global_ep, global_ep_iter, pt etc
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), self.prune_rate)

        if self.algo in ['imp']:
            # no STE, no subnet. Mask is handled outside
            w = self.weight
            b = self.bias
        else:
            w = self.weight * subnet
            if self.args_bias:
                b = self.bias * bias_subnet
            else:
                b = self.bias

        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )

        return x


class SubnetLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: hacky. trying to mimic frankle in having biases but not pruning them
        self.args_bias = False
        self.algo = 'hc'
        self.prune_rate = 0.5
        # resnet50 has bias=True only for the FC layer

        # initialize flag (representing the pruned weights)
        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        if self.args_bias:
            self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_flag = nn.Parameter(torch.Tensor(1))

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if self.args_bias:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            # dummy variable just so other things don't break
            self.bias_scores = nn.Parameter(torch.Tensor(1))
        
        # prune scores below this for global EP in bottom-k
        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf
        
        if self.algo in ['hc', 'hc_iter']:
            # score init is always uniform
            nn.init.uniform_(self.scores, a=0.0, b=1.0)
            nn.init.uniform_(self.bias_scores, a=0.0, b=1.0)
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
            nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0) # can't do kaiming here. picking U[-1, 1] for no real reason

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.flag.requires_grad = False
        self.bias_flag.requires_grad = False
        # TODO: Hacky. I'm trying to mimic frankle etc in that we have biases, but we don't prune them
        self.bias.requires_grad = False

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        if self.algo in ['hc', 'hc_iter']:
            # don't need a mask here. the scores are directly multiplied with weights
            subnet, bias_subnet = GetSubnet.apply(self.scores, self.bias_scores, self.prune_rate)
            subnet = subnet * self.flag.data.float()
            bias_subnet = subnet * self.bias_flag.data.float()
        elif self.algo in ['imp']:
            # no STE, no subnet. Mask is handled outside
            pass
        elif parser_args.algo in ['global_ep', 'global_ep_iter']:
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0, self.scores_prune_threshold, self.bias_scores_prune_threshold)
        else:
            # ep, global_ep, global_ep_iter, pt etc
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), self.prune_rate)

        if self.algo in ['imp']:
            # no STE, no subnet. Mask is handled outside
            w = self.weight
            b = self.bias
        else:
            w = self.weight * subnet
            if self.args_bias:
                b = self.bias * bias_subnet
            else:
                b = self.bias

        x = F.linear(x, w, b)
        return x


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


class AffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(AffineBatchNorm, self).__init__(dim, affine=True)
