# based off of: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

import torch
import torch.nn as nn
import collections

from args_helper import parser_args
from utils.builder import get_builder
from models.efficient_utils import *

class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, builder, block_args, global_params, image_size=None):
        super().__init__()
        self.builder = builder
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = self.builder.conv1x1(in_channels=inp, out_channels=oup)
            self._bn0 = builder.batchnorm(oup, momentum=self._bn_mom, eps=self._bn_eps)
            #nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = builder.conv(k, in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
                             stride=s)
        self._bn1 = builder.batchnorm(oup, momentum=self._bn_mom, eps=self._bn_eps)
        #nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        calculate_output_image_size(s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            #Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = builder.conv1x1(oup, num_squeezed_channels)
            #Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = builder.conv1x1(num_squeezed_channels,oup)
            #Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        #Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = builder.conv1x1(oup, final_oup)
        #self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = builder.batchnorm(final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        #nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


########################################################################################################################################################################


class EfficientNet(nn.Module):
    def __init__(self, builder, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        parser_args.img_height = global_params.image_size
        parser_args.img_width = global_params.image_size

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = builder.conv3x3(in_channels, out_channels, stride=2)

        self._bn0 = builder.batchnorm(out_channels, momentum=bn_mom, eps=bn_eps)
        calculate_output_image_size(2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(builder, block_args, self._global_params))
            calculate_output_image_size(block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(builder, block_args, self._global_params))#, image_size=image_size))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = builder.conv1x1(in_channels, out_channels)
        self._bn1 = builder.batchnorm(out_channels, momentum=bn_mom, eps=bn_eps)
        #nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = builder.conv1x1(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def extract_fetaures(self, inputs):
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

########################################################################################################################################################################

def TinyEfficientNet(pretrained=False):

    # params_dict = {
    #     # Coefficients:   width,depth,res,dropout
    #     'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    #     'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    #     'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    #     'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    #     'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    #     'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    #     'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    #     'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    #     'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    #     'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    # maually forcing this network to be efficnetnet-b1 - incorperate the above dict to make it modular

    # get_info will return the blocks and params needed for building the network - in models/efficient_util
    blocks_args, global_params = get_info(width_coefficient=1.0, depth_coefficient=1.1, dropout_rate=0.2, image_size=64)
    return EfficientNet(get_builder(), blocks_args, global_params) 

    # model params 'efficientnet-b1': (widht = 1.0, depth = 1.1, im_size = 240, dropout_rate = 0.2),


# cls._check_model_name_is_valid(model_name)
#         blocks_args, global_params = get_model_params(model_name, override_params)
#         model = cls(blocks_args, global_params)
#         model._change_in_channels(in_channels)
#         return model