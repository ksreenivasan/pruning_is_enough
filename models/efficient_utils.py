import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from utils.builder import get_builder
from utils.conv_type import SubnetConv
from args_helper import parser_args

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

# Swish activation function
if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.
    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.
    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))

def drop_connect(inputs, p, training):
    """Drop connect.
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.
    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output

def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'
        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            id_skip=('noskip' not in block_string))


def get_width_and_height_from_size(x):
    """Obtain height and width from x.
    Args:
        x (int, tuple or list): Data size.
    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.
    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.
    Returns:
        output_image_size: A list [H,W].
    """
    # if input_image_size is None:
    #     return None
    # image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    parser_args.img_height = int(math.ceil(parser_args.img_height / stride))
    parser_args.img_width = int(math.ceil(parser_args.img_width / stride))


# i think we wonly ned the static part of this method
# def get_same_padding_conv2d(image_size=None):
#     """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
#        Static padding is necessary for ONNX exporting of models.
#     Args:
#         image_size (int or tuple): Size of the image.
#     Returns:
#         Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
#     """

#     # I do not think we need this
#     # if image_size is None:
#     #     return Conv2dDynamicSamePadding
#     # else:
#     return partial(Conv2dStaticSamePadding, image_size=image_size)



# class Conv2dStaticSamePadding(nn.Conv2d):
#     """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
#        The padding mudule is calculated in construction function, then used in forward.
#     """

#     # With the same calculation as Conv2dDynamicSamePadding

#     def __init__(self, in_channels, out_channels, builder, kernel_size, stride=1, image_size=None, **kwargs):
#         super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
#         self.builder = builder
#         self.kernel_size = kernel_size
#         self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

#         # Calculate padding based on image size and save it
#         assert image_size is not None
#         ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
#         kh, kw = self.weight.size()[-2:]
#         sh, sw = self.stride
#         oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
#         pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
#         pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
#         if pad_h > 0 or pad_w > 0:
#             self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
#                                                 pad_h // 2, pad_h - pad_h // 2))
#         else:
#             self.static_padding = nn.Identity()

#     def forward(self, x):
#         x = self.static_padding(x)
#         kernel_size, in_planes, out_planes, stride=1, first_layer=False, groups=1
#         builder.conv()
#         x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         return x


## 
# This method will return the block and params needed for building the network
# manually set all of these values for EfficientNet-B1
def get_info(width_coefficient, depth_coefficient, dropout_rate, image_size,
                drop_connect_rate=0.2, num_classes=200, include_top=True):
    string_list = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
        blocks_args.append(_decode_block_string(block_string))
    
    params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        dropout_rate=dropout_rate,

        num_classes=num_classes,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top,
    )

    return blocks_args, params