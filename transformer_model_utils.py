from utils.builder import get_builder
# from args_helper import parser_args
# from utils.net_utils import prune
import torch.nn as nn
import collections.abc
from itertools import repeat
from torch.nn import ModuleList
import copy
import torch
import math


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


class BertSelfAttention(nn.Module):
    def __init__(self, builder, hidden_size, num_attention_heads, dropout):
        super().__init__()

        self.num_attention_heads = num_attention_heads  # 2
        self.attention_head_size = int(hidden_size / num_attention_heads)  
        # hidden_size = 200, num_attention_heads = 2, self.attention_head_size = 100
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.all_head_size = 2 * 100 = 200

        # self.query = nn.Linear(hidden_size, self.all_head_size)  # 200 -> 200
        self.query = builder.conv1x1(hidden_size, self.all_head_size)        
        self.key = builder.conv1x1(hidden_size, self.all_head_size)  # 200 -> 200
        self.value = builder.conv1x1(hidden_size, self.all_head_size)  # 200 -> 200

        self.dropout = nn.Dropout(dropout)
        
    # def transpose_for_scores(self, x):
    #     new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [35, 20, 2, 100]
    #     x = x.view(*new_x_shape)
    #     x = x.permute(1, 2, 0, 3).contiguous()  # [20, 2, 35, 100]
    #     x = x.view(-1, x.size()[2], x.size()[3])  # [40, 35, 100]
    #     return x  # x.permute(0, 2, 1, 3)

    def transpose_for_scores(self, x, N, C, H):  # [700, 200, 1, 1]
        # N: 35, C: 20, H: 200
        x = x.view(N * C, H).contiguous()  # [700, 200]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [35, 20, 2, 100]
        x = x.view(*new_x_shape)
        x = x.permute(1, 2, 0, 3).contiguous()  # [20, 2, 35, 100]
        x = x.view(-1, x.size()[2], x.size()[3])  # [40, 35, 100]
        return x  # x.permute(0, 2, 1, 3)

    def transpose_back(self, x):  # [40, 35, 100]
        dim1, dim2, dim3 = x.shape  # dim1 = 40, dim2 = 35, dim3 = 100
        x = x.view(-1, self.num_attention_heads, dim2, dim3)  # [20, 2, 35, 100]
        x = x.permute(2, 0, 1, 3).contiguous()  # -> [35, 20, 2, 100]
        x = x.view(dim2, x.size()[1], -1)  # -> [35, 20, 200]
        return x


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # hidden_states shape [35, 20, 200]
        N, C, H = hidden_states.shape
        hidden_states = hidden_states.view(N * C, H, 1, 1).contiguous()。# -> [700, 200, 1, 1]
        # linear layer [700, 200, 1, 1] -> [700, 200, 1, 1]
        key_layer = self.transpose_for_scores(self.key(hidden_states), N, C, H)  # [40, 35, 100]
        value_layer = self.transpose_for_scores(self.value(hidden_states), N, C, H)  # [40, 35, 100]
        query_layer = self.transpose_for_scores(self.query(hidden_states), N, C, H)  # [40, 35, 100]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [40, 35, 35]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_mask = attention_mask.unsqueeze(0)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # [40, 35, 35]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # [40, 35, 35] @ [40, 35, 100] -> [40, 35, 100]
        context_layer = self.transpose_back(context_layer)  # [35, 20, 200]
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else context_layer

        return outputs



class Block(nn.Module):

    def __init__(self, builder, dim, num_heads, mlp_hidden_dim, qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim, elementwise_affine=False)
        self.attn = BertSelfAttention(builder, hidden_size=dim, num_attention_heads=num_heads, dropout=drop)
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=False)
        self.mlp = Mlp(builder, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, src_mask, src_key_padding_mask=None):
        x = x + self.attn(self.norm1(x), attention_mask=src_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, builder, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = builder.conv1x1(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = builder.conv1x1(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x
