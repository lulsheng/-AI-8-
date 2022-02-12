"""
## Uformer: A General U-Shaped Transformer for Image Restoration
## Zhendong Wang, Xiaodong Cun, Jianmin Bao, Jianzhuang Liu
## https://arxiv.org/abs/2106.03106
"""
from re import A
import x2paddle
from x2paddle import torch2paddle
import paddle
import paddle.nn as nn
from utils import to_2tuple, trunc_normal_, DropPath, masked_fill
import paddle.nn.functional as F

import math
import numpy as np


class ConvBlock(nn.Layer):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2D(in_channel,
                      out_channel,
                      kernel_size=3,
                      stride=strides,
                      padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv2D(out_channel,
                      out_channel,
                      kernel_size=3,
                      stride=strides,
                      padding=1), nn.LeakyReLU(inplace=True))
        self.conv11 = nn.Conv2D(in_channel,
                                out_channel,
                                kernel_size=1,
                                stride=strides,
                                padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

    def flops(self, H, W):
        flops = H * W * self.in_channel * self.out_channel * (
            3 * 3 + 1) + H * W * self.out_channel * self.out_channel * 3 * 3
        return flops


class UNet(nn.Layer):
    def __init__(self, block=ConvBlock, dim=32):
        super(UNet, self).__init__()
        self.dim = dim
        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1 = nn.Conv2D(dim, dim, kernel_size=4, stride=2, padding=1)
        self.ConvBlock2 = block(dim, dim * 2, strides=1)
        self.pool2 = nn.Conv2D(dim * 2,
                               dim * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.ConvBlock3 = block(dim * 2, dim * 4, strides=1)
        self.pool3 = nn.Conv2D(dim * 4,
                               dim * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.ConvBlock4 = block(dim * 4, dim * 8, strides=1)
        self.pool4 = nn.Conv2D(dim * 8,
                               dim * 8,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.ConvBlock5 = block(dim * 8, dim * 16, strides=1)
        self.upv6 = nn.Conv2DTranspose(dim * 16, dim * 8, 2, stride=2)
        self.ConvBlock6 = block(dim * 16, dim * 8, strides=1)
        self.upv7 = nn.Conv2DTranspose(dim * 8, dim * 4, 2, stride=2)
        self.ConvBlock7 = block(dim * 8, dim * 4, strides=1)
        self.upv8 = nn.Conv2DTranspose(dim * 4, dim * 2, 2, stride=2)
        self.ConvBlock8 = block(dim * 4, dim * 2, strides=1)
        self.upv9 = nn.Conv2DTranspose(dim * 2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim * 2, dim, strides=1)
        self.conv10 = nn.Conv2D(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.ConvBlock5(pool4)
        up6 = self.upv6(conv5)
        up6 = paddle.concat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)
        up7 = self.upv7(conv6)
        up7 = paddle.concat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)
        up8 = self.upv8(conv7)
        up8 = paddle.concat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)
        up9 = self.upv9(conv8)
        up9 = paddle.concat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)
        conv10 = self.conv10(conv9)
        out = x + conv10
        return out

    def flops(self, H, W):
        flops = 0
        flops += self.ConvBlock1.flops(H, W)
        flops += H / 2 * W / 2 * self.dim * self.dim * 4 * 4
        flops += self.ConvBlock2.flops(H / 2, W / 2)
        flops += H / 4 * W / 4 * self.dim * 2 * self.dim * 2 * 4 * 4
        flops += self.ConvBlock3.flops(H / 4, W / 4)
        flops += H / 8 * W / 8 * self.dim * 4 * self.dim * 4 * 4 * 4
        flops += self.ConvBlock4.flops(H / 8, W / 8)
        flops += H / 16 * W / 16 * self.dim * 8 * self.dim * 8 * 4 * 4
        flops += self.ConvBlock5.flops(H / 16, W / 16)
        flops += H / 8 * W / 8 * self.dim * 16 * self.dim * 8 * 2 * 2
        flops += self.ConvBlock6.flops(H / 8, W / 8)
        flops += H / 4 * W / 4 * self.dim * 8 * self.dim * 4 * 2 * 2
        flops += self.ConvBlock7.flops(H / 4, W / 4)
        flops += H / 2 * W / 2 * self.dim * 4 * self.dim * 2 * 2 * 2
        flops += self.ConvBlock8.flops(H / 2, W / 2)
        flops += H * W * self.dim * 2 * self.dim * 2 * 2
        flops += self.ConvBlock9.flops(H, W)
        flops += H * W * self.dim * 3 * 3 * 3
        return flops


class PosCNN(nn.Layer):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2D(in_chans,
                      embed_dim,
                      3,
                      s,
                      1,
                      groups=embed_dim,
                      bias_attr=True))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return [('proj.%d.weight' % i) for i in range(4)]


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1D(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias_attr=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias_attr=False),
            nn.Sigmoid())

    def forward(self, x):
        x = paddle.transpose(x, (0, 2, 1))
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = paddle.transpose(x, (0, 2, 1))
        return x


class SepConv2d(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = paddle.nn.Conv2D(in_channels,
                                          in_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=in_channels)
        self.pointwise = paddle.nn.Conv2D(in_channels,
                                          out_channels,
                                          kernel_size=1)
        self.act_layer = act_layer(
        ) if act_layer is not None else torch2paddle.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += (H * W * self.in_channels * self.kernel_size**2 /
                  self.stride**2)
        flops += H * W * self.in_channels * self.out_channels
        return flops


class ConvProjection(nn.Layer):
    def __init__(self,
                 dim,
                 heads=8,
                 dim_head=64,
                 kernel_size=3,
                 q_stride=1,
                 k_stride=1,
                 v_stride=1,
                 dropout=0.0,
                 last_stage=False,
                 bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))
        attn_kv = x if attn_kv is None else attn_kv
        # x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        x = x.reshape(b, l, w, c).permute(0, 3, 1, 2)
        # attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = attn_kv.reshape(b, l, w, c).permute(0, 3, 1, 2)

        q = self.to_q(x)
        # q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)
        q = q.reshape(b, h, -1, l, w).flatten(-1).permute(0, 1, 3, 2)
        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        # k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        k = k.reshape(b, h, -1, l, w).flatten(-1).permute(0, 1, 3, 2)
        # v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        v = v.shape(b, h, -1, l, w).faltte(-1).permute(0, 1, 3, 2)
        return q, k, v

    def flops(self, H, W):
        flops = 0
        flops += self.to_q.flops(H, W)
        flops += self.to_k.flops(H, W)
        flops += self.to_v.flops(H, W)
        return flops


class LinearProjection(nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias_attr=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias_attr=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads,
                                 C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads,
                                         C // self.heads).permute(
                                             2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, H, W):
        flops = H * W * self.dim * self.inner_dim * 3
        return flops


class LinearProjection_Concat_kv(nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias_attr=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias_attr=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads,
                                         C // self.heads).permute(
                                             2, 0, 3, 1, 4)
        kv_enc = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads,
                                             C // self.heads).permute(
                                                 2, 0, 3, 1, 4)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]
        k_e, v_e = kv_enc[0], kv_enc[1]
        k = paddle.concat((k_d, k_e), dim=2)
        v = paddle.concat((v_d, v_e), dim=2)
        return q, k, v

    def flops(self, H, W):
        flops = H * W * self.dim * self.inner_dim * 5
        return flops


class WindowAttention(nn.Layer):
    def __init__(self,
                 dim,
                 win_size,
                 num_heads,
                 token_projection='linear',
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 se_layer=False):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.relative_position_bias_table = paddle.create_parameter(shape=\
            paddle.zeros([(2 * win_size[0] - 1) * (2 * win_size[1] - 1),
            num_heads]).requires_grad_(False).shape, dtype=str(paddle.zeros
            ([(2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads]).
            requires_grad_(False).numpy().dtype), default_initializer=\
            paddle.nn.initializer.Assign(paddle.zeros([(2 * win_size[0] - 1
            ) * (2 * win_size[1] - 1), num_heads]).requires_grad_(False)))
        self.relative_position_bias_table.stop_gradient = False
        coords_h = paddle.arange(self.win_size[0]).requires_grad_(False)
        coords_w = paddle.arange(self.win_size[1]).requires_grad_(False)
        coords = paddle.stack(paddle.meshgrid([coords_h,
                                               coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:,
                                                                      None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.win_size[0] - 1
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index',
                             relative_position_index)
        if token_projection == 'conv':
            self.qkv = ConvProjection(dim,
                                      num_heads,
                                      dim // num_heads,
                                      bias=qkv_bias)
        elif token_projection == 'linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim,
                                                  num_heads,
                                                  dim // num_heads,
                                                  bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim,
                                        num_heads,
                                        dim // num_heads,
                                        bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = SELayer(dim) if se_layer else torch2paddle.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
        # trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.win_size[0] * self.win_size[1],
                self.win_size[0] * self.win_size[1], -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias,'nH l c -> nH l (c d)', d=ratio)
        relative_position_bias = relative_position_bias.repeat(1, 1, 1, ratio)
        attn = attn + relative_position_bias
        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            mask = mask.repeat(1, 1, 1, ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N * ratio) + mask.unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'
        )

    def flops(self, H, W):
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        flops += self.qkv.flops(H, W)
        if self.token_projection != 'linear_concat':
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
            flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        else:
            flops += nW * self.num_heads * N * (self.dim //
                                                self.num_heads) * N * 2
            flops += nW * self.num_heads * N * N * 2 * (self.dim //
                                                        self.num_heads)
        flops += nW * N * self.dim * self.dim
        print('W-MSA:{%.2f}' % (flops / 1000000000.0))
        return flops


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.in_features * self.hidden_features
        flops += H * W * self.hidden_features * self.out_features
        print('MLP:{%.2f}' % (flops / 1000000000.0))
        return flops


class LeFF(nn.Layer):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2D(hidden_dim,
                      hidden_dim,
                      groups=hidden_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)  # x: 32, 16384,128,128
        # x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x = x.reshape(-1, hh, hh, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.dwconv(x)  # 32, 128,128, 128
        # x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = self.linear2(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.dim * self.hidden_dim
        flops += H * W * self.hidden_dim * 3 * 3
        flops += H * W * self.hidden_dim * self.dim
        print('LeFF:{%.2f}' % (flops / 1000000000.0))
        return flops


def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x,
                     kernel_size=win_size,
                     dilation=dilation_rate,
                     padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size,
                                                       win_size)
        windows = windows.permute(0, 2, 3, 1).contiguous()
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4,
                            5).contiguous().view(-1, win_size, win_size, C)
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()
        x = F.fold(x, (H, W),
                   kernel_size=win_size,
                   dilation=dilation_rate,
                   padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Downsample(nn.Layer):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(in_channel,
                      out_channel,
                      kernel_size=4,
                      stride=2,
                      padding=1))
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()
        return out

    def flops(self, H, W):
        flops = 0
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print('Downsample:{%.2f}' % (flops / 1000000000.0))
        return flops


class Upsample(nn.Layer):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.Conv2DTranspose(in_channel,
                               out_channel,
                               kernel_size=2,
                               stride=2))
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()
        return out

    def flops(self, H, W):
        flops = 0
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print('Upsample:{%.2f}' % (flops / 1000000000.0))
        return flops


class InputProj(nn.Layer):

    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=\
        1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2D(in_channel,
                      out_channel,
                      kernel_size=3,
                      stride=stride,
                      padding=kernel_size // 2), act_layer())
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.in_channel * self.out_channel * 3 * 3
        if self.norm is not None:
            flops += H * W * self.out_channel
        print('Input_proj:{%.2f}' % (flops / 1000000000.0))
        return flops


class OutputProj(nn.Layer):

    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=\
        1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2D(in_channel,
                      out_channel,
                      kernel_size=3,
                      stride=stride,
                      padding=kernel_size // 2))
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H * W * self.in_channel * self.out_channel * 3 * 3
        if self.norm is not None:
            flops += H * W * self.out_channel
        print('Output_proj:{%.2f}' % (flops / 1000000000.0))
        return flops


class LeWinTransformerBlock(nn.Layer):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 win_size=8,
                 shift_size=0,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 token_projection='linear',
                 token_mlp='leff',
                 se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, 'shift_size must in 0-win_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, win_size=to_2tuple(self.win_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, token_projection=\
            token_projection, se_layer=se_layer)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else torch2paddle.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(
                           dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'
        )

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)
            attn_mask = input_mask_windows.view(-1,
                                                self.win_size * self.win_size)
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)
            attn_mask = masked_fill(attn_mask, attn_mask != 0, float(-100.0))
            attn_mask = masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        if self.shift_size > 0:
            shift_mask = paddle.zeros(
                (1, H, W, 1)).requires_grad_(False).astype(x.dtype)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)
            shift_mask_windows = shift_mask_windows.view(
                -1, self.win_size * self.win_size)
            shift_attn_mask = shift_mask_windows.unsqueeze(
                1) - shift_mask_windows.unsqueeze(2)
            shift_attn_mask = masked_fill(shift_attn_mask,
                                          shift_attn_mask != 0, float(-100.0))
            shift_attn_mask = masked_fill(shift_attn_mask,
                                          shift_attn_mask == 0, float(0.0))
            attn_mask = (attn_mask + shift_attn_mask
                         if attn_mask is not None else shift_attn_mask)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = paddle.roll(x,
                                    shifts=(-self.shift_size,
                                            -self.shift_size),
                                    axis=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)
        if self.shift_size > 0:
            x = paddle.roll(shifted_x,
                            shifts=(self.shift_size, self.shift_size),
                            axis=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        flops += self.attn.flops(H, W)
        flops += self.dim * H * W
        flops += self.mlp.flops(H, W)
        print('LeWin:{%.2f}' % (flops / 1000000000.0))
        return flops


class LeWinTransformer_Cross(nn.Layer):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 win_size=8,
                 shift_size=0,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 token_projection='linear',
                 token_mlp='ffn'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, 'shift_size must in 0-win_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, win_size=to_2tuple(self.win_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, token_projection=\
            token_projection)
        self.norm2 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.cross_attn = WindowAttention(dim, win_size=to_2tuple(self.
            win_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=\
            qk_scale, attn_drop=attn_drop, proj_drop=drop, token_projection
            =token_projection)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else torch2paddle.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(
                           dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'
        )

    def forward(self, x, attn_kv=None, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)
            attn_mask = input_mask_windows.view(-1,
                                                self.win_size * self.win_size)
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)
            attn_mask = masked_fill(attn_mask, attn_mask != 0, float(-100.0))
            attn_mask = masked_fill(attn_mask, attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        if self.shift_size > 0:
            shift_mask = paddle.zeros(
                (1, H, W, 1)).requires_grad_(False).astype(x.dtype)
            h_slices = slice(0,
                             -self.win_size), slice(-self.win_size,
                                                    -self.shift_size), slice(
                                                        -self.shift_size, None)
            w_slices = slice(0,
                             -self.win_size), slice(-self.win_size,
                                                    -self.shift_size), slice(
                                                        -self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)
            shift_mask_windows = shift_mask_windows.view(
                -1, self.win_size * self.win_size)
            shift_attn_mask = shift_mask_windows.unsqueeze(
                1) - shift_mask_windows.unsqueeze(2)
            shift_attn_mask = masked_fill(shift_attn_mask,
                                          shift_attn_mask != 0, float(-100.0))
            shift_attn_mask = masked_fill(shift_attn_mask,
                                          shift_attn_mask == 0, float(0.0))
            attn_mask = (attn_mask + shift_attn_mask
                         if attn_mask is not None else shift_attn_mask)
        attn_kv = attn_kv.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_kv = paddle.roll(attn_kv,
                                     shifts=(-self.shift_size,
                                             -self.shift_size),
                                     axis=(1, 2))
        else:
            shifted_kv = attn_kv
        attn_kv_windows = window_partition(shifted_kv, self.win_size)
        attn_kv_windows = attn_kv_windows.view(-1,
                                               self.win_size * self.win_size,
                                               C)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = paddle.roll(x,
                                    shifts=(-self.shift_size,
                                            -self.shift_size),
                                    axis=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)
        shortcut1 = x_windows
        x_windows = self.norm1(x_windows)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        x_windows = shortcut1 + self.drop_path(attn_windows)
        shortcut2 = x_windows
        x_windows = self.norm2(x_windows)
        attn_kv_windows = self.norm_kv(attn_kv_windows)
        attn_windows = self.cross_attn(x_windows,
                                       attn_kv=attn_kv_windows,
                                       mask=attn_mask)
        attn_windows = shortcut2 + self.drop_path(attn_windows)
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)
        if self.shift_size > 0:
            x = paddle.roll(shifted_x,
                            shifts=(self.shift_size, self.shift_size),
                            axis=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        del attn_mask
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        flops += self.attn.flops(H, W)
        flops += self.cross_attn.flops(H, W)
        flops += self.dim * H * W
        flops += self.mlp.flops(H, W)
        print('LeWin:{%.2f}' % (flops / 1000000000.0))
        return flops


class LeWinTransformer_CatCross(nn.Layer):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 win_size=8,
                 shift_size=0,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 token_projection='linear',
                 token_mlp='ffn'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, 'shift_size must in 0-win_size'
        self.norm1 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.cross_attn = WindowAttention(dim, win_size=to_2tuple(self.
            win_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=\
            qk_scale, attn_drop=attn_drop, proj_drop=drop, token_projection
            ='linear_concat')
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else torch2paddle.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop) if token_mlp == 'ffn' else LeFF(
                           dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}'
        )

    def forward(self, x, attn_kv=None, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)
            attn_mask = input_mask_windows.view(-1,
                                                self.win_size * self.win_size)
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)
            attn_mask = masked_fill(attn_mask, attn_mask != 0, float(-100.0))
            attn_mask = masked_fill(attn_mask, attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        if self.shift_size > 0:
            shift_mask = paddle.zeros(
                (1, H, W, 1)).requires_grad_(False).type_as(x)
            h_slices = slice(0,
                             -self.win_size), slice(-self.win_size,
                                                    -self.shift_size), slice(
                                                        -self.shift_size, None)
            w_slices = slice(0,
                             -self.win_size), slice(-self.win_size,
                                                    -self.shift_size), slice(
                                                        -self.shift_size, None)
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)
            shift_mask_windows = shift_mask_windows.view(
                -1, self.win_size * self.win_size)
            shift_attn_mask = shift_mask_windows.unsqueeze(
                1) - shift_mask_windows.unsqueeze(2)
            shift_attn_mask = masked_fill(shift_attn_mask,
                                          shift_attn_mask != 0, float(-100.0))
            shift_attn_mask = masked_fill(shift_attn_mask,
                                          shift_attn_mask == 0, float(0.0))
            attn_mask = (attn_mask + shift_attn_mask
                         if attn_mask is not None else shift_attn_mask)
        attn_kv = attn_kv.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_kv = paddle.roll(attn_kv,
                                     shifts=(-self.shift_size,
                                             -self.shift_size),
                                     axis=(1, 2))
        else:
            shifted_kv = attn_kv
        attn_kv_windows = window_partition(shifted_kv, self.win_size)
        attn_kv_windows = attn_kv_windows.view(-1,
                                               self.win_size * self.win_size,
                                               C)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = paddle.roll(x,
                                    shifts=(-self.shift_size,
                                            -self.shift_size),
                                    axis=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.win_size)
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)
        shortcut1 = x_windows
        x_windows = self.norm1(x_windows)
        attn_kv_windows = self.norm_kv(attn_kv_windows)
        attn_windows = self.cross_attn(x_windows,
                                       attn_kv=attn_kv_windows,
                                       mask=attn_mask)
        attn_windows = shortcut1 + self.drop_path(attn_windows)
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)
        if self.shift_size > 0:
            x = paddle.roll(shifted_x,
                            shifts=(self.shift_size, self.shift_size),
                            axis=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        flops += self.cross_attn.flops(H, W)
        flops += self.dim * H * W
        flops += self.mlp.flops(H, W)
        print('LeWin:{%.2f}' % (flops / 1000000000.0))
        return flops


class BasicUformerLayer(nn.Layer):
    def __init__(self,
                 dim,
                 output_dim,
                 input_resolution,
                 depth,
                 num_heads,
                 win_size,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 token_projection='linear',
                 token_mlp='ffn',
                 se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.LayerList([LeWinTransformerBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads,win_size=win_size, shift_size=0 if i % 2 == 0 else win_size //
            2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if
            isinstance(drop_path, list) else drop_path, norm_layer=\
            norm_layer, token_projection=token_projection, token_mlp=\
            token_mlp, se_layer=se_layer) for i in range(depth)])

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'
        )

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                pass
            else:
                x = blk(x, mask)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class CrossUformerLayer(nn.Layer):
    def __init__(self,
                 dim,
                 output_dim,
                 input_resolution,
                 depth,
                 num_heads,
                 win_size,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 token_projection='linear',
                 token_mlp='ffn'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.LayerList([LeWinTransformer_Cross(dim=dim,
            input_resolution=input_resolution, num_heads=num_heads,
            win_size=win_size, shift_size=0 if i % 2 == 0 else win_size //
            2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if
            isinstance(drop_path, list) else drop_path, norm_layer=\
            norm_layer, token_projection=token_projection, token_mlp=\
            token_mlp) for i in range(depth)])

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'
        )

    def forward(self, x, attn_kv=None, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                pass
            else:
                x = blk(x, attn_kv, mask)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class CatCrossUformerLayer(nn.Layer):
    def __init__(self,
                 dim,
                 output_dim,
                 input_resolution,
                 depth,
                 num_heads,
                 win_size,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 token_projection='linear',
                 token_mlp='ffn'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.LayerList([LeWinTransformer_CatCross(dim=dim,
            input_resolution=input_resolution, num_heads=num_heads,
            win_size=win_size, shift_size=0 if i % 2 == 0 else win_size //
            2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if
            isinstance(drop_path, list) else drop_path, norm_layer=\
            norm_layer, token_projection=token_projection, token_mlp=\
            token_mlp) for i in range(depth)])

    def extra_repr(self) -> str:
        return (
            f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'
        )

    def forward(self, x, attn_kv=None, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                pass
            else:
                x = blk(x, attn_kv, mask)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class Uformer(nn.Layer):

    def __init__(self, img_size=128, in_chans=3, embed_dim=32, depths=[2, 2,
        2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
        win_size=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=\
        0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.
        LayerNorm, patch_norm=True, use_checkpoint=False, token_projection=\
        'linear', token_mlp='ffn', se_layer=False, dowsample=Downsample,
        upsample=Upsample, **kwargs):
        super().__init__()
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        enc_dpr = [
            x.item() for x in paddle.linspace(
                0, stop=drop_path_rate, num=sum(
                    depths[:self.num_enc_layers])).requires_grad_(False)
        ]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]
        self.input_proj = InputProj(in_channel=in_chans, out_channel=\
            embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim,
                                      out_channel=in_chans,
                                      kernel_size=3,
                                      stride=1)
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim, output_dim=\
            embed_dim, input_resolution=(img_size, img_size), depth=depths[
            0], num_heads=num_heads[0], win_size=win_size, mlp_ratio=self.
            mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=enc_dpr[sum(depths[:0]):sum
            (depths[:1])], norm_layer=norm_layer, use_checkpoint=\
            use_checkpoint, token_projection=token_projection, token_mlp=\
            token_mlp, se_layer=se_layer)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
            output_dim=embed_dim * 2, input_resolution=(img_size // 2,
            img_size // 2), depth=depths[1], num_heads=num_heads[1],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])], norm_layer=\
            norm_layer, use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
            output_dim=embed_dim * 4, input_resolution=(img_size // 2 ** 2,
            img_size // 2 ** 2), depth=depths[2], num_heads=num_heads[2],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])], norm_layer=\
            norm_layer, use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
            output_dim=embed_dim * 8, input_resolution=(img_size // 2 ** 3,
            img_size // 2 ** 3), depth=depths[3], num_heads=num_heads[3],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])], norm_layer=\
            norm_layer, use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)
        self.conv = BasicUformerLayer(dim=embed_dim * 16, output_dim=\
            embed_dim * 16, input_resolution=(img_size // 2 ** 4, img_size //
            2 ** 4), depth=depths[4], num_heads=num_heads[4], win_size=\
            win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale
            =qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=\
            conv_dpr, norm_layer=norm_layer, use_checkpoint=use_checkpoint,
            token_projection=token_projection, token_mlp=token_mlp,
            se_layer=se_layer)
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
            output_dim=embed_dim * 16, input_resolution=(img_size // 2 ** 3,
            img_size // 2 ** 3), depth=depths[5], num_heads=num_heads[5],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dec_dpr[:depths[5]], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(
            dim=embed_dim * 8,
            output_dim=embed_dim * 8,
            input_resolution=(img_size // 2**2, img_size // 2**2),
            depth=depths[6],
            num_heads=num_heads[6],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(
            dim=embed_dim * 4,
            output_dim=embed_dim * 4,
            input_resolution=(img_size // 2, img_size // 2),
            depth=depths[7],
            num_heads=num_heads[7],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp,
            se_layer=se_layer)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim * 2,
            output_dim=embed_dim * 2, input_resolution=(img_size, img_size),
            depth=depths[8], num_heads=num_heads[8], win_size=win_size,
            mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dec_dpr[sum
            (depths[5:8]):sum(depths[5:9])], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                m.bias.set_value(np.zeros(shape=m.bias.shape, dtype='float32'))
        elif isinstance(m, paddle.nn.LayerNorm):
            m.bias.set_value(np.zeros(shape=m.bias.shape, dtype='float32'))
            m.weight.set_value(np.ones(shape=m.weight.shape, dtype='float32'))

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}'
        )

    def forward(self, x, mask=None):
        y = self.input_proj(x)
        y = self.pos_drop(y)
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.dowsample_3(conv3)
        conv4 = self.conv(pool3, mask=mask)
        up0 = self.upsample_0(conv4)
        deconv0 = paddle.concat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)
        up1 = self.upsample_1(deconv0)
        deconv1 = paddle.concat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)
        up2 = self.upsample_2(deconv1)
        deconv2 = paddle.concat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)
        up3 = self.upsample_3(deconv2)
        deconv3 = paddle.concat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)
        y = self.output_proj(deconv3)
        return x + y

    def flops(self):
        flops = 0
        flops += self.input_proj.flops(self.reso, self.reso)
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(
            self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(
            self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(
            self.reso // 2**2, self.reso // 2**2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(
            self.reso // 2**3, self.reso // 2**3)
        flops += self.conv.flops()
        flops += self.upsample_0.flops(self.reso // 2**4, self.reso //
                                       2**4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2**3, self.reso //
                                       2**3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2**2, self.reso //
                                       2**2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(
            self.reso // 2, self.reso // 2) + self.decoderlayer_3.flops()
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops


class Uformer_Cross(nn.Layer):

    def __init__(self, img_size=128, in_chans=3, embed_dim=32, depths=[2, 2,
        2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 8, 4, 2, 1],
        win_size=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=\
        0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.
        LayerNorm, patch_norm=True, use_checkpoint=False, token_projection=\
        'linear', token_mlp='ffn', dowsample=Downsample, upsample=Upsample,
        **kwargs):
        super().__init__()
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        enc_dpr = [
            x.item() for x in paddle.linspace(
                0, stop=drop_path_rate, num=sum(
                    depths[:self.num_enc_layers])).requires_grad_(False)
        ]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]
        self.input_proj = InputProj(in_channel=in_chans, out_channel=\
            embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=\
            in_chans, kernel_size=3, stride=1)
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim, output_dim=\
            embed_dim, input_resolution=(img_size, img_size), depth=depths[
            0], num_heads=num_heads[0], win_size=win_size, mlp_ratio=self.
            mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=enc_dpr[sum(depths[:0]):sum
            (depths[:1])], norm_layer=norm_layer, use_checkpoint=\
            use_checkpoint, token_projection=token_projection, token_mlp=\
            token_mlp)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
            output_dim=embed_dim * 2, input_resolution=(img_size // 2,
            img_size // 2), depth=depths[1], num_heads=num_heads[1],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])], norm_layer=\
            norm_layer, use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
            output_dim=embed_dim * 4, input_resolution=(img_size // 2 ** 2,
            img_size // 2 ** 2), depth=depths[2], num_heads=num_heads[2],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])], norm_layer=\
            norm_layer, use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
            output_dim=embed_dim * 8, input_resolution=(img_size // 2 ** 3,
            img_size // 2 ** 3), depth=depths[3], num_heads=num_heads[3],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])], norm_layer=\
            norm_layer, use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)
        self.conv = BasicUformerLayer(dim=embed_dim * 16, output_dim=\
            embed_dim * 16, input_resolution=(img_size // 2 ** 4, img_size //
            2 ** 4), depth=depths[4], num_heads=num_heads[4], win_size=\
            win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale
            =qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=\
            conv_dpr, norm_layer=norm_layer, use_checkpoint=use_checkpoint,
            token_projection=token_projection, token_mlp=token_mlp)
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = CrossUformerLayer(dim=embed_dim * 8,
            output_dim=embed_dim * 8, input_resolution=(img_size // 2 ** 3,
            img_size // 2 ** 3), depth=depths[5], num_heads=num_heads[5],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dec_dpr[:depths[5]], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp)
        self.upsample_1 = upsample(embed_dim * 8, embed_dim * 4)
        self.decoderlayer_1 = CrossUformerLayer(
            dim=embed_dim * 4,
            output_dim=embed_dim * 4,
            input_resolution=(img_size // 2**2, img_size // 2**2),
            depth=depths[6],
            num_heads=num_heads[6],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp)
        self.upsample_2 = upsample(embed_dim * 4, embed_dim * 2)
        self.decoderlayer_2 = CrossUformerLayer(
            dim=embed_dim * 2,
            output_dim=embed_dim * 2,
            input_resolution=(img_size // 2, img_size // 2),
            depth=depths[7],
            num_heads=num_heads[7],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp)
        self.upsample_3 = upsample(embed_dim * 2, embed_dim)
        self.decoderlayer_3 = CrossUformerLayer(dim=embed_dim, output_dim=\
            embed_dim, input_resolution=(img_size, img_size), depth=depths[
            8], num_heads=num_heads[8], win_size=win_size, mlp_ratio=self.
            mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dec_dpr[sum(depths[5:8]):
            sum(depths[5:9])], norm_layer=norm_layer, use_checkpoint=\
            use_checkpoint, token_projection=token_projection, token_mlp=\
            token_mlp)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                m.bias.set_value(np.zeros(shape=m.bias.shape, dtype='float32'))
        elif isinstance(m, paddle.nn.LayerNorm):
            m.bias.set_value(np.zeros(shape=m.bias.shape, dtype='float32'))
            m.weight.set_value(np.ones(shape=m.weight.shape, dtype='float32'))

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}'
        )

    def forward(self, x, mask=None):
        y = self.input_proj(x)
        y = self.pos_drop(y)
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.dowsample_3(conv3)
        conv4 = self.conv(pool3, mask=mask)
        up0 = self.upsample_0(conv4)
        deconv0 = self.decoderlayer_0(up0, attn_kv=conv3, mask=mask)
        up1 = self.upsample_1(deconv0)
        deconv1 = self.decoderlayer_1(up1, attn_kv=conv2, mask=mask)
        up2 = self.upsample_2(deconv1)
        deconv2 = self.decoderlayer_2(up2, attn_kv=conv1, mask=mask)
        up3 = self.upsample_3(deconv2)
        deconv3 = self.decoderlayer_3(up3, attn_kv=conv0, mask=mask)
        y = self.output_proj(deconv3)
        return x + y

    def flops(self):
        flops = 0
        flops += self.input_proj.flops(self.reso, self.reso)
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(
            self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(
            self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(
            self.reso // 2**2, self.reso // 2**2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(
            self.reso // 2**3, self.reso // 2**3)
        flops += self.conv.flops()
        flops += self.upsample_0.flops(self.reso // 2**4, self.reso //
                                       2**4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2**3, self.reso //
                                       2**3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2**2, self.reso //
                                       2**2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(
            self.reso // 2, self.reso // 2) + self.decoderlayer_3.flops()
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops


class Uformer_CatCross(nn.Layer):

    def __init__(self, img_size=128, in_chans=3, embed_dim=32, depths=[2, 2,
        2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 8, 4, 2, 1],
        win_size=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=\
        0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.
        LayerNorm, patch_norm=True, use_checkpoint=False, token_projection=\
        'linear', token_mlp='ffn', dowsample=Downsample, upsample=Upsample,
        **kwargs):
        super().__init__()
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        enc_dpr = [
            x.item() for x in paddle.linspace(
                0, stop=drop_path_rate, num=sum(
                    depths[:self.num_enc_layers])).requires_grad_(False)
        ]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]
        self.input_proj = InputProj(in_channel=in_chans, out_channel=\
            embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=\
            in_chans, kernel_size=3, stride=1)
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim, output_dim=\
            embed_dim, input_resolution=(img_size, img_size), depth=depths[
            0], num_heads=num_heads[0], win_size=win_size, mlp_ratio=self.
            mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=enc_dpr[sum(depths[:0]):sum
            (depths[:1])], norm_layer=norm_layer, use_checkpoint=\
            use_checkpoint, token_projection=token_projection, token_mlp=\
            token_mlp)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
            output_dim=embed_dim * 2, input_resolution=(img_size // 2,
            img_size // 2), depth=depths[1], num_heads=num_heads[1],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])], norm_layer=\
            norm_layer, use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
            output_dim=embed_dim * 4, input_resolution=(img_size // 2 ** 2,
            img_size // 2 ** 2), depth=depths[2], num_heads=num_heads[2],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])], norm_layer=\
            norm_layer, use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
            output_dim=embed_dim * 8, input_resolution=(img_size // 2 ** 3,
            img_size // 2 ** 3), depth=depths[3], num_heads=num_heads[3],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])], norm_layer=\
            norm_layer, use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)
        self.conv = BasicUformerLayer(dim=embed_dim * 16, output_dim=\
            embed_dim * 16, input_resolution=(img_size // 2 ** 4, img_size //
            2 ** 4), depth=depths[4], num_heads=num_heads[4], win_size=\
            win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale
            =qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=\
            conv_dpr, norm_layer=norm_layer, use_checkpoint=use_checkpoint,
            token_projection=token_projection, token_mlp=token_mlp)
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = CatCrossUformerLayer(dim=embed_dim * 8,
            output_dim=embed_dim * 8, input_resolution=(img_size // 2 ** 3,
            img_size // 2 ** 3), depth=depths[5], num_heads=num_heads[5],
            win_size=win_size, mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dec_dpr[:depths[5]], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp)
        self.upsample_1 = upsample(embed_dim * 8, embed_dim * 4)
        self.decoderlayer_1 = CatCrossUformerLayer(
            dim=embed_dim * 4,
            output_dim=embed_dim * 4,
            input_resolution=(img_size // 2**2, img_size // 2**2),
            depth=depths[6],
            num_heads=num_heads[6],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp)
        self.upsample_2 = upsample(embed_dim * 4, embed_dim * 2)
        self.decoderlayer_2 = CatCrossUformerLayer(
            dim=embed_dim * 2,
            output_dim=embed_dim * 2,
            input_resolution=(img_size // 2, img_size // 2),
            depth=depths[7],
            num_heads=num_heads[7],
            win_size=win_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            token_projection=token_projection,
            token_mlp=token_mlp)
        self.upsample_3 = upsample(embed_dim * 2, embed_dim)
        self.decoderlayer_3 = CatCrossUformerLayer(dim=embed_dim,
            output_dim=embed_dim, input_resolution=(img_size, img_size),
            depth=depths[8], num_heads=num_heads[8], win_size=win_size,
            mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dec_dpr[sum
            (depths[5:8]):sum(depths[5:9])], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                m.bias.set_value(np.zeros(shape=m.bias.shape, dtype='float32'))
        elif isinstance(m, paddle.nn.LayerNorm):
            m.bias.set_value(np.zeros(shape=m.bias.shape, dtype='float32'))
            m.weight.set_value(np.ones(shape=m.weight.shape, dtype='float32'))

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}'
        )

    def forward(self, x, mask=None):
        y = self.input_proj(x)
        y = self.pos_drop(y)
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.dowsample_3(conv3)
        conv4 = self.conv(pool3, mask=mask)
        up0 = self.upsample_0(conv4)
        deconv0 = self.decoderlayer_0(up0, attn_kv=conv3, mask=mask)
        up1 = self.upsample_1(deconv0)
        deconv1 = self.decoderlayer_1(up1, attn_kv=conv2, mask=mask)
        up2 = self.upsample_2(deconv1)
        deconv2 = self.decoderlayer_2(up2, attn_kv=conv1, mask=mask)
        up3 = self.upsample_3(deconv2)
        deconv3 = self.decoderlayer_3(up3, attn_kv=conv0, mask=mask)
        y = self.output_proj(deconv3)
        return x + y

    def flops(self):
        flops = 0
        flops += self.input_proj.flops(self.reso, self.reso)
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(
            self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(
            self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(
            self.reso // 2**2, self.reso // 2**2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(
            self.reso // 2**3, self.reso // 2**3)
        flops += self.conv.flops()
        flops += self.upsample_0.flops(self.reso // 2**4, self.reso //
                                       2**4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2**3, self.reso //
                                       2**3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2**2, self.reso //
                                       2**2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(
            self.reso // 2, self.reso // 2) + self.decoderlayer_3.flops()
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops


class Uformer_singlescale(nn.Layer):

    def __init__(self, img_size=128, in_chans=3, embed_dim=32, depths=[2, 2,
        2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
        win_size=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=\
        0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.
        LayerNorm, patch_norm=True, use_checkpoint=False, token_projection=\
        'linear', token_mlp='ffn', se_layer=False, downsample=Downsample,
        upsample=Upsample, **kwargs):
        super().__init__()
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        enc_dpr = [
            x.item() for x in paddle.linspace(
                0, stop=drop_path_rate, num=sum(
                    depths[:self.num_enc_layers])).requires_grad_(False)
        ]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]
        self.input_proj = InputProj(in_channel=in_chans, out_channel=\
            embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim,
                                      out_channel=in_chans,
                                      kernel_size=3,
                                      stride=1)
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim, output_dim=\
            embed_dim, input_resolution=(img_size, img_size), depth=depths[
            0], num_heads=num_heads[0], win_size=win_size, mlp_ratio=self.
            mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=enc_dpr[sum(depths[:0]):sum
            (depths[:1])], norm_layer=norm_layer, use_checkpoint=\
            use_checkpoint, token_projection=token_projection, token_mlp=\
            token_mlp, se_layer=se_layer)
        self.downsample_0 = downsample(embed_dim,
                                       embed_dim * 2,
                                       downsample=False)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
            output_dim=embed_dim * 2, input_resolution=(img_size, img_size),
            depth=depths[1], num_heads=num_heads[1], win_size=win_size,
            mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=enc_dpr[sum
            (depths[:1]):sum(depths[:2])], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.downsample_1 = downsample(embed_dim * 2,
                                       embed_dim * 4,
                                       downsample=False)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
            output_dim=embed_dim * 4, input_resolution=(img_size, img_size),
            depth=depths[2], num_heads=num_heads[2], win_size=win_size,
            mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=enc_dpr[sum
            (depths[:2]):sum(depths[:3])], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.downsample_2 = downsample(embed_dim * 4,
                                       embed_dim * 8,
                                       downsample=False)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
            output_dim=embed_dim * 8, input_resolution=(img_size, img_size),
            depth=depths[3], num_heads=num_heads[3], win_size=win_size,
            mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=enc_dpr[sum
            (depths[:3]):sum(depths[:4])], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.downsample_3 = downsample(embed_dim * 8,
                                       embed_dim * 16,
                                       downsample=False)
        self.conv = BasicUformerLayer(dim=embed_dim * 16, output_dim=\
            embed_dim * 16, input_resolution=(img_size, img_size), depth=\
            depths[4], num_heads=num_heads[4], win_size=win_size, mlp_ratio
            =self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=\
            drop_rate, attn_drop=attn_drop_rate, drop_path=conv_dpr,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint,
            token_projection=token_projection, token_mlp=token_mlp,
            se_layer=se_layer)
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8, upsample=\
            False)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
            output_dim=embed_dim * 16, input_resolution=(img_size, img_size
            ), depth=depths[5], num_heads=num_heads[5], win_size=win_size,
            mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dec_dpr[:
            depths[5]], norm_layer=norm_layer, use_checkpoint=\
            use_checkpoint, token_projection=token_projection, token_mlp=\
            token_mlp, se_layer=se_layer)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4, upsample=\
            False)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 8,
            output_dim=embed_dim * 8, input_resolution=(img_size, img_size),
            depth=depths[6], num_heads=num_heads[6], win_size=win_size,
            mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dec_dpr[sum
            (depths[5:6]):sum(depths[5:7])], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.upsample_2 = upsample(embed_dim * 8,
                                   embed_dim * 2,
                                   upsample=False)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
            output_dim=embed_dim * 4, input_resolution=(img_size, img_size),
            depth=depths[7], num_heads=num_heads[7], win_size=win_size,
            mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dec_dpr[sum
            (depths[5:7]):sum(depths[5:8])], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim, upsample=False)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim * 2,
            output_dim=embed_dim * 2, input_resolution=(img_size, img_size),
            depth=depths[8], num_heads=num_heads[8], win_size=win_size,
            mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dec_dpr[sum
            (depths[5:8]):sum(depths[5:9])], norm_layer=norm_layer,
            use_checkpoint=use_checkpoint, token_projection=\
            token_projection, token_mlp=token_mlp, se_layer=se_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                m.bias.set_value(np.zeros(shape=m.bias.shape, dtype='float32'))
        elif isinstance(m, paddle.nn.LayerNorm):
            m.bias.set_value(np.zeros(shape=m.bias.shape, dtype='float32'))
            m.weight.set_value(np.ones(shape=m.weight.shape, dtype='float32'))

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return (
            f'embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}'
        )

    def forward(self, x, mask=None):
        y = self.input_proj(x)
        y = self.pos_drop(y)
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.downsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.downsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.downsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.downsample_3(conv3)
        conv4 = self.conv(pool3, mask=mask)
        up0 = self.upsample_0(conv4)
        deconv0 = paddle.concat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)
        up1 = self.upsample_1(deconv0)
        deconv1 = paddle.concat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)
        up2 = self.upsample_2(deconv1)
        deconv2 = paddle.concat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)
        up3 = self.upsample_3(deconv2)
        deconv3 = paddle.concat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)
        y = self.output_proj(deconv3)
        return x + y


if __name__ == '__main__':
    arch = Uformer
    input_size = 256
    depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    model_restoration = arch(img_size=input_size, embed_dim=44, depths=\
        depths, win_size=8, mlp_ratio=4.0, qkv_bias=True, token_projection=\
        'linear', token_mlp='leff', downsample=Downsample, upsample=\
        Upsample, se_layer=False)
    print('number of GFLOPs: %.2f G' %
          (model_restoration.flops() / 1000000000.0))
