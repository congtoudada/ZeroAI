import time
from collections.abc import Iterable
from functools import partial
from itertools import repeat
import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from timm.layers import DropPath, trunc_normal_
from torchsummary import summary


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PointwiseConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features, with_cls_token=True):
        super().__init__()
        self.with_cls_token = with_cls_token
        self.net = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_features, in_features, 1),
        )

    def forward(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.net(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.with_cls_token:
            x = torch.cat((cls_token, x), dim=1)
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 height=20,
                 width=40,
                 norm_layer=nn.LayerNorm,
                 official=True,
                 **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim_in)
        if official:
            self.attn = Attention(
                dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop,
                **kwargs
            )
        else:
            # official num_heads=4
            self.attn = AgentAttention(
                dim=dim_in, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                agent_num=49, height=height, width=width, **kwargs
            )
            # self.attn = HiLo(
            #     dim=dim_in, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            #     window_size=2, alpha=0.5
            # )
            # self.attn = P2TAttention(
            #     dim=dim_in, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=None,
            #     attn_drop=attn_drop, proj_drop=drop, pool_ratios=(1, 3, 6)
            # )

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        # self.mlp = Mlp(
        #     in_features=dim_out,
        #     hidden_features=dim_mlp_hidden,
        #     act_layer=act_layer,
        #     drop=drop
        # )
        self.mlp = PointwiseConvMlp(in_features=dim_out, hidden_features=dim_mlp_hidden)

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x), h=h, w=w))

        return x


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding
    """

    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, embed_dim)
            )
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs
            )
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=0.02)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logging.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] == '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    if 'pos_embed' in k and v.size() != model_dict[k].size():
                        size_pretrained = v.size()
                        size_new = model_dict[k].size()
                        logging.info(
                            '=> load_pretrained: resized variant: {} to {}'
                            .format(size_pretrained, size_new)
                        )

                        ntok_new = size_new[1]
                        ntok_new -= 1

                        posemb_tok, posemb_grid = v[:, :1], v[0, 1:]

                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))

                        logging.info(
                            '=> load_pretrained: grid-size from {} to {}'
                            .format(gs_old, gs_new)
                        )

                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = scipy.ndimage.zoom(
                            posemb_grid, zoom, order=1
                        )
                        posemb_grid = posemb_grid.reshape(1, gs_new ** 2, -1)
                        v = torch.tensor(
                            np.concatenate([posemb_tok, posemb_grid], axis=1)
                        )

                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')

        return layers

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h=20, w=40):
        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @staticmethod
    def compute_macs(module, input, output):
        # T: num_token
        # S: num_token
        input = input[0]
        flops = 0

        _, T, C = input.shape
        H = W = int(np.sqrt(T - 1)) if module.with_cls_token else int(np.sqrt(T))

        H_Q = H / module.stride_q
        W_Q = H / module.stride_q
        T_Q = H_Q * W_Q + 1 if module.with_cls_token else H_Q * W_Q

        H_KV = H / module.stride_kv
        W_KV = W / module.stride_kv
        T_KV = H_KV * W_KV + 1 if module.with_cls_token else H_KV * W_KV

        # C = module.dim
        # S = T
        # Scaled-dot-product macs
        # [B x T x C] x [B x C x T] --> [B x T x S]
        # multiplication-addition is counted as 1 because operations can be fused
        flops += T_Q * T_KV * module.dim
        # [B x T x S] x [B x S x C] --> [B x T x C]
        flops += T_Q * module.dim * T_KV

        if (
                hasattr(module, 'conv_proj_q')
                and hasattr(module.conv_proj_q, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_q.conv.parameters()
                ]
            )
            flops += params * H_Q * W_Q

        if (
                hasattr(module, 'conv_proj_k')
                and hasattr(module.conv_proj_k, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_k.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        if (
                hasattr(module, 'conv_proj_v')
                and hasattr(module.conv_proj_v, 'conv')
        ):
            params = sum(
                [
                    p.numel()
                    for p in module.conv_proj_v.conv.parameters()
                ]
            )
            flops += params * H_KV * W_KV

        params = sum([p.numel() for p in module.proj_q.parameters()])
        flops += params * T_Q
        params = sum([p.numel() for p in module.proj_k.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj_v.parameters()])
        flops += params * T_KV
        params = sum([p.numel() for p in module.proj.parameters()])
        flops += params * T

        module.__flops__ += flops


class HiLo(nn.Module):
    """
    HiLo Attention

    Paper: Fast Vision Transformers with HiLo Attention
    Link: https://arxiv.org/abs/2205.13213
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 window_size=2, alpha=0.5, with_cls_token=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)  # 每个注意力头的通道数
        self.dim = dim
        self.with_cls_token = with_cls_token

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)  # 根据alpha来确定分配给低频注意力的注意力头的数量
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim  # 确定低频注意力的通道数

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads  # 总的注意力头个数-低频注意力头的个数==高频注意力头的个数
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim  # 确定高频注意力的通道数, 总通道数-低频注意力通道数==高频注意力通道数

        self.ws = window_size  # 窗口的尺寸, 如果ws==2, 那么这个窗口就包含4个patch(或token)

        # 如果窗口的尺寸等于1,这就相当于标准的自注意力机制了, 不存在窗口注意力了; 因此,也就没有高频的操作了,只剩下低频注意力机制了
        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention (Hi-Fi)
        # 如果高频注意力头的个数大于0, 那就说明存在高频注意力机制
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    # 高频注意力机制
    def hifi(self, x):
        B, H, W, C = x.shape

        # 每行有w_group个窗口, 每列有h_group个窗口;
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1,
                                                                                                              4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
        x = self.h_proj(x)
        return x

    # 低频注意力机制
    def lofi(self, x):
        B, H, W, C = x.shape
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        # 如果窗口尺寸大于1, 在每个窗口执行池化 (如果窗口尺寸等于1,没有池化的必要)
        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x, H=20, W=40):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, H * W], 1)
        B, N, C = x.shape

        assert N == H * W, "N must equal H*W!"
        # H = W = 每一列/行有多少个patch
        # H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C)

        # 如果分配给高频注意力的注意力头的个数为0,那么仅仅执行低频注意力
        if self.h_heads == 0:
            x = self.lofi(x)
            return x.reshape(B, N, C)

        # 如果分配给低频注意力的注意力头的个数为0,那么仅仅执行高频注意力
        if self.l_heads == 0:
            x = self.hifi(x)
            return x.reshape(B, N, C)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, N, C)

        if self.with_cls_token:
            x = torch.cat((cls_token, x), dim=1)
        return x


class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=49, height=20, width=20, method='dw_bn', kernel_size=3, stride_kv=1,
                 stride_q=1, padding_kv=1, padding_q=1, with_cls_token=True, **kwargs):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.with_cls_token = with_cls_token

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.conv_proj_q = self._build_projection(
        #     dim, dim, kernel_size, padding_q,
        #     stride_q, 'linear' if method == 'avg' else method
        # )
        # self.conv_proj_k = self._build_projection(
        #     dim, dim, kernel_size, padding_kv,
        #     stride_kv, method
        # )
        # self.conv_proj_v = self._build_projection(
        #     dim, dim, kernel_size, padding_kv,
        #     stride_kv, method
        # )
        # self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        # self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num

        # 窗口大小,等于第71/72行的h和w, 即h=w=window
        self.height = height
        self.width = width
        # self.window = window

        # 深度卷积
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)

        # 池化尺寸 49**0.5=7   agent_num = pool_size*pool_size
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

        # agent bias in attention_1
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7,
                                                7))  # block bias (agent_num,h0,w0)   h0,w0是比h,w(h=w=window)小得多的预定义超参数, 因为后面要经过插值恢复h,w
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, height, 1))  # 行bias (agent_num,h,1)
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, width))  # 列bias (agent_num,1,w)

        # agent bias in attention_2
        self.na_bias = nn.Parameter(
            torch.zeros(num_heads, agent_num, 7, 7))  # block bias (agent_num,h0,w0)   h0,w0是比h,w小得多的预定义超参数
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, height, 1, agent_num))  # 行bias (h,1,agent_num)
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, width, agent_num))  # 列bias (1,w,agent_num)

        # 生成std=.02的正态分布数据
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        return q, k, v

    def forward(self, x, h=20, w=20):
        """
        Args:
            :param x: input features with shape of (num_windows*B, N, C)
        """
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, self.height * self.width], 1)

        b, n, c = x.shape
        # h = int(n ** 0.5)  # 每列有h个token
        # w = int(n ** 0.5)  # 每行有w个token
        num_heads = self.num_heads  # 注意力头的个数
        head_dim = c // num_heads  # 每个头的通道数

        # kv表将输入x通过线性层生成q示: (b,n,c) --qkv-> (b,n,3c) --reshape-> (b,n,3,c) --permute-> (3,b,n,c)
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        # 将qkv拆分: q:(b,n,c)  k:(b,n,c)  v:(b,n,c)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q, k, v = self.forward_conv(x, self.height, self.width)
        # q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        # k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        # v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        q_t = q.reshape(b, self.height, self.width, c).permute(0, 3, 1, 2)
        agent_tokens = self.pool(q_t).reshape(b, c, -1).permute(0, 2, 1)

        # 分别对q/k/v进行变换
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # 聚合来自K和V的信息 (第一次注意力)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.height, self.width), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # 把信息广播回Q (第二次注意力)
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.height, self.width), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        # 增加深度卷积
        x = x.transpose(1, 2).reshape(b, n, c)
        v_ = v.transpose(1, 2).reshape(b, self.height, self.width, c).permute(0, 3, 1, 2)
        x = x + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.with_cls_token:
            x = torch.cat((cls_token, x), dim=1)
        return x


class P2TAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=(1, 2, 3, 6), with_cls_token=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_cls_token = with_cls_token

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios  # 池化窗口大小
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)
        self.d_convs1 = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp in self.pool_ratios])

    def forward(self, x, H=20, W=40, d_convs=None):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, H * W], 1)

        B, N, C = x.shape

        # 通过输入x生成q矩阵: (B,N,C) --q-> (B,N,C) --reshape-> (B,N,h,d) --permute-> (B,h,N,d);   C=h*d
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        pools = []

        # 为了便于在x上执行多尺度池化操作,我们将其reshape重塑为2D类型: (B,N,C) --permute-> (B,C,N) --reshape-> (B,C,H,W)
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # 遍历多个池化层, 假设池化窗口为: [1 ,2 ,3 ,6]
        for (pool_ratio, l) in zip(self.pool_ratios, self.d_convs1):
            # 分别计算当前池化窗口下的输出: input:(B,C,H,W);  1th_pool: (B,C,H/1,W/1); 2th_pool: (B,C,H/2,W/2); 3th_pool: (B,C,H/3,W/3); 4th_pool: (B,C,H/6,W/6)
            pool = F.adaptive_avg_pool2d(x_, (round(H / pool_ratio), round(W / pool_ratio)))
            # 将每一个尺度对应的池化层的输出, 再通过3*3的深度卷积进行相对位置编码, 然后与池化的输出相加
            pool = pool + l(pool)
            # 将每个尺度的输出重塑为与原始输入相同的shape: 1th_pool: (B,C,H/1,W/1) -->(B,C,(HW/1^2));  2th_pool: (B,C,H/2,W/2) --> (B,C,(HW/2^2));  3th_pool: (B,C,H/3,W/3) --> (B,C,(HW/3^2));   3th_pool: (B,C,H/6,W/6) --> (B,C,(HW/6^2));
            pools.append(pool.view(B, C, -1))

        # 将多个尺度池化层的输出在token维度进行拼接,其具有多尺度的上下文信息: (B,C,(HW/1^2)+(HW/2^2)+(HW/3^2)+(HW/6^2))==(B,C,token_num) , 令token_num = (HW/1^2)+(HW/2^2)+(HW/3^2)+(HW/6^2)
        pools = torch.cat(pools, dim=2)
        # 将其进行维度转换, 以便于后续计算: (B,C,token_num)--permute->(B,token_num,C)
        pools = self.norm(pools.permute(0, 2, 1))

        # 多尺度的上下文信息生成kv: (B,token_num,C) --kv-> (B,token_num,2C) --reshape-> (B,token_num,2,h,d) --permute-> (2,B,h,token_num,d);   C=h*d
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k:(B,h,token_num,d); v:(B,h,token_num,d)
        k, v = kv[0], kv[1]

        # 计算Token-to-Region化的注意力矩阵(region是指池化是在窗口上进行的,窗口可以看作region): (B,h,N,d) @ (B,h,d,token_num) = (B,h,N,token_num)  N:输入的token总数, token_num:池化后的Token总数量
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 通过注意力矩阵对value加权求和: (B,h,N,token_num) @ (B,h,token_num,d) = (B,h,N,d)
        x = (attn @ v)

        # 通过对输入进行重塑shape得到与原始输入相同的shape: (B,h,N,d) --transpose-> (B,N,h,d) --reshape-> (B,N,C)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)
        # 最后通过一个线性层进行映射, 得到最终输出: (B,N,C)-->(B,N,C)
        x = self.proj(x)

        if self.with_cls_token:
            x = torch.cat((cls_token, x), dim=1)
        return x


if __name__ == '__main__':
    device = "cuda"
    X = torch.randn(1, 800, 256).to(device)
    B, N, D = X.size()
    num_heads = 8
    run_cnt = 10000
    H = 20
    W = 40

    # print("**************************************************")
    # Baseline = Attention(dim_in=D, dim_out=D, num_heads=num_heads, with_cls_token=False)
    # Baseline.to(device)
    # out = Baseline(X, H, W)
    # X = torch.randn(1, N, D).to(device)
    # start_time = time.time()
    # for i in range(run_cnt):
    #     out = Baseline(X, H, W)
    # end_time = time.time()
    # print(f"Attention执行时间: {end_time - start_time} 秒")
    # print(out.shape)
    # summary(Baseline, (N, D))
    #
    # print("*********************************************")
    # Model1 = AgentAttention(dim=D, num_heads=num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.,
    #                         agent_num=49, height=H, width=W, with_cls_token=False)
    # Model1.to(device)
    # out = Model1(X)
    # X = torch.randn(1, N, D).to(device)
    # start_time = time.time()
    # for i in range(run_cnt):
    #     out = Model1(X)
    # end_time = time.time()
    # print(f"AgentAttention执行时间: {end_time - start_time} 秒")
    # print(out.shape)
    # summary(Model1, (N, D))

    # print("**************************************************")
    # Model2 = HiLo(dim=D, num_heads=num_heads, window_size=2, alpha=0.5, with_cls_token=False)
    # Model2.to(device)
    # out = Model2(X, H, W)
    # X = torch.randn(1, N, D).to(device)
    # start_time = time.time()
    # for i in range(run_cnt):
    #     out = Model2(X)
    # end_time = time.time()
    # print(f"HiLoAttention执行时间: {end_time - start_time} 秒")
    # print(out.shape)
    # summary(Model2, (N, D))

    # print("*********************************************")
    # Model3 = EfficientAdditiveAttention(in_dims=D, token_dim=D, with_cls_token=False)
    # Model3.to(device)
    # X = X.reshape(B, N, D)
    # out = Model3(X)
    # X = torch.randn(1, N, D).to(device)
    # start_time = time.time()
    # for i in range(run_cnt):
    #     out = Model3(X)
    # end_time = time.time()
    # print(f"EfficientAdditiveAttention执行时间: {end_time - start_time} 秒")
    # print(out.shape)
    # summary(Model3, (N, D))

    print("*********************************************")
    Model4 = P2TAttention(
        dim=256, num_heads=8, qkv_bias=True, qk_scale=None,
        attn_drop=0., proj_drop=0., pool_ratios=(1, 3, 6)).to("cuda")
    Model4.to(device)
    X = X.reshape(B, N, D)
    out = Model4(X)
    X = torch.randn(1, N, D).to(device)
    start_time = time.time()
    for i in range(run_cnt):
        out = Model4(X)
    end_time = time.time()
    print(f"P2TAttention执行时间: {end_time - start_time} 秒")
    print(out.shape)
    summary(Model4, (N, D))
