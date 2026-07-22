"""
Module defining generic neural netork component in pytorch.
"""

import torch
from torch.nn import Module, ModuleList
from torch.nn import Conv2d, Dropout, Dropout2d, Linear
from torch.nn import ReLU
from torch.nn.functional import pad
import random
from torch.nn import LayerNorm, InstanceNorm2d, MultiheadAttention
from torch import relu, softmax
from torch.nn.init import xavier_uniform_


class PositionalEncoding1DOnTheFly(Module):
    """
    Class defining 1D positional encoding.
    Positional encoding is computed at each call (no limitation is size)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        div = torch.exp(-torch.arange(0., dim, 2, requires_grad=False) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(0)
        self.register_buffer('div', div)


    def forward(self, indices):
        emb_indices = torch.zeros((indices.size(0), self.dim, indices.size(1)), device=indices.device)
        emb_indices[:, ::2, :] = torch.sin(indices.unsqueeze(2) * self.div).permute(0, 2, 1)
        emb_indices[:, 1::2, :] = torch.cos(indices.unsqueeze(2) * self.div).permute(0, 2, 1)
        return emb_indices


class PositionalEncoding1D(Module):
    """
    Class defining 1D positional encoding.
    Positional encoding is computed once and for all (max length must be specified)
    """

    def __init__(self, dim, len_max):
        super().__init__()
        self.len_max = len_max
        self.dim = dim
        pe = torch.zeros((1, dim, len_max), requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x, start):
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            return x + self.pe[:, :, start:start+x.size(2)].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, :, start[i]:start[i]+x.size(2)]
            return x


class PositionalEncoding2D(Module):
    """
    Class defining 1D positional encoding.
    Positional encoding is computed once and for all (max height and width must be specified)
    """

    def __init__(self, dim, h_max, w_max):
        super().__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim
        pe = torch.zeros((1, dim, h_max, w_max), requires_grad=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        w_pos = torch.arange(0., w_max)
        h_pos = torch.arange(0., h_max)
        pe[:, :dim // 2:2, :, :] = torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        pe[:, 1:dim // 2:2, :, :] = torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        pe[:, dim // 2::2, :, :] = torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        pe[:, dim // 2 + 1::2, :, :] = torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)


class CustomMultiHeadAttention(Module):
    """
    Re-implementation of Multi-head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout=0, proj_value=True):
        super().__init__()

        self.proj_value = proj_value

        self.in_proj_q = Linear(embed_dim, embed_dim)
        self.in_proj_k = Linear(embed_dim, embed_dim)
        if self.proj_value:
            self.in_proj_v = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5
        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, output_weights=True):
        target_len, b, c = query.size()
        source_len = key.size(0)
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        v = self.in_proj_v(value) if self.proj_value else value
        q = q * self.scale_factor

        q = torch.reshape(q, (target_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        k = torch.reshape(k, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        v = torch.reshape(v, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = torch.repeat_interleave(attn_mask.unsqueeze(1), self.num_heads, dim=1).reshape(
                    b * self.num_heads, target_len, source_len)

            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(b, self.num_heads, target_len, source_len)

            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(b * self.num_heads, target_len, source_len)

        attn_output_weights_raw = softmax(attn_output_weights, dim=-1)

        attn_output_weights = self.dropout(attn_output_weights_raw)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)

        if output_weights:
            attn_output_weights_raw = attn_output_weights_raw.view(b, self.num_heads, target_len, source_len)
            return attn_output, attn_output_weights_raw.sum(dim=1) / self.num_heads
        return attn_output

    def init_weights(self):
        xavier_uniform_(self.in_proj_q.weight)
        xavier_uniform_(self.in_proj_k.weight)
        if self.proj_value:
            xavier_uniform_(self.in_proj_v.weight)


class GlobalDecoderLayer(Module):
    """
    Transformer Decoder Layer
    """

    def __init__(self, params, last=False):
        super(GlobalDecoderLayer, self).__init__()
        self.emb_dim = params["enc_dim"]
        self.num_heads = params["dec_num_heads"]
        self.dropout_rate = params["dec_dropout"]
        self.dim_feedforward = params["dec_dim_feedforward"]

        self.self_att = MultiheadAttention(embed_dim=self.emb_dim,
                                       num_heads=self.num_heads,
                                       dropout=self.dropout_rate,
                                       bias=True,
                                       add_bias_kv=False,
                                       add_zero_attn=False,
                                       kdim=None,
                                       vdim=None,
                                       batch_first=False)

        self.norm1 = LayerNorm(self.emb_dim)
        self.att = MultiheadAttention(embed_dim=self.emb_dim,
                                       num_heads=self.num_heads,
                                       dropout=self.dropout_rate,
                                       bias=True,
                                       add_bias_kv=False,
                                       add_zero_attn=False,
                                       kdim=None,
                                       vdim=None,
                                       batch_first=False)

        self.linear1 = Linear(self.emb_dim, self.dim_feedforward)
        self.linear2 = Linear(self.dim_feedforward, self.emb_dim)

        self.dropout = Dropout(params["dec_dropout"])

        self.norm2 = LayerNorm(self.emb_dim)
        self.norm3 = LayerNorm(self.emb_dim)
        self.drop_last_res = (last and "dec_drop_last_res" in params and params["dec_drop_last_res"]) or ("dec_drop_all_res" in params and params["dec_drop_all_res"])

    def forward(self, tgt, memory_key, memory_value=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, predict_last_n_only=None):

        if memory_value is None:
            memory_value = memory_key

        self_att_query = tgt[-predict_last_n_only:] if predict_last_n_only else tgt


        tgt2, weights_self = self.self_att(query=self_att_query,
                                 key=tgt,
                                 value=tgt,
                                 attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask,
                                 need_weights=True,
                                 average_attn_weights=True,
                                 is_causal=False)

        tgt = self_att_query + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        att_query = tgt

        tgt2, weights = self.att(query=att_query,
                                 key=memory_key,
                                 value=memory_value,
                                 attn_mask=memory_mask,
                                 key_padding_mask=memory_key_padding_mask,
                                 need_weights=True,
                                 average_attn_weights=True,
                                 is_causal=False)

        if self.drop_last_res and self.training and random.randint(1, 5) == 1: # 20% to drop something
            if random.randint(1, 2) == 1: # 50% to drop tgt
                tgt = tgt2
            # 50% to drop tgt2 (tgt remains as it is)
        else:
            tgt = att_query + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt, weights, weights_self


class GlobalAttDecoder(Module):
    """
    Stack of transformer decoder layers
    """

    def __init__(self, params):
        super(GlobalAttDecoder, self).__init__()

        self.decoder_layers = ModuleList([GlobalDecoderLayer(params, last=i==params["dec_num_layers"]-1) for i in range(params["dec_num_layers"])])

    def forward(self, tgt, memory_key, memory_value, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                use_cache=False, cache=None, predict_last_n_only=False, keep_all_weights=False):
        output = tgt
        cache_t = list()
        all_weights = {
            "self": list(),
            "mix": list()
        }
        for i, dec_layer in enumerate(self.decoder_layers):
            output, weights, weights_self = dec_layer(output, memory_key=memory_key,
                                        memory_value=memory_value,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        predict_last_n_only=predict_last_n_only)
            if use_cache:
                cache_t.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)
            if keep_all_weights:
                all_weights["self"].append(weights_self)
                all_weights["mix"].append(weights)
        if use_cache:
            cache = torch.cat([cache, torch.stack(cache_t, dim=0)], dim=1) if cache is not None else torch.stack(cache_t, dim=0)

        if predict_last_n_only:
            output = output[-predict_last_n_only:]

        if keep_all_weights:
            return output, all_weights, cache

        return output, weights, cache


class DepthSepConv2D(Module):
    """
    Depthwise Separable Convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class MixDropout(Module):
    """
    Layer alternating between standard dropout and 2D dropout.
    """
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)


class FCN_Encoder(Module):
    """
    Encoder defined in Coquenet et al., "End-to-end Handwritten Paragraph Text 
    Recognition Using a Vertical Attention Network", TPAMI 2023.
    """
    def __init__(self, params):
        super(FCN_Encoder, self).__init__()

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), params=params),
            ConvBlock(16, 32, stride=(2, 2), params=params),
            ConvBlock(32, 64, stride=(2, 2), params=params),
            ConvBlock(64, 128, stride=(2, 2), params=params),
            ConvBlock(128, 128, stride=(2, 1), params=params),
            ConvBlock(128, 128, stride=(2, 1), params=params),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, stride=(1, 1), params=params),
            DSCBlock(128, 128, stride=(1, 1), params=params),
            DSCBlock(128, 128, stride=(1, 1), params=params),
            DSCBlock(128, 256, stride=(1, 1), params=params),
        ])

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        return x


class FCN_Encoder_M256(Module):
    def __init__(self, params):
        super().__init__()

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 32, stride=(1, 1), params=params),
            ConvBlock(32, 64, stride=(2, 2), params=params),
            ConvBlock(64, 128, stride=(2, 2), params=params),
            ConvBlock(128, 256, stride=(2, 2), params=params),
            ConvBlock(256, 512, stride=(2, 1), params=params),
            ConvBlock(512, 512, stride=(2, 1), params=params),
        ])
        self.blocks = ModuleList([
            DSCBlock(512, 512, stride=(1, 1), params=params),
            DSCBlock(512, 512, stride=(1, 1), params=params),
            DSCBlock(512, 512, stride=(1, 1), params=params),
            DSCBlock(512, 256, stride=(1, 1), params=params),
        ])

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        return x
    

class ConvBlock(Module):
    """
    Convolution block of the encoder defined in Coquenet et al., "End-to-end Handwritten Paragraph Text 
    Recognition Using a Vertical Attention Network", TPAMI 2023.
    """

    def __init__(self, in_, out_, stride, params):
        super(ConvBlock, self).__init__()
        dropout = params["enc_dropout"]
        self.activation = ReLU()
        self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=3, padding=1)
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)

        return x


class DSCBlock(Module):

    """
    Despthwise Separable Convolution block of the encoder defined in Coquenet et al., "End-to-end Handwritten Paragraph Text 
    Recognition Using a Vertical Attention Network", TPAMI 2023.
    """

    def __init__(self, in_, out_, stride, params):
        super(DSCBlock, self).__init__()

        dropout = params["enc_dropout"]
        self.activation = ReLU()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)

        return x