"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np

from torch import nn, einsum
from einops import rearrange, repeat
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache

import ipdb

@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


### Perceiver Attention
@dataclass
class PerceiverConfig:
    dim: int
    latent_dim: int
    num_latents: int
    depth: int
    
    cross_heads: int
    cross_dim_head: int
    latent_heads: int
    latent_dim_head: int
    attn_dropout: float
    ff_dropout: float


def get_sinusoid_encoding_table(n_position, d_hid):
    """ Sinusoid position encoding table """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

'''
Credits to https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
'''
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

## a little modification on GEGLU()
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), # nn.Linear(dim, dim * mult * 2)
            nn.GELU(), # GEGLU()
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class PerAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=True)
        self.to_out = nn.Linear(inner_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, context = None, mask = None):
        h = self.heads
        
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)
        
        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class Perceiver(nn.Module):
    def __init__(self,
                 dim,
                 latent_dim,
                 num_latents,
                 depth,
                 cross_heads = 1,
                 cross_dim_head = 64,
                 latent_heads = 8,
                 latent_dim_head = 64,
                 attn_dropout = 0.,
                 ff_dropout = 0.,
                 ) -> None:
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self.cross_attn_blocks = nn.ModuleList([
            PreNorm(latent_dim, PerAttention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        ])

        self.layers = nn.ModuleList([])
        
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(latent_dim, PerAttention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout)),
                PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
            ]))
        
    def forward(self, data, mask = None):
        # ipdb.set_trace()
        b = data.shape[0]
        
        x = repeat(self.latents, 'n d -> b n d', b = b)
        
        cross_attn, cross_ff = self.cross_attn_blocks

        # cross attention only happens once for Perceiver IO
        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x

        # layers
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x


# deprecated
class PerBlock(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0) -> None:
        super().__init__()
        self.attn = PerAttention(query_dim, context_dim, heads, dim_head, dropout)
        self.ln1 = nn.LayerNorm(context_dim)
        self.ln2 = nn.LayerNorm(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, 4 * query_dim),
            nn.GELU(),
            nn.Linear(4 * query_dim, query_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, context=None):
        context = self.ln1(context) if context is not None else None
        x_attn = self.attn(x, context)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x

'''
----------- dividing line ------------
'''

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None, input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        device = sequences.device
        num_steps = sequences.size(1)
        prev_steps = past_keys_values.size if past_keys_values is not None else 0

        base_pos = prev_steps + torch.arange(num_steps, device=device)
        if input_mask is not None and not input_mask[:, 0].all():
            assert past_keys_values is None
            rolls_step = input_mask.sum(-1)
            base_pos = base_pos[(torch.arange(len(base_pos), device=device)[:, None] + rolls_step) % len(base_pos)].transpose(0, 1) # roll leftwards
            # print('passed here...')

        # ipdb.set_trace()
        sequences += self.pos_emb(base_pos)

        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i], input_mask)

        x = self.ln_f(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None, input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_attn = self.attn(self.ln1(x), past_keys_values, input_mask)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        assert config.attention in ('causal', 'block_causal')
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        ### 这里改动不同的mask来实现不同的transformer特性
        global_mask = torch.ones(config.max_tokens, config.max_tokens)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        self.register_buffer('mask', causal_mask if config.attention == 'causal' else block_causal_mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None, input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if input_mask is not None:
            attn_mask = repeat(input_mask.to(torch.float32), 'b l -> b nh l 1', nh = self.num_heads)
            attn_mask = attn_mask @ attn_mask.transpose(-2, -1)
            attn_mask = torch.tril(attn_mask)
        else:
            attn_mask = None

        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if exists(attn_mask):
            att = att.masked_fill(attn_mask[:, :, L:L + T, :L + T] == 0, -1e10) # -1e10
        else:
            att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        
        # att.nan_to_num_(0.)
        # att = att.masked_fill(attn_mask[L:L + T, :L + T] == 0, 0.)

        att = self.attn_drop(att)
        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')

        y = self.resid_drop(self.proj(y))

        return y
