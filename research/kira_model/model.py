"""Petit décodeur Transformer causal standard, poids initialisés aléatoirement."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import VOCAB_SIZE


@dataclass
class Config:
    width: int = 64
    heads: int = 4
    layers: int = 2
    context: int = 128

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError("Dimensions entières strictement positives requises")
        if self.width % self.heads:
            raise ValueError("width doit être divisible par heads")


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads = cfg.heads
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.mlp = nn.Sequential(nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(),
                                 nn.Linear(4 * cfg.width, cfg.width))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for t in (q, k, v))
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        x = x + self.proj(attention.transpose(1, 2).contiguous().view(batch, length, width))
        return x + self.mlp(self.norm2(x))


class KiraSeed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(VOCAB_SIZE, cfg.width)
        self.position = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.Sequential(*(Block(cfg) for _ in range(cfg.layers)))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, VOCAB_SIZE, bias=False)
        self.apply(self._init)
        self.head.weight = self.token.weight

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if not 0 < tokens.size(1) <= self.cfg.context:
            raise ValueError("Séquence vide ou supérieure au contexte")
        x = self.token(tokens) + self.position(torch.arange(tokens.size(1), device=tokens.device))
        return self.head(self.norm(self.blocks(x)))
