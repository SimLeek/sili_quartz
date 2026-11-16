"""Shared pieces: transformer block, the chunk/prefix/suffix attention mask, loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


def causal(n: int, device) -> Tensor:
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


def segment_mask(prefix: int, L: int, suffix: int, device) -> Tensor:
    """Mask (True = blocked) for [prefix ; chunk(L) ; suffix].

    prefix tokens (memory reads, retrievals) see only the prefix, causally;
    chunk tokens see the whole prefix and the chunk causally;
    suffix tokens (memory writes) see everything.
    """
    n = prefix + L + suffix
    m = torch.ones(n, n, dtype=torch.bool, device=device)
    m[:prefix, :prefix] = causal(prefix, device)
    m[prefix : prefix + L, :prefix] = False
    m[prefix : prefix + L, prefix : prefix + L] = causal(L, device)
    m[prefix + L :, :] = False
    return m


def recall_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy over positions where targets != -100."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
