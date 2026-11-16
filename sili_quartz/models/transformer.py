"""A. Base: chunked causal transformer, no memory of any kind. The control."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sili_quartz.models.common import Block, causal


class Transformer(nn.Module):
    def __init__(self, vocab: int, d: int = 64, *, n_heads: int = 4, n_layers: int = 2, chunk: int = 16):
        super().__init__()
        self.vocab, self.d, self.chunk = vocab, d, chunk
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(chunk, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def init_state(self, batch: int, device=None):
        return None

    def forward(self, tokens: Tensor, state=None) -> tuple[Tensor, None]:
        B, T = tokens.shape
        dev = tokens.device
        out = []
        for s in range(0, T, self.chunk):
            ids = tokens[:, s : s + self.chunk]
            L = ids.shape[1]
            x = self.embed(ids) + self.pos(torch.arange(L, device=dev))
            cm = causal(L, dev)
            for blk in self.blocks:
                x = blk(x, cm)
            out.append(x)
        return self.head(self.ln_f(torch.cat(out, dim=1))), None
