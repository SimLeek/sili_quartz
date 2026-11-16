"""B. Passed state tokens: Recurrent Memory Transformer (Bulatov et al. 2022), chunked.

Per chunk: prepend the m memory tokens (read), append them again (write), run
the whole stack over [read ; chunk ; write]; the write positions' outputs become
the next chunk's memory tokens. The memory is a fixed-size vector state, so
what it can carry is bounded by m*d, and what it learns to carry is bounded by
how far gradient flows across chunks (`tbptt_chunks`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sili_quartz.models.common import Block, segment_mask


@dataclass
class RMTState:
    mem: Tensor  # [B, m, d]

    def detach(self) -> RMTState:
        return RMTState(self.mem.detach())


class RMT(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 64,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        chunk: int = 16,
        mem_tokens: int = 4,
        tbptt_chunks: int = 64,
    ):
        super().__init__()
        self.vocab, self.d, self.chunk, self.m = vocab, d, chunk, mem_tokens
        self.tbptt_chunks = tbptt_chunks
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(chunk, d)
        self.mem_init = nn.Parameter(torch.randn(mem_tokens, d) * 0.02)
        self.read_pos = nn.Parameter(torch.zeros(mem_tokens, d))
        self.write_pos = nn.Parameter(torch.zeros(mem_tokens, d))
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def init_state(self, batch: int, device=None) -> RMTState:
        return RMTState(self.mem_init.unsqueeze(0).expand(batch, -1, -1).to(device))

    def forward(self, tokens: Tensor, state: RMTState | None = None) -> tuple[Tensor, RMTState]:
        B, T = tokens.shape
        dev = tokens.device
        if state is None:
            state = self.init_state(B, dev)
        out, n_chunks = [], 0
        for s in range(0, T, self.chunk):
            ids = tokens[:, s : s + self.chunk]
            L = ids.shape[1]
            x = self.embed(ids) + self.pos(torch.arange(L, device=dev))
            x = torch.cat([state.mem + self.read_pos, x, state.mem + self.write_pos], dim=1)
            mm = segment_mask(self.m, L, self.m, dev)
            for blk in self.blocks:
                x = blk(x, mm)
            state = RMTState(x[:, -self.m :])
            out.append(x[:, self.m : self.m + L])
            n_chunks += 1
            if n_chunks % self.tbptt_chunks == 0:
                state = state.detach()
        return self.head(self.ln_f(torch.cat(out, dim=1))), state
