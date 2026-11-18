"""B + C: Titans memory-as-context plus RMT state tokens (ARMT-style).

Associative memory holds bindings; the passed tokens hold state. Per chunk the
attention input is [retrievals ; rmt read ; chunk ; rmt write].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sili_quartz.models.memory import MemoryState
from sili_quartz.models.titans import TitansMAC


@dataclass
class CombinedState:
    mem: MemoryState
    rmt: Tensor  # [B, m, d]

    def detach(self) -> CombinedState:
        return CombinedState(self.mem.detach(), self.rmt.detach())


class Combined(TitansMAC):
    def __init__(self, vocab: int, d: int = 64, *, mem_tokens: int = 4, **kw):
        super().__init__(vocab, d, **kw)
        self.m = mem_tokens
        self.rmt_init = nn.Parameter(torch.randn(mem_tokens, d) * 0.02)
        self.read_pos = nn.Parameter(torch.zeros(mem_tokens, d))
        self.write_pos = nn.Parameter(torch.zeros(mem_tokens, d))

    def init_state(self, batch: int, device=None) -> CombinedState:
        mem = self.memory.init_state(batch, device=device, dtype=self.embed.weight.dtype)
        return CombinedState(mem, self.rmt_init.unsqueeze(0).expand(batch, -1, -1).to(device))

    def _extra_prefix(self, state, L, dev):
        return state.rmt + self.read_pos, self.m

    def _extra_suffix(self, state, L, dev):
        return state.rmt + self.write_pos, self.m

    def _consume_suffix(self, x_suffix, state):
        return CombinedState(state.mem, x_suffix)

    def _mem(self, state):
        return state.mem

    def _set_mem(self, state, mem):
        return CombinedState(mem, state.rmt)

    def _detach(self, state):
        return CombinedState(self.memory.with_credit(state.mem), state.rmt.detach())
