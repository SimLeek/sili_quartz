"""C. Memory as weights: Titans memory-as-context (MAC, arXiv:2501.00663 §4.1), chunked.

Per chunk of L tokens: run the layers before the memory, retrieve one memory
vector per token (eq 15) from the memory as it stood at the chunk start,
prepend those L retrievals as context tokens, run the remaining layers with
attention over [retrievals ; chunk] (retrievals fully visible, chunk causal),
then write the chunk into memory (eqs 10-13). Memory state is detached every
`tbptt_chunks` chunks, so the per-chunk backward cost is bounded regardless of
sequence length.

Keys, values and queries for the memory all come from the SAME representation
(the layer input, eq 11). `write_key_from_prev` keys each write on the previous
token's representation instead, so a (key, value) pair is stored as
"key token -> value token" even when the two tokens never share a position.
Measured necessary for beyond-window recall, as is `tbptt_chunks` spanning the
gap; both are defaults.

Memory switches (Omega window, gates) are `TitansMemory`'s.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sili_quartz.models.common import Block, causal, segment_mask
from sili_quartz.models.memory import MemoryState, TitansMemory


class TitansMAC(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 64,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        chunk: int = 16,
        mem_after_layer: int = 1,
        tbptt_chunks: int = 64,
        write_key_from_prev: bool = True,
        output_gate: bool = True,
        **mem_kwargs,
    ):
        super().__init__()
        if not 1 <= mem_after_layer <= n_layers:
            raise ValueError("mem_after_layer must be in [1, n_layers]")
        self.vocab, self.d, self.chunk = vocab, d, chunk
        self.tbptt_chunks = tbptt_chunks
        self.write_key_from_prev, self.output_gate = write_key_from_prev, output_gate
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(chunk, d)
        self.mem_pos = nn.Embedding(chunk, d)
        self.pre = nn.ModuleList(Block(d, n_heads) for _ in range(mem_after_layer))
        self.post = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers - mem_after_layer))
        self.memory = TitansMemory(d, **mem_kwargs)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def init_state(self, batch: int, device=None) -> MemoryState:
        return self.memory.init_state(batch, device=device, dtype=self.embed.weight.dtype)

    # Hooks so Combined can add its own prefix/suffix tokens without copying the loop.
    def _extra_prefix(self, state, L: int, dev) -> tuple[Tensor | None, int]:
        return None, 0

    def _extra_suffix(self, state, L: int, dev) -> tuple[Tensor | None, int]:
        return None, 0

    def _consume_suffix(self, x_suffix: Tensor, state):
        return state

    def forward(self, tokens: Tensor, state: MemoryState | None = None) -> tuple[Tensor, MemoryState]:
        """tokens [B, T] -> logits [B, T, vocab]. T is processed in chunks; the last may be short."""
        B, T = tokens.shape
        dev = tokens.device
        if state is None:
            state = self.init_state(B, dev)
        out, prev_h, n_chunks = [], None, 0
        for s in range(0, T, self.chunk):
            ids = tokens[:, s : s + self.chunk]
            L = ids.shape[1]
            pos = torch.arange(L, device=dev)
            x = self.embed(ids) + self.pos(pos)
            cm = causal(L, dev)
            for blk in self.pre:
                x = blk(x, cm)
            h = x
            read = self.memory.read(h, self._mem(state)) + self.mem_pos(pos)
            extra_pre, n_pre = self._extra_prefix(state, L, dev)
            extra_suf, n_suf = self._extra_suffix(state, L, dev)
            parts = [read] + ([extra_pre] if n_pre else []) + [h] + ([extra_suf] if n_suf else [])
            x = torch.cat(parts, dim=1)
            mm = segment_mask(L + n_pre, L, n_suf, dev)
            for blk in self.post:
                x = blk(x, mm)
            if n_suf:
                state = self._consume_suffix(x[:, -n_suf:], state)
            x = x[:, L + n_pre : L + n_pre + L]
            if self.output_gate:  # eq 24: y = o * M*(o)
                x = x * torch.sigmoid(self.memory.read(x, self._mem(state)))
            k_src = None
            if self.write_key_from_prev:
                first = h[:, :1] if prev_h is None else prev_h[:, -1:]
                k_src = torch.cat([first, h[:, :-1]], dim=1)
            state = self._set_mem(state, self.memory.write(h, self._mem(state), k_src=k_src))
            prev_h = h
            n_chunks += 1
            if n_chunks % self.tbptt_chunks == 0:
                state = self._detach(state)
            out.append(x)
        return self.head(self.ln_f(torch.cat(out, dim=1))), state

    # State accessors, overridden by Combined whose state carries two memories.
    def _mem(self, state):
        return state

    def _set_mem(self, state, mem):
        return mem

    def _detach(self, state):
        return state.detach()
