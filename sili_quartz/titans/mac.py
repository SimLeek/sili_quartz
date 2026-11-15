"""Titans memory-as-context (MAC, arXiv:2501.00663 §4.1), chunked.

Per chunk of L tokens: run the layers before the memory, retrieve one memory
vector per token (eq 15) from the memory as it stood at the chunk start,
prepend those L retrievals as context tokens, run the remaining layers with
attention over [retrievals ; chunk] (retrievals fully visible, chunk causal),
then write the chunk into memory (eqs 10-13). Memory state is detached every
`tbptt_chunks` chunks, so the per-chunk backward cost is bounded regardless of
sequence length.

Keys, values and queries for the memory all come from the SAME representation
(the layer input, eq 11). `write_key_from_prev` instead keys each write on the
previous token's representation, so a (key, value) pair is stored as
"key token -> value token" even when the two tokens never share a position.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sili_quartz.titans.memory import MemoryState, TitansMemory


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


def _causal(n: int, device) -> Tensor:
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


def _mac_mask(L: int, device) -> Tensor:
    """[retrievals(L) ; chunk(L)]: retrievals see only themselves, chunk is causal
    over itself and sees every retrieval. True = masked."""
    m = torch.ones(2 * L, 2 * L, dtype=torch.bool, device=device)
    m[:L, :L] = _causal(L, device)
    m[L:, :L] = False
    m[L:, L:] = _causal(L, device)
    return m


class TitansMAC(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 64,
        *,
        n_heads: int = 4,
        n_layers: int = 2,
        chunk: int = 16,
        memory: bool = True,
        mem_after_layer: int = 1,
        tbptt_chunks: int = 1,
        write_key_from_prev: bool = False,
        output_gate: bool = True,
        **mem_kwargs,
    ):
        super().__init__()
        if not 1 <= mem_after_layer <= n_layers:
            raise ValueError("mem_after_layer must be in [1, n_layers]")
        self.vocab, self.d, self.chunk = vocab, d, chunk
        self.memory_on, self.tbptt_chunks = memory, tbptt_chunks
        self.write_key_from_prev, self.output_gate = write_key_from_prev, output_gate
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(chunk, d)
        self.mem_pos = nn.Embedding(chunk, d)
        self.pre = nn.ModuleList(Block(d, n_heads) for _ in range(mem_after_layer))
        self.post = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers - mem_after_layer))
        self.memory = TitansMemory(d, **mem_kwargs) if memory else None
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def init_state(self, batch: int, device=None) -> MemoryState | None:
        if self.memory is None:
            return None
        return self.memory.init_state(batch, device=device, dtype=self.embed.weight.dtype)

    def forward(self, tokens: Tensor, state: MemoryState | None = None) -> tuple[Tensor, MemoryState | None]:
        """tokens [B, T] -> logits [B, T, vocab]. T is processed in chunks; the
        last chunk may be short."""
        B, T = tokens.shape
        dev = tokens.device
        if self.memory is not None and state is None:
            state = self.init_state(B, dev)
        out, prev_h, n_chunks = [], None, 0
        for s in range(0, T, self.chunk):
            ids = tokens[:, s : s + self.chunk]
            L = ids.shape[1]
            pos = torch.arange(L, device=dev)
            x = self.embed(ids) + self.pos(pos)
            cm = _causal(L, dev)
            for blk in self.pre:
                x = blk(x, cm)
            h = x
            if self.memory is None:
                for blk in self.post:
                    x = blk(x, cm)
            else:
                read = self.memory.read(h, state) + self.mem_pos(pos)
                x = torch.cat([read, h], dim=1)
                mm = _mac_mask(L, dev)
                for blk in self.post:
                    x = blk(x, mm)
                x = x[:, L:]
                if self.output_gate:  # eq 24: y = o * M*(o)
                    x = x * torch.sigmoid(self.memory.read(x, state))
                k_src = None
                if self.write_key_from_prev:
                    first = h[:, :1] if prev_h is None else prev_h[:, -1:]
                    k_src = torch.cat([first, h[:, :-1]], dim=1)
                state = self.memory.write(h, state, k_src=k_src)
                prev_h = h
                n_chunks += 1
                if n_chunks % self.tbptt_chunks == 0:
                    state = state.detach()
            out.append(x)
        logits = self.head(self.ln_f(torch.cat(out, dim=1)))
        return logits, state


def recall_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy over positions where targets != -100."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
