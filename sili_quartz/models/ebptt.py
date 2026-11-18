"""Fake backprop for e-BPTT: credit a detached write with remembered inputs.

When the memory state is detached, the gradient G = dL/dM that reaches the
detached M has nowhere to go. `credit` routes it into W_V / W_K as if it had
backpropagated through the earlier writes, with their saved inputs replaced
by remembered ones (RTRL / e-prop: learning signal x eligibility trace).
Like e-prop, the trace keeps the memory's own state dynamics: each later
write multiplies M by (I - c k k^T), and the remembered keys are carried
through it (credit_carry="full"). Dropping that ("none", a plain sum) is NOT
e-prop and measured far worse (0.268 vs 0.969 on ledger row 15).

For a write M += c (v - M k) k^T, with v = W_V x and k = W_K x_k:

    dL/dW_V = G  sum_i c_i k_i x_i^T          = G   E_V
    dL/dW_K = G^T sum_i c_i (v_i - M k_i) x_k,i^T = G^T E_K

E_V and E_K are [B, d_mem, d_in], the same size as M. They are either a running
sum over every write ("trace") or rebuilt from a buffer of the last n writes
that were not ~zero ("buffer"). The traces are detached; nothing of the
earlier graph is kept.

`substitute_saved` is the general form: during backward, a saved tensor is
replaced by another one, and torch recomputes the gradients that use it.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor


def _top(score: Tensor, k: int) -> Tensor:
    return score.topk(k, dim=1).indices  # [B, k]


class _Credit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, w_v, w_k, E_V, E_K, density, score, u):
        ctx.save_for_backward(E_V, E_K, u)
        ctx.density, ctx.score = density, score
        return M.view_as(M)

    @staticmethod
    def backward(ctx, G):
        E_V, E_K, u = ctx.saved_tensors
        none = (None,) * 6
        if ctx.density >= 1.0:
            g_v = torch.einsum("bij,bjk->ik", G, E_V)
            g_k = torch.einsum("bji,bjk->ik", G, E_K)
            return (None, g_v, g_k) + none[:5]
        # Sparse credit: keep the top `density` of M's key dims for G.E_V and of
        # its value dims for G^T.E_K, skip the rest. Cost 2 p d^3 instead of 2 d^3.
        d = G.shape[1]
        k = max(1, round(ctx.density * d))

        def score(norm):
            if ctx.score == "grad":
                return norm
            if ctx.score == "noise":
                return u
            return norm * u  # grad_noise

        j = _top(score(G.norm(dim=1)), k)  # key dims (columns of G, rows of E_V)
        Gj = torch.gather(G, 2, j[:, None, :].expand(-1, d, -1))
        Ej = torch.gather(E_V, 1, j[:, :, None].expand(-1, -1, E_V.shape[2]))
        g_v = torch.einsum("bij,bjk->ik", Gj, Ej)
        i = _top(score(G.norm(dim=2)), k)  # value dims (rows of G and of E_K)
        Gi = torch.gather(G, 1, i[:, :, None].expand(-1, -1, d))
        Ei = torch.gather(E_K, 1, i[:, :, None].expand(-1, -1, E_K.shape[2]))
        g_k = torch.einsum("bij,bik->jk", Gi, Ei)
        return (None, g_v, g_k) + none[:5]


def credit(M, w_v, w_k, E_V, E_K, density: float = 1.0, score: str = "grad", u: Tensor | None = None):
    """M unchanged forward; backward sends G E_V to w_v and G^T E_K to w_k,
    optionally over only the top `density` of memory dims (by |G|, |G| x u, or u)."""
    if u is None:
        u = torch.ones(M.shape[0], M.shape[1], device=M.device, dtype=M.dtype)
    return _Credit.apply(M.detach(), w_v, w_k, E_V, E_K, density, score, u)


@contextmanager
def substitute_saved(target: Tensor, replacement: Tensor):
    """Inside this block, ops that save `target` for backward get `replacement`
    instead when backward runs. Forward values are untouched."""

    def pack(t):
        return replacement if t is target else t

    def unpack(t):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield
