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


class _Credit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, w_v, w_k, E_V, E_K):
        ctx.save_for_backward(E_V, E_K)
        return M.view_as(M)

    @staticmethod
    def backward(ctx, G):
        E_V, E_K = ctx.saved_tensors
        g_v = torch.einsum("bij,bjk->ik", G, E_V)
        g_k = torch.einsum("bji,bjk->ik", G, E_K)
        return None, g_v, g_k, None, None


def credit(M: Tensor, w_v: Tensor, w_k: Tensor, E_V: Tensor, E_K: Tensor) -> Tensor:
    """M unchanged forward; backward sends G E_V to w_v and G^T E_K to w_k."""
    return _Credit.apply(M.detach(), w_v, w_k, E_V, E_K)


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
