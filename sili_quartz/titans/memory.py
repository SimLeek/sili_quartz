"""Titans neural long-term memory (arXiv:2501.00663 §3.1), matrix-valued, batched.

    k_t = x_t W_K,  v_t = x_t W_V,  q_t = x_t W_Q                 (11, 14)
    l(M; x_t)  = || M k_t - v_t ||^2                               (12)
    S_t = eta_t S_{t-1} - theta_t grad_M l(M_{t-1}; x_t)           (10)
    M_t = (1 - alpha_t) M_{t-1} + S_t                              (13)
    y_t = M* q_t                                                   (15)

For a matrix memory the inner gradient is closed form, 2 (M k - v) k^T, so a
write is an outer-product update inside the forward pass. theta defaults to
0.5 / ||k||^2, the normalized delta rule, which stores one pair exactly in one
shot from M = 0 (test_memory.py::test_one_shot_store).

Gates: "const" uses the given alpha/eta. "kernel_matched" has no free knob:
with b = eta the retention kernel is (n+1) b^n (Law 5 in
sili__new/bend_proofs/titans_gate_retention_kernel), peaking at ~1/(1-b);
matching that peak to the measured retrieval lag L gives
alpha = 1/L, eta = 1 - 1/L, and theta scaled by (1 - eta) so eta does not
also change the effective step size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MemoryState:
    M: Tensor  # [B, d, d]
    S: Tensor  # [B, d, d] momentum / surprise trace
    lag: Tensor | None = None  # [B] EMA of retrieval lag (kernel_matched only)
    keys: Tensor | None = None  # [B, K, d] recent write keys, for lag estimation
    ticks: Tensor | None = None  # [B, K] write tick of each buffered key
    tick: int = 0

    def detach(self) -> MemoryState:
        return MemoryState(
            self.M.detach(),
            self.S.detach(),
            None if self.lag is None else self.lag.detach(),
            None if self.keys is None else self.keys.detach(),
            self.ticks,
            self.tick,
        )


class TitansMemory(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_mem: int | None = None,
        *,
        alpha: float = 0.0,
        eta: float = 0.0,
        theta_scale: float = 1.0,
        gates: str = "const",
        lag_beta: float = 0.02,
        lag_buffer: int = 64,
        eps: float = 1e-6,
    ):
        super().__init__()
        if gates not in ("const", "kernel_matched"):
            raise ValueError(f"gates must be 'const' or 'kernel_matched', got {gates!r}")
        d_mem = d_mem or d_in
        self.d_in, self.d_mem = d_in, d_mem
        self.alpha, self.eta, self.theta_scale = alpha, eta, theta_scale
        self.gates, self.lag_beta, self.lag_buffer, self.eps = gates, lag_beta, lag_buffer, eps
        self.W_K = nn.Linear(d_in, d_mem, bias=False)
        self.W_V = nn.Linear(d_in, d_mem, bias=False)
        self.W_Q = nn.Linear(d_in, d_mem, bias=False)

    def init_state(self, batch: int, device=None, dtype=None) -> MemoryState:
        d = self.d_mem
        z = torch.zeros(batch, d, d, device=device, dtype=dtype)
        st = MemoryState(z, z.clone())
        if self.gates == "kernel_matched":
            st.lag = torch.ones(batch, device=device, dtype=dtype)
            st.keys = torch.zeros(batch, self.lag_buffer, d, device=device, dtype=dtype)
            st.ticks = torch.full((batch, self.lag_buffer), -(10**9), device=device, dtype=torch.long)
        return st

    def _gates(self, state: MemoryState) -> tuple[Tensor | float, Tensor | float, Tensor | float]:
        """Returns (alpha, eta, theta multiplier), scalars or [B,1,1] tensors."""
        if self.gates == "const":
            return self.alpha, self.eta, self.theta_scale
        lag = state.lag.clamp_min(1.0).view(-1, 1, 1)
        eta = 1.0 - 1.0 / lag
        return 1.0 / lag, eta, self.theta_scale * (1.0 - eta)

    def read(self, x: Tensor, state: MemoryState) -> Tensor:
        """y = M q for x of shape [B, ..., d_in]; eq 15. Updates the lag estimate."""
        q = self.W_Q(x)
        if self.gates == "kernel_matched":
            self._note_read(q, state)
        return torch.einsum("bij,b...j->b...i", state.M, q)

    @torch.no_grad()
    def _note_read(self, q: Tensor, state: MemoryState) -> None:
        # Lag of the best-matching buffered key is the stream's own retrieval
        # horizon -- supplied by the data, so no oracle. EMA over queries.
        # Needs W_Q ~ W_K alignment, which retrieval itself requires; until the
        # model learns it, no key matches and the lag stays at its default 1,
        # i.e. a plain delta rule (alpha=1, eta=0).
        qf = q.reshape(q.shape[0], -1, q.shape[-1])  # [B, L, d]
        qn = qf / qf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        kn = state.keys / state.keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sims = torch.einsum("bld,bkd->blk", qn, kn)
        best, idx = sims.max(dim=-1)  # [B, L]
        lag = (state.tick - torch.gather(state.ticks, 1, idx)).to(q.dtype)
        valid = best > 0.5
        n = valid.sum(dim=1).clamp_min(1)
        mean_lag = (lag * valid).sum(dim=1) / n
        has = valid.any(dim=1)
        state.lag = torch.where(has, state.lag + self.lag_beta * (mean_lag - state.lag), state.lag)

    def write(self, x: Tensor, state: MemoryState, k_src: Tensor | None = None) -> MemoryState:
        """Sequential inner-loop steps (eqs 10-13) over the L tokens of x [B, L, d_in].

        k_src lets the key come from a different representation than the value
        (same shape as x); default is x itself, as in eq 11.
        """
        k_all = self.W_K(x if k_src is None else k_src)
        v_all = self.W_V(x)
        M, S = state.M, state.S
        alpha, eta, th_mul = self._gates(state)
        for t in range(x.shape[1]):
            k, v = k_all[:, t], v_all[:, t]  # [B, d]
            err = torch.einsum("bij,bj->bi", M, k) - v
            grad = 2.0 * err.unsqueeze(-1) * k.unsqueeze(-2)  # [B, d, d]
            theta = 0.5 / (k.pow(2).sum(-1) + self.eps)
            theta = theta.view(-1, 1, 1) * th_mul
            S = eta * S - theta * grad
            M = (1.0 - alpha) * M + S
        out = MemoryState(M, S, state.lag, state.keys, state.ticks, state.tick + x.shape[1])
        if self.gates == "kernel_matched":
            self._buffer_keys(k_all.detach(), state, out)
        return out

    @torch.no_grad()
    def _buffer_keys(self, k_all: Tensor, prev: MemoryState, out: MemoryState) -> None:
        L = k_all.shape[1]
        ticks = torch.arange(prev.tick, prev.tick + L, device=k_all.device).expand(k_all.shape[0], L)
        out.keys = torch.cat([prev.keys, k_all], dim=1)[:, -self.lag_buffer :]
        out.ticks = torch.cat([prev.ticks, ticks], dim=1)[:, -self.lag_buffer :]
