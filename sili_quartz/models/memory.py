"""Titans neural long-term memory (arXiv:2501.00663 §3.1), matrix-valued, batched,
with ATLAS's Omega window (arXiv:2505.23735).

    k_t = x_t W_K,  v_t = x_t W_V,  q_t = x_t W_Q                 (11, 14)
    l(M; x_t)  = || M k_t - v_t ||^2                               (12)
    S_t = eta_t S_{t-1} - theta_t grad_M l                         (10)
    M_t = (1 - alpha_t) M_{t-1} + S_t                              (13)
    y_t = M q_t                                                    (15)

ATLAS's polynomial key features were
tried and dropped: they stall together with the Omega window at scale, and the
low-rank variant fails to start on some seeds
(docs/results/2026-11-17-seven-seeds-split-omega.md).

`window`: tokens per inner step. 1 is Titans' online rule. c > 1 is the Omega
rule: one step on the joint loss of the last c pairs, so the memory fits the
context rather than each token in turn. The step is normalized per key
(sum_i gamma_i (M k_i - v_i) k_i^T / ||k_i||^2, theta = 0.5), which
stores a window of orthogonal keys exactly in one step and reduces to the
0.5/||k||^2 delta rule at c = 1. `inner_steps` iterates toward the
least-squares solution for correlated keys.

`credit`: how a write gets gradient once the state is detached. "bptt" (default):
only through the graph. "trace": e-BPTT eligibility traces, a running sum over
every write (fixed size, = M). "buffer": the same, rebuilt from the last
`credit_buffer` writes whose update was not ~zero. See models/ebptt.py.

`gates`: "const" uses alpha/eta as given. "kernel_matched" has no free knob:
alpha = 1/L, eta = 1 - 1/L, theta scaled by (1 - eta), from the retention
kernel's Law 5 (sili__new/bend_proofs/titans_gate_retention_kernel) and the
measured retrieval lag L.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MemoryState:
    M: Tensor  # [B, d_mem, d_mem]
    S: Tensor  # [B, d_mem, d_mem] momentum / surprise trace
    lag: Tensor | None = None  # [B] EMA of retrieval lag (kernel_matched only)
    keys: Tensor | None = None  # [B, K, d_mem] recent raw write keys, for lag estimation
    ticks: Tensor | None = None  # [B, K]
    tick: int = 0
    E_V: Tensor | None = None  # [B, d_mem, d_in] e-BPTT eligibility traces (credit != "bptt")
    E_K: Tensor | None = None
    buf: tuple | None = None  # credit="buffer": last n non-zero writes (ck, x, ce, x_k), each [B, n, .]

    def detach(self) -> MemoryState:
        return MemoryState(
            self.M.detach(),
            self.S.detach(),
            None if self.lag is None else self.lag.detach(),
            None if self.keys is None else self.keys.detach(),
            self.ticks,
            self.tick,
            self.E_V,
            self.E_K,
            self.buf,
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
        window: int = 1,
        inner_steps: int = 1,
        window_decay: float = 1.0,
        lag_beta: float = 0.02,
        lag_buffer: int = 64,
        eps: float = 1e-6,
        credit: str = "bptt",
        credit_buffer: int = 16,
        credit_tol: float = 1e-3,
        credit_k: bool = True,
    ):
        super().__init__()
        if gates not in ("const", "kernel_matched"):
            raise ValueError(f"gates must be 'const' or 'kernel_matched', got {gates!r}")
        if window < 1 or inner_steps < 1:
            raise ValueError("window and inner_steps must be >= 1")
        if credit not in ("bptt", "trace", "buffer"):
            raise ValueError(f"credit must be 'bptt', 'trace' or 'buffer', got {credit!r}")
        if credit != "bptt" and (eta != 0.0 or inner_steps != 1 or gates != "const"):
            raise ValueError("e-BPTT credit assumes eta=0, inner_steps=1, const gates")
        self.credit, self.credit_buffer, self.credit_tol = credit, credit_buffer, credit_tol
        self.credit_k = credit_k  # False: credit W_V only (the exactly-carried side)
        d_mem = d_mem or d_in
        self.d_in, self.d_mem = d_in, d_mem
        self.alpha, self.eta, self.theta_scale = alpha, eta, theta_scale
        self.gates = gates
        self.window, self.inner_steps, self.window_decay = window, inner_steps, window_decay
        self.lag_beta, self.lag_buffer, self.eps = lag_beta, lag_buffer, eps
        self.W_K = nn.Linear(d_in, d_mem, bias=False)
        self.W_V = nn.Linear(d_in, d_mem, bias=False)
        self.W_Q = nn.Linear(d_in, d_mem, bias=False)

    def init_state(self, batch: int, device=None, dtype=None) -> MemoryState:
        z = torch.zeros(batch, self.d_mem, self.d_mem, device=device, dtype=dtype)
        st = MemoryState(z, z.clone())
        if self.credit == "trace":
            st.E_V = torch.zeros(batch, self.d_mem, self.d_in, device=device, dtype=dtype)
            st.E_K = torch.zeros_like(st.E_V)
        elif self.credit == "buffer":
            e = lambda d: torch.zeros(batch, 0, d, device=device, dtype=dtype)  # noqa: E731
            st.buf = (e(self.d_mem), e(self.d_in), e(self.d_mem), e(self.d_in))
        if self.gates == "kernel_matched":
            st.lag = torch.ones(batch, device=device, dtype=dtype)
            st.keys = torch.zeros(batch, self.lag_buffer, self.d_mem, device=device, dtype=dtype)
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
        qf = q.reshape(q.shape[0], -1, q.shape[-1])
        qn = qf / qf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        kn = state.keys / state.keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        best, idx = torch.einsum("bld,bkd->blk", qn, kn).max(dim=-1)
        lag = (state.tick - torch.gather(state.ticks, 1, idx)).to(q.dtype)
        valid = best > 0.5
        mean_lag = (lag * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        state.lag = torch.where(
            valid.any(dim=1), state.lag + self.lag_beta * (mean_lag - state.lag), state.lag
        )

    def write(self, x: Tensor, state: MemoryState, k_src: Tensor | None = None) -> MemoryState:
        """Inner-loop steps over x [B, L, d_in], `window` tokens per step (eqs 10-13).

        k_src lets the key come from a different representation than the value
        (same shape as x); default is x itself, as in eq 11.
        """
        xk = x if k_src is None else k_src
        k_raw = self.W_K(xk)
        K, V = k_raw, self.W_V(x)  # [B, L, d_mem] each
        M, S = state.M, state.S
        alpha, eta, th_mul = self._gates(state)
        L, c = x.shape[1], self.window
        E_V, E_K, buf = state.E_V, state.E_K, state.buf
        for s in range(0, L, c):
            Kw, Vw = K[:, s : s + c], V[:, s : s + c]
            n = Kw.shape[1]
            gamma = self.window_decay ** torch.arange(n - 1, -1, -1, device=x.device, dtype=x.dtype)
            w = (gamma / (Kw.pow(2).sum(-1) + self.eps)).unsqueeze(-1)  # [B, n, 1] per-key normalizer
            # Step = 1 / lambda_max of the normalized Gram matrix: exactly 1 for
            # orthogonal keys (one-shot store), <= 1/n for fully correlated ones
            # (no overshoot along shared directions). Differentiable, so the outer
            # loop sees how the projections change the step.
            Kn = Kw * w.sqrt()
            lam = torch.linalg.matrix_norm(Kn @ Kn.transpose(1, 2), ord=2).clamp_min(1e-6)
            step = (th_mul / lam).view(-1, 1, 1) if torch.is_tensor(th_mul) else th_mul / lam.view(-1, 1, 1)
            for _ in range(self.inner_steps):
                err = torch.einsum("bij,bnj->bni", M, Kw) - Vw  # [B, n, d_mem]
                grad = torch.einsum("bni,bnj->bij", err * w, Kw)  # sum_i w_i err_i k_i^T
                if self.credit != "bptt":
                    E_V, E_K, buf = self._note_write(
                        E_V, E_K, buf, (step * w).detach(), Kw.detach(), -err.detach(),
                        x[:, s : s + c].detach(), xk[:, s : s + c].detach(), alpha,
                    )
                S = eta * S - step * grad
                M = (1.0 - alpha) * M + S
        out = MemoryState(M, S, state.lag, state.keys, state.ticks, state.tick + L, E_V, E_K, buf)
        if self.gates == "kernel_matched":
            self._buffer_keys(k_raw.detach(), state, out)
        return out

    @torch.no_grad()
    def _note_write(self, E_V, E_K, buf, cw, k, e, x, xk, alpha):
        """Record one window's write M <- (1-alpha) M (I - K^T C K) + sum_i c_i v_i k_i^T.

        Earlier writes' keys are carried through (I - K^T C K), the exact
        state-to-state Jacobian on the V side (O(d * d_in) per window). The K
        side keeps the plain sum: its exact form needs a d^3 trace."""
        ck = cw * k  # [B, n, d_mem]
        a = 1.0 - alpha
        if self.credit == "trace":
            # (I - K^T C K) E_V, then add this window's sum_i c_i k_i x_i^T
            E_V = a * (E_V - torch.einsum("bnd,bnj->bdj", k, torch.einsum("bnd,bdj->bnj", ck, E_V)))
            E_V = E_V + torch.einsum("bni,bnj->bij", ck, x)
            E_K = a * E_K + torch.einsum("bni,bnj->bij", cw * e, xk)
            return E_V, E_K, buf
        # buffer: carry the stored keys, then keep the last n tokens whose
        # update cw*|e|*|k| is not ~zero, per batch row.
        old_ck = a * (buf[0] - torch.einsum("bmn,bnd->bmd", buf[0] @ ck.transpose(1, 2), k))
        buf = (old_ck, buf[1], a * buf[2], buf[3])
        new = (ck, x, cw * e, xk)
        size = (cw.squeeze(-1) * e.norm(dim=-1) * k.norm(dim=-1)).abs()  # [B, n]
        live = size > self.credit_tol
        cat = [torch.cat([b, n * live.unsqueeze(-1)], dim=1) for b, n in zip(buf, new)]
        keep = torch.cat([buf[1].abs().sum(-1) > 0, live], dim=1)
        # stable sort puts kept entries last, in time order; take the last n
        order = torch.sort(keep.to(torch.int8), dim=1, stable=True).indices[:, -self.credit_buffer :]
        buf = tuple(torch.gather(t, 1, order.unsqueeze(-1).expand(-1, -1, t.shape[-1])) for t in cat)
        return E_V, E_K, buf

    def with_credit(self, state: MemoryState) -> MemoryState:
        """Detached copy of `state` whose M routes dL/dM into W_V/W_K (e-BPTT)."""
        from sili_quartz.models.ebptt import credit

        st = state.detach()
        if self.credit == "bptt":
            return st
        if self.credit == "trace":
            E_V, E_K = st.E_V, st.E_K
        else:
            ck, x, ce, xk = st.buf
            E_V, E_K = torch.einsum("bni,bnj->bij", ck, x), torch.einsum("bni,bnj->bij", ce, xk)
        if not self.credit_k:
            E_K = torch.zeros_like(E_K)
        st.M = credit(st.M, self.W_V.weight, self.W_K.weight, E_V, E_K)
        return st

    @torch.no_grad()
    def _buffer_keys(self, k_all: Tensor, prev: MemoryState, out: MemoryState) -> None:
        L = k_all.shape[1]
        ticks = torch.arange(prev.tick, prev.tick + L, device=k_all.device).expand(k_all.shape[0], L)
        out.keys = torch.cat([prev.keys, k_all], dim=1)[:, -self.lag_buffer :]
        out.ticks = torch.cat([prev.ticks, ticks], dim=1)[:, -self.lag_buffer :]
