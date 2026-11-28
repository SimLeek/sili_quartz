"""Energy for a live actor and its trainer: one GPU scan over tokens, dose in a tensor.

`energy_hooks.NeuronEnergy` steps every neuron token by token from Python and
reads results back to the host, which is too slow for a 250-token frame in a
0.3 s decision and cannot run inside a CUDA graph. This is the same dynamics
for the configuration used on pretrained models (p = 1, no wake gate, fire
resets to zero) as a Triton kernel: one thread per neuron, a loop over the
call's tokens, state kept on the GPU between calls.

- The dose (drive, cost, decay, exploration) is read from a tensor, so a
  controller (the dual lambda) changes it in place, also under a CUDA graph.
- Noise comes from torch's default CUDA generator (graph-safe).
- The kernel returns a gate (0/1) and a constant per entry; the module output
  is h * gate + const, so gradients pass through kept neurons as in
  `energy.py` (fired and shut-off neurons output constants).
- No aux loss: energy here is a perturbation of the forward pass, not trained.

`energy_scan_reference` is the plain-torch loop the kernel is tested against.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only test machines
    triton = None


ACTIVATION_THRESHOLD = 1e-4


def energy_scan_reference(h: Tensor, e: Tensor, noise: Tensor, dose: Tensor, scale: float = 1.0):
    """h, noise: (T, M); e: (M,) state, updated in place; dose: [drive, cost, decay, exploration].
    Returns gate (T, M) in h's dtype and const (T, M) in h's dtype."""
    drive, cost, decay, expl = (float(v) for v in dose)
    gates, consts = [], []
    for t in range(h.shape[0]):
        x = h[t].float() / scale
        alive = x.abs() > ACTIVATION_THRESHOLD
        ne = e * (1 - decay) + drive + expl * noise[t] - cost * torch.where(alive, x, 0.0).abs()
        fire, shut = ne >= 2.0, ne <= -2.0
        shut_v = torch.where(shut, e + 2.0, 0.0)
        ne = torch.where(fire, 0.0, torch.where(shut, -2.0, ne))
        gates.append((~fire & ~shut & alive).to(h.dtype))
        consts.append((torch.where(fire, 2.0, shut_v) * scale).to(h.dtype))
        e.copy_(ne)
    return torch.stack(gates), torch.stack(consts)


if triton is not None:
    @triton.jit
    def _energy_scan_kernel(h_ptr, noise_ptr, e_ptr, gate_ptr, const_ptr, dose_ptr, T, M, scale,
                            BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < M
        drive = tl.load(dose_ptr + 0)
        cost = tl.load(dose_ptr + 1)
        decay = tl.load(dose_ptr + 2)
        expl = tl.load(dose_ptr + 3)
        e = tl.load(e_ptr + offs, mask=mask, other=0.0)
        for t in range(T):
            x = tl.load(h_ptr + t * M + offs, mask=mask, other=0.0).to(tl.float32) / scale
            nz = tl.load(noise_ptr + t * M + offs, mask=mask, other=0.0)
            alive = tl.abs(x) > 1e-4
            ne = e * (1.0 - decay) + drive + expl * nz - cost * tl.abs(tl.where(alive, x, 0.0))
            fire = ne >= 2.0
            shut = ne <= -2.0
            shut_v = tl.where(shut, e + 2.0, 0.0)
            ne = tl.where(fire, 0.0, tl.where(shut, -2.0, ne))
            keep = (fire == 0) & (shut == 0) & alive
            tl.store(gate_ptr + t * M + offs, keep.to(tl.float32).to(gate_ptr.dtype.element_ty), mask=mask)
            tl.store(const_ptr + t * M + offs,
                     (tl.where(fire, 2.0, shut_v) * scale).to(const_ptr.dtype.element_ty), mask=mask)
            e = ne
        tl.store(e_ptr + offs, e, mask=mask)


def energy_scan(h: Tensor, e: Tensor, noise: Tensor, dose: Tensor, scale: float = 1.0):
    """Same contract as energy_scan_reference; Triton on CUDA, the loop elsewhere."""
    if triton is None or not h.is_cuda:
        return energy_scan_reference(h, e, noise, dose, scale)
    h = h.contiguous()
    T, M = h.shape
    gate = torch.empty_like(h)
    const = torch.empty_like(h)
    block = 256
    _energy_scan_kernel[(triton.cdiv(M, block),)](h, noise, e, gate, const, dose, T, M, float(scale),
                                                 BLOCK=block)
    return gate, const


@dataclass
class ScanSite:
    module: nn.Module
    mean: float  # mean |h / scale| on ordinary input
    scale: float = 1.0
    cost: float = 0.03
    flat_tokens: bool = False  # the module outputs (tokens, n), e.g. the vision merger


class ScanEnergy:
    """Energy on module outputs (B, T, n), stepped over T, state per (batch row, neuron).

    The dose at multiplier m: cost = base*m_cost, drive = cost*mean*m_drive,
    exploration = min(0.001, base*mean/4)*m_noise, decay = drive/decay_div,
    as in NeuronEnergy. `set_dose` writes the per-site dose tensors in place.
    """

    def __init__(self, sites: list[ScanSite], decay_div: float = 40.0, batch: int = 1, col: int = 0):
        """col > 1: column-averaged energy. Each neuron keeps its own energy state and
        decision, but the change energy makes to its output (fired/shut value minus
        its own value) is averaged over its column of `col` contiguous neurons, so
        one silenced or forced neuron moves the column by 1/col. Sites whose width
        is not a multiple of col are left per-neuron."""
        self.sites, self.decay_div, self.batch, self.col = sites, decay_div, batch, col
        self.enabled = True
        self.states: list[Tensor | None] = [None] * len(sites)
        self.doses: list[Tensor | None] = [None] * len(sites)
        self.fired: list[Tensor | None] = [None] * len(sites)  # running device counters
        self.intervened: list[Tensor | None] = [None] * len(sites)  # forced to fire or shut off
        self.mult = (1.0, 1.0, 1.0)
        # trace_sites: record, per call, the share of each position's neurons that energy
        # forced to fire or shut off (batch 1), in trace[site] (T,); None = off
        self.trace_sites: set[int] | None = None
        self.trace: dict[int, Tensor] = {}
        # training mode (begin_step): each call restarts from the step's saved
        # state with the step's noise, so a checkpointed layer's recompute in
        # backward gives exactly the forward's output
        self.step_seed: int | None = None
        self.snapshots: list[Tensor | None] = [None] * len(sites)
        self.handles = [s.module.register_forward_hook(self._hook(i)) for i, s in enumerate(sites)]

    def _dose_values(self, site: ScanSite) -> list[float]:
        mc, mn, md = self.mult
        cost = site.cost * mc
        drive = cost * site.mean * md
        expl = min(0.001, site.cost * site.mean / 4) * mn
        decay = drive / self.decay_div if self.decay_div else 0.0
        return [drive, cost, decay, expl]

    def set_dose(self, cost_scale: float = 1.0, noise_scale: float = 1.0, drive_scale: float = 1.0) -> None:
        self.mult = (cost_scale, noise_scale, drive_scale)
        for i, site in enumerate(self.sites):
            if self.doses[i] is not None:
                self.doses[i].copy_(torch.tensor(self._dose_values(site)))

    def begin_step(self, seed: int, at_rest: bool = True) -> None:
        """Training: start a step (one replayed window). at_rest: energy starts at 0."""
        self.step_seed = seed
        self.snapshots = [None] * len(self.sites)
        if at_rest:
            self.reset()

    def end_step(self) -> None:
        self.step_seed = None
        self.snapshots = [None] * len(self.sites)

    def reset(self) -> None:
        for st in self.states:
            if st is not None:
                st.zero_()

    def apply(self, i: int, h: Tensor) -> Tensor:
        """Energy of site i on h: (B, T, n), or (B, n) for one decode step."""
        site = self.sites[i]
        one_step = h.dim() == 2
        if one_step:
            h = h[:, None]
        b, t, n = h.shape
        if self.states[i] is None or self.states[i].numel() != b * n:
            self.states[i] = torch.zeros(b * n, device=h.device, dtype=torch.float32)  # at rest
            self.doses[i] = torch.tensor(self._dose_values(site), device=h.device, dtype=torch.float32)
            self.fired[i] = torch.zeros((), device=h.device, dtype=torch.float32)
            self.intervened[i] = torch.zeros((), device=h.device, dtype=torch.float32)
        x = h.transpose(0, 1).reshape(t, b * n)
        if self.step_seed is None:
            noise = torch.randn(t, b * n, device=h.device, dtype=torch.float32)
        else:
            if self.snapshots[i] is None:
                self.snapshots[i] = self.states[i].clone()
            else:
                self.states[i].copy_(self.snapshots[i])
            gen = torch.Generator(device=h.device).manual_seed(self.step_seed * 1000 + i)
            noise = torch.randn(t, b * n, device=h.device, dtype=torch.float32, generator=gen)
        gate, const = energy_scan(x.detach(), self.states[i], noise, self.doses[i], site.scale)
        if self.col > 1 and n % self.col == 0:
            delta = x * (gate - 1) + const  # energy's change to each neuron's output
            out = x + delta.reshape(t, b * n // self.col, self.col).mean(-1, keepdim=True).expand(
                t, b * n // self.col, self.col).reshape(t, b * n)
        else:
            out = x * gate + const
        self.fired[i].add_((gate == 0).sum())
        # forced to fire (const 2) or shut off while active; a held shut-off has const 0, so the
        # gate and the activity tell it apart from the dead zone
        hit = (const != 0) | ((gate == 0) & (x.detach().abs() / site.scale > ACTIVATION_THRESHOLD))
        self.intervened[i].add_(hit.sum())
        if self.trace_sites is not None and i in self.trace_sites and b == 1 and not one_step:
            self.trace[i] = hit.float().mean(-1).detach()
        out = out.reshape(t, b, n).transpose(0, 1)
        return out[:, 0] if one_step else out

    def _hook(self, i: int):
        def hook(_module, _args, output):
            if not self.enabled:
                return None
            h = output[0] if isinstance(output, tuple) else output
            if self.sites[i].flat_tokens and h.dim() == 2:
                out = self.apply(i, h[None])[0]
            else:
                out = self.apply(i, h)
            return (out, *output[1:]) if isinstance(output, tuple) else out

        return hook

    def site_index(self, module: nn.Module) -> int | None:
        for i, s in enumerate(self.sites):
            if s.module is module:
                return i
        return None

    def load_states(self, states: list, device) -> None:
        """Start from saved states (e.g. the actor's, `snapshot_states`) instead of
        rest; entries that are None or of the wrong size are left as they are."""
        for i, st in enumerate(states):
            if st is None:
                continue
            self.states[i] = st.to(device=device, dtype=torch.float32).clone()
            self.doses[i] = torch.tensor(self._dose_values(self.sites[i]), device=device, dtype=torch.float32)
            self.fired[i] = torch.zeros((), device=device, dtype=torch.float32)
            self.intervened[i] = torch.zeros((), device=device, dtype=torch.float32)

    def snapshot_states(self) -> list:
        """The current states on the CPU, for load_states."""
        return [None if st is None else st.detach().cpu().clone() for st in self.states]

    def intervened_count(self) -> float:
        """Entries forced to fire or shut off since the last call. Syncs."""
        total = sum(float(f) for f in self.intervened if f is not None)
        for f in self.intervened:
            if f is not None:
                f.zero_()
        return total

    def changed_count(self) -> float:
        """Entries changed (fired, shut off or dead-zoned) since the last call. Syncs."""
        total = sum(float(f) for f in self.fired if f is not None)
        for f in self.fired:
            if f is not None:
                f.zero_()
        return total

    def remove(self) -> None:
        for hd in self.handles:
            hd.remove()
