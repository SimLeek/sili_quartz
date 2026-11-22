"""Homeostatic neuron energy: a torch port of sili__new `sili/energy.py` EnergyDynamics.

Same names, order and semantics as sili; the design rationale for every
parameter is in sili__new `docs/research/energy.rst`. What had to change:

- `abs_slope1_at_zero`: sili's abs backward has slope 1 at exactly 0 (so a
  neuron can wake from zero), torch's has slope 0.
- `rng` is a torch.Generator, and `noise` can be passed in (the parity tests
  give sili and torch the same draw).
- Top-p ties can resolve differently: numpy argpartition vs torch.topk.

As in sili there is no batch dimension: the caller iterates batches and time.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn


def abs_slope1_at_zero(x: Tensor) -> Tensor:
    """|x| with gradient 1 at exactly 0, as in sili `cpu.py` abs_backward."""
    return torch.where(x == 0, x, x.abs())


def _apply_energy_dynamics(
    h: Tensor,
    energy: Tensor,
    drive: float,
    activation_cost: float,
    precision: float,
    density: float,
    exploration: float = 0.001,
    setpoint: float = 1.0,
    activation_threshold: float = 1e-4,
    reactivity: float = 0.01,
    p: float = 0.05,
    fire_reset_to_zero: bool = False,
    fire_cost: float | None = None,
    fire_wake_gradient: float | None = None,
    wake_sign: Tensor | None = None,
    rng: torch.Generator | None = None,
    wake_gate_steps: int | None = None,
    steps_since_fired: Tensor | None = None,
    decay: float = 0.0,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, float, Tensor, Tensor]:
    """Returns h_out, new_energy, aux_loss, actual_p, kept_indices, new_steps_since_fired."""
    n = h.numel()
    device = h.device
    h_np = h.detach().float().reshape(-1)  # decisions are made outside the graph
    energy_flat = energy.detach().to(device, torch.float32).reshape(-1).clone()

    # 1. dead zone
    alive = h_np.abs() > activation_threshold
    h_dz = h_np * alive

    # 2. energy update
    if noise is None:
        noise_device = rng.device if rng is not None else device
        noise = exploration * torch.randn(n, generator=rng, device=noise_device)
    noise = noise.to(device, torch.float32).reshape(-1)

    if steps_since_fired is None:
        ssf_flat = torch.zeros(n, dtype=torch.long, device=device)
    else:
        ssf_flat = steps_since_fired.to(device).reshape(-1)
    if wake_gate_steps is None:
        stale = torch.ones(n, dtype=torch.bool, device=device)
    else:
        stale = ssf_flat >= wake_gate_steps
    new_energy = energy_flat * (1.0 - decay) + drive + noise - activation_cost * h_dz.abs()

    # 3. hard thresholds (integrate and fire)
    fire_mask = (new_energy >= 2.0) & stale
    shutoff_mask = (new_energy <= -2.0) & stale
    shutoff_values = torch.where(shutoff_mask, energy_flat + 2.0, 0.0)
    new_energy = torch.where(shutoff_mask, -2.0, new_energy)
    if fire_reset_to_zero:
        fired_energy = torch.zeros_like(new_energy)
    elif fire_cost is not None:
        fired_energy = new_energy - float(fire_cost)
    else:
        fired_energy = new_energy - 2.0 * activation_cost
    new_energy = torch.where(fire_mask, fired_energy, new_energy)
    new_energy = torch.where(stale, new_energy, energy_flat)  # awake neurons are frozen

    # 4. hard-ceiling top-p gate: fired by energy first, then the others by |h|
    n_stale = int(stale.sum())
    k = max(1, round(p * n_stale)) if n_stale > 0 else 0
    fire_idx = fire_mask.nonzero().squeeze(1)
    non_fire_idx = (stale & ~fire_mask).nonzero().squeeze(1)
    n_fired = len(fire_idx)

    kept_nfire = non_fire_idx[:0]
    if n_fired >= k:
        kept_fire = fire_idx[new_energy[fire_idx].topk(k, sorted=False).indices]
    else:
        kept_fire = fire_idx
        fill_k = min(k - n_fired, len(non_fire_idx))
        if fill_k > 0:
            scores = h_dz[non_fire_idx].abs()
            kept_nfire = non_fire_idx[scores.topk(fill_k, sorted=False).indices]

    kept_fire_mask = torch.zeros(n, dtype=torch.bool, device=device)
    kept_fire_mask[kept_fire] = True
    kept_nfire_mask = torch.zeros(n, dtype=torch.bool, device=device)
    kept_nfire_mask[kept_nfire] = True
    shutoff_in_kept = kept_nfire_mask & shutoff_mask
    normal_kept = kept_nfire_mask & ~shutoff_mask & alive

    # 5. h_out: kept -> h, fired/shutoff -> constant, suppressed -> 0, awake -> h
    gate = (normal_kept | ~stale).to(h.dtype)
    const = torch.where(kept_fire_mask, 2.0, torch.where(shutoff_in_kept, shutoff_values, 0.0))
    h_out = h * gate.reshape(h.shape) + const.to(h.dtype).reshape(h.shape)

    # 6. aux_loss
    n_active = int(
        normal_kept.sum()
        + kept_fire_mask.sum()
        + (shutoff_in_kept & (const.abs() > activation_threshold)).sum()
    )
    actual_p = n_active / n_stale if n_stale > 0 else 0.0
    rho = min(max(actual_p, 1e-5), 1.0 - 1e-5)
    off = (1.0 - rho) * math.log((1.0 - rho) / (1.0 - density))
    kl_val = precision * (rho * math.log(rho / density) + off)

    # indexing, not masking: an overflowed awake |h| would give inf * 0 = nan
    stale_idx = stale.nonzero().squeeze(1)
    h_flat = h.reshape(-1).float()
    c_stale = (energy_flat + drive + noise)[stale_idx]
    new_energy_t_stale = c_stale - activation_cost * abs_slope1_at_zero(h_flat[stale_idx])
    energy_loss = (reactivity / 2.0) * ((new_energy_t_stale - setpoint) ** 2).sum()
    aux_loss = kl_val + energy_loss

    if fire_wake_gradient is not None and len(kept_fire) > 0:
        assert wake_sign is not None, "wake_sign is required when fire_wake_gradient is set"
        signs = wake_sign.to(device, torch.float32).reshape(-1)[kept_fire]
        aux_loss = aux_loss + (h_flat[kept_fire] * (fire_wake_gradient * signs)).sum()

    kept_indices = (normal_kept | kept_fire_mask | shutoff_in_kept).nonzero().squeeze(1)
    new_ssf = torch.where(kept_fire_mask, 0, ssf_flat + 1)
    return (
        h_out,
        new_energy.reshape(h.shape),
        aux_loss,
        actual_p,
        kept_indices,
        new_ssf.reshape(h.shape),
    )


class EnergyDynamics(nn.Module):
    """Per-region homeostatic energy. Nothing here is trained by the optimizer.

    >>> ed = EnergyDynamics(drive=0.15, activation_cost=0.08, precision=0.04, density=0.25, p=0.5)
    >>> h_out, aux_loss, actual_p = ed(h)
    >>> (task_loss(h_out) + aux_loss).backward()
    """

    def __init__(
        self,
        drive: float,
        activation_cost: float,
        precision: float,
        density: float,
        exploration: float = 0.001,
        setpoint: float = 1.0,
        activation_threshold: float = 1e-4,
        reactivity: float = 0.01,
        p: float = 0.05,
        fire_reset_to_zero: bool = False,
        fire_cost: float | None = None,
        fire_wake_gradient: float | None = None,
        wake_seed: int = 0,
        rng: torch.Generator | None = None,
        wake_gate_steps: int | None = None,
        stagger_wake_init: bool = False,
        decay: float = 0.0,
    ):
        super().__init__()
        assert np.finfo(np.float32).eps * 2 <= activation_cost <= 4.0, (
            "activation_cost (gamma) must be positive and <= 4.0"
        )
        assert 0.0 < density < 1.0, "density (beta) must be in (0, 1)"
        assert 0.0 < p <= 1.0, "p must be in (0, 1]"
        assert decay >= 0.0, "decay must be non-negative"
        if decay > 0.0:
            assert drive > 2.0 * decay, "drive must exceed 2*decay or a quiet neuron can never fire"
        assert density <= p * 0.8, "density must stay below 0.8*p: p is a ceiling, not the sparsity target"

        self._energy_start = max(0.0, 2.0 - drive * 10)
        self.drive = float(drive)
        self.activation_cost = float(activation_cost)
        self.precision = float(precision)
        self.density = float(density)
        self.exploration = float(exploration)
        self.setpoint = float(setpoint)
        self.activation_threshold = float(activation_threshold)
        self.reactivity = float(reactivity)
        self.p = float(p)
        self.fire_reset_to_zero = bool(fire_reset_to_zero)
        self.fire_cost = None if fire_cost is None else float(fire_cost)
        self.fire_wake_gradient = None if fire_wake_gradient is None else float(fire_wake_gradient)
        self.wake_seed = int(wake_seed)
        self.rng = rng
        self.wake_gate_steps = None if wake_gate_steps is None else int(wake_gate_steps)
        self.stagger_wake_init = bool(stagger_wake_init)
        self.decay = float(decay)

        # running state, shaped by the first h it sees
        self.energy: Tensor | None = None
        self.steps_since_fired: Tensor | None = None
        self._wake_sign: Tensor | None = None

        self.aux_loss: Tensor | None = None
        self.actual_p: float = 0.0
        self.kept_indices: Tensor | None = None

    def get_extra_state(self) -> dict:
        return {"energy": self.energy, "steps_since_fired": self.steps_since_fired}

    def set_extra_state(self, state: dict) -> None:
        self.energy = state["energy"]
        self.steps_since_fired = state["steps_since_fired"]

    def forward(
        self, h: Tensor, density_override: float | None = None, noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor, float]:
        if self.energy is None or self.energy.shape != h.shape:
            self.energy = torch.full(h.shape, self._energy_start, dtype=torch.float32, device=h.device)
            self._wake_sign = None
            if self.wake_gate_steps is not None and self.stagger_wake_init:
                if self.rng is not None:
                    ssf = torch.randint(
                        self.wake_gate_steps, h.shape, generator=self.rng, device=self.rng.device
                    )
                else:
                    draw = np.random.RandomState(self.wake_seed).randint(0, self.wake_gate_steps, h.shape)
                    ssf = torch.as_tensor(draw, dtype=torch.long)
                self.steps_since_fired = ssf.to(h.device)
            else:
                self.steps_since_fired = torch.zeros(h.shape, dtype=torch.long, device=h.device)

        if self.fire_wake_gradient is not None and self._wake_sign is None:
            signs = np.random.RandomState(self.wake_seed).choice([-1.0, 1.0], size=h.numel())
            self._wake_sign = torch.as_tensor(signs, dtype=torch.float32, device=h.device)

        density = self.density if density_override is None else float(density_override)
        (h_out, self.energy, aux_loss, self.actual_p, self.kept_indices, self.steps_since_fired) = (
            _apply_energy_dynamics(
                h,
                self.energy,
                self.drive,
                self.activation_cost,
                self.precision,
                density,
                self.exploration,
                self.setpoint,
                self.activation_threshold,
                self.reactivity,
                self.p,
                self.fire_reset_to_zero,
                self.fire_cost,
                self.fire_wake_gradient,
                self._wake_sign,
                self.rng,
                self.wake_gate_steps,
                self.steps_since_fired,
                self.decay,
                noise,
            )
        )
        self.aux_loss = aux_loss.detach()  # for logging; holding the graph would leak it across steps
        return h_out, aux_loss, self.actual_p
