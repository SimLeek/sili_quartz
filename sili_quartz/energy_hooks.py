"""Put EnergyDynamics on any module's output: MLP neurons, attention, residual.

`h` is (batch, tokens, neurons), or a tuple whose first item is that. Energy is
stepped over tokens, so a whole sequence at once (training) and token by token
(generation with a cache) give the same result.

Sites differ in size (a residual is about 1/3 of an MLP activation in
Qwen3.5-0.8B), and energy's constants are absolute (fire = 2, thresholds ±2).
So a site may have a `scale`: energy sees h / scale and its output is
multiplied back, which makes one cost mean the same dose at every site.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sili_quartz.energy import EnergyDynamics


@dataclass(frozen=True)
class EnergySite:
    module: nn.Module
    mean: float  # mean |h / scale| on ordinary input: energy is neutral there
    scale: float = 1.0
    cost: float | None = None  # this site's own cost, else NeuronEnergy's


def first_tensor(output) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def mean_abs_outputs(modules: Iterable[nn.Module], run) -> list[float]:
    """Mean |output| of each module during `run()`."""
    modules = list(modules)
    means = [0.0] * len(modules)

    def record(i):
        def hook(_module, _args, output):
            means[i] = first_tensor(output).float().abs().mean().item()

        return hook

    handles = [module.register_forward_hook(record(i)) for i, module in enumerate(modules)]
    try:
        run()
    finally:
        for handle in handles:
            handle.remove()
    return means


class NeuronEnergy:
    """One energy state per site, per stream (batch row), per part.

    `parts` > 1 is for backpropping a batch in pieces: set `part` before each.
    decay = drive / decay_div, 0 = none. The neurons that can never reach a
    threshold are those averaging within mean * (1 ± 2 / decay_div), so a small
    decay_div lets energy settle instead of building up.
    """

    def __init__(
        self,
        sites: list[EnergySite],
        cost: float,
        seed: int = 0,
        train: bool = False,
        parts: int = 1,
        decay_div: float = 4,
    ):
        self.cost, self.seed, self.train, self.decay_div = cost, seed, train, decay_div
        self.sites = sites
        # Dose multipliers a controller moves at run time (see set_dose).
        self.cost_scale = self.noise_scale = self.drive_scale = 1.0
        self.enabled = True
        self.part = 0
        self.aux: list[Tensor] = []
        self.changed: Tensor | float = 0.0
        self.count = 0
        self.states: list[list[EnergyDynamics | None]] = [[None] * parts for _ in sites]
        self.handles = [
            site.module.register_forward_hook(self._hook(i, site)) for i, site in enumerate(sites)
        ]

    def _dose(self, site: EnergySite) -> tuple[float, float, float, float]:
        """(cost, drive, exploration, decay) for a site at the current scales.
        At drive_scale 1 the drive is neutral on average (cost * mean)."""
        base = self.cost if site.cost is None else site.cost
        cost = base * self.cost_scale
        drive = cost * site.mean * self.drive_scale
        exploration = min(0.001, base * site.mean / 4) * self.noise_scale
        decay = drive / self.decay_div if self.decay_div else 0.0
        return cost, drive, exploration, decay

    def set_dose(self, cost_scale: float = 1.0, noise_scale: float = 1.0, drive_scale: float = 1.0) -> None:
        """Rescale every site's cost, noise (exploration) and drive, live
        states included. 1, 1, 1 is the dose the sites were built with."""
        self.cost_scale, self.noise_scale, self.drive_scale = cost_scale, noise_scale, drive_scale
        for site, states in zip(self.sites, self.states):
            cost, drive, exploration, decay = self._dose(site)
            for ed in states:
                if ed is not None:
                    ed.activation_cost, ed.drive, ed.exploration, ed.decay = cost, drive, exploration, decay

    @contextmanager
    def disabled(self):
        """Run without energy (and without touching its state), e.g. for the
        clean pass of the effect meter."""
        was, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = was

    def _new_state(self, index: int, site: EnergySite, h: Tensor) -> EnergyDynamics:
        cost, drive, exploration, decay = self._dose(site)
        ed = EnergyDynamics(
            drive,
            cost,
            precision=0.04,
            density=0.5,
            p=1.0,  # pretrained and dense: energy only fires and shuts off
            exploration=exploration,
            decay=decay,
            fire_reset_to_zero=True,
            rng=torch.Generator(h.device).manual_seed(self.seed + 1000 * self.part + index),
        )
        # start at rest: sili's near-threshold start fires a whole layer at once
        ed.energy = torch.zeros(h.shape, device=h.device)
        ed.steps_since_fired = torch.zeros(h.shape, dtype=torch.long, device=h.device)
        return ed

    def _hook(self, index: int, site: EnergySite):
        def hook(_module, _args, output):
            if not self.enabled:
                return None  # keep the module's output
            seq = first_tensor(output)
            ed = self.states[index][self.part]
            if ed is None:
                ed = self.states[index][self.part] = self._new_state(index, site, seq[:, 0])
            outs = []
            for t in range(seq.shape[1]):
                h = seq[:, t]
                out, aux, _ = ed(h if site.scale == 1.0 else h.float() / site.scale)
                if site.scale != 1.0:
                    out = (out * site.scale).to(h.dtype)
                if self.train:
                    self.aux.append(aux)
                self.changed = self.changed + ((out - h).detach().abs() > 1e-3 * site.scale).sum()
                self.count += h.numel()
                outs.append(out)
            stacked = torch.stack(outs, 1)
            return (stacked, *output[1:]) if isinstance(output, tuple) else stacked

        return hook

    def changed_fraction(self) -> float:
        fraction = float(self.changed / max(self.count, 1))
        self.changed, self.count = 0.0, 0
        return fraction

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
