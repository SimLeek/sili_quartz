"""Meters and dose controllers for energy (track P of docs/energy-controller).

Meters read the model; controllers turn meter readings into one dose level
and apply it to a NeuronEnergy through one knob:

- "cost": cost (drive follows, so energy stays neutral on average): fatigue
  and forced firing both scale.
- "noise": exploration noise only (random exploration, the x R variant).
- "drive": drive only, cost fixed (directed forced firing, the x D variant).

Every controller moves the level by x/÷ STEP per update, as Plappert et al.
2018 move their parameter-noise scale, and clamps it to [1/LEVEL_LIMIT,
LEVEL_LIMIT] as a guard rail.

| Name | Holds | Moved by |
|---|---|---|
| P1 | effect KL at a target delta | delta by surprise against tau (Mirostat-style integral) |
| P1a | effect KL at a fixed delta | nothing |
| P1b | surprise at tau, no effect meter | the dose directly |
| P2 | effect KL at delta | delta by LC engagement (two-timescale learning progress) |
| P2a | as P2, one timescale | |
| P2+A | as P2, delta's rate scaled by unsigned surprise (Pearce-Hall) | |

Mirostat (Basu et al. 2021, algorithm 1) is here too, as the P0b baseline.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import torch
from torch import Tensor

STEP = 1.01
LEVEL_LIMIT = 100.0
KNOBS = ("cost", "noise", "drive")


# -- meters -------------------------------------------------------------------


def effect_kl(logits_energy: Tensor, logits_clean: Tensor) -> Tensor:
    """KL(p_energy || p_clean) in nats over the last dim, per row."""
    lp = logits_energy.float().log_softmax(-1)
    lq = logits_clean.float().log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


class EffectMeter:
    """Effect of energy on next-step outputs, measured every `every` steps.

    `measure(energy_logits, clean_fn)`: clean_fn() returns the logits of the
    same step with energy off -- the caller decides how (e.g. a copy of the
    cache, inside `energy.disabled()`). Keeps the latest reading and the
    seconds each clean pass took.
    """

    def __init__(self, every: int = 1):
        self.every = max(1, int(every))
        self.steps = 0
        self.value: float | None = None
        self.seconds: list[float] = []

    def due(self) -> bool:
        return self.steps % self.every == 0

    def measure(self, energy_logits: Tensor, clean_fn) -> float | None:
        due = self.due()
        self.steps += 1
        if not due:
            return self.value
        t = time.perf_counter()
        with torch.no_grad():
            clean = clean_fn()
        self.value = float(effect_kl(energy_logits, clean).mean())
        self.seconds.append(time.perf_counter() - t)
        return self.value


class SurpriseMeter:
    """Running mean of -log p (nats) of the tokens the model itself chose,
    over about `window` tokens (exponential average)."""

    def __init__(self, window: int = 64):
        self.alpha = 1.0 / window
        self.mean: float | None = None
        self.last: float | None = None

    def update(self, logits: Tensor, tokens: Tensor) -> float:
        lp = logits.float().log_softmax(-1).gather(-1, tokens.reshape(-1, 1)).squeeze(-1)
        s = float(-lp.mean())
        self.last = s
        self.mean = s if self.mean is None else self.mean + self.alpha * (s - self.mean)
        return self.mean


class ProgressMeter:
    """Learning progress over a window w: mean loss over the w steps before
    the last w, minus mean loss over the last w (positive while the loss is
    falling). `utility(w)` divides that by its noise level, std / sqrt(w), so
    it is in noise units and needs no hand-set scale."""

    def __init__(self, windows: tuple[int, ...] = (16, 256)):
        self.windows = tuple(int(w) for w in windows)
        self.losses: deque[float] = deque(maxlen=2 * max(self.windows))

    def update(self, loss: float) -> None:
        self.losses.append(float(loss))

    def ready(self, w: int) -> bool:
        return len(self.losses) >= 2 * w

    def progress(self, w: int) -> float:
        if not self.ready(w):
            return 0.0
        recent = list(self.losses)[-2 * w :]
        return sum(recent[:w]) / w - sum(recent[w:]) / w

    def noise(self, w: int) -> float:
        """Per-step loss noise over the last 2w steps, from first differences
        (std of the steps / sqrt 2), so a steady trend doesn't count as noise."""
        recent = list(self.losses)[-2 * w :]
        diffs = [b - a for a, b in zip(recent, recent[1:])]
        if len(diffs) < 2:
            return 0.0
        mean = sum(diffs) / len(diffs)
        return math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1) / 2.0)

    def utility(self, w: int) -> float:
        if not self.ready(w):
            return 0.0
        std = self.noise(w)
        if not std > 0.0:  # zero, or nan from a bad loss
            return 0.0
        u = self.progress(w) / (std / math.sqrt(w))
        return u if math.isfinite(u) else 0.0


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def engagement(u_short: float, u_long: float) -> float:
    """Aston-Jones & Cohen 2005, eq. 1: (1 - logistic(u_s)) * logistic(u_l).

    u_short is the short-term utility *relative to the long-term trend*: a
    short-term drop (negative) raises engagement while long-term utility is
    high, which is how the paper reads the equation. Taken literally on raw
    utilities, steady progress on both windows would read as disengaged."""
    return (1.0 - logistic(u_short)) * logistic(u_long)


ENGAGEMENT_AT_ZERO = engagement(0.0, 0.0)  # 0.25: no progress on either window
# Progress counts as real at 2 noise units (about a one-sided 2-sigma test).
# Neutral engagement is the value there: steady real progress reads above it
# (focus), a loss floor reads below it (re-melt).
U_SIGNIFICANT = 2.0
ENGAGEMENT_NEUTRAL = engagement(0.0, U_SIGNIFICANT)  # about 0.44


# -- controllers ----------------------------------------------------------------


@dataclass
class Controller:
    """Base: a dose level, a knob, and `apply(energy)`."""

    knob: str = "cost"
    level: float = 1.0
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.knob not in KNOBS:
            raise ValueError(f"knob must be one of {KNOBS}, got {self.knob!r}")

    def _move(self, up: bool) -> None:
        self.level = self.level * STEP if up else self.level / STEP
        self.level = min(LEVEL_LIMIT, max(1.0 / LEVEL_LIMIT, self.level))

    def scales(self) -> dict:
        return {
            "cost_scale": self.level if self.knob == "cost" else 1.0,
            "noise_scale": self.level if self.knob == "noise" else 1.0,
            "drive_scale": self.level if self.knob == "drive" else 1.0,
        }

    def apply(self, energy) -> None:
        energy.set_dose(**self.scales())

    def _log(self, **values) -> None:
        self.history.append({"level": self.level, **values})


@dataclass
class EffectTarget(Controller):
    """P1a, and the inner loop of P1 and P2: hold the effect KL at delta."""

    delta: float = 0.01  # nats per token

    def hold(self, effect: float | None) -> None:
        if effect is not None and math.isfinite(effect):
            self._move(up=effect < self.delta)

    def update(self, effect: float | None, **_) -> None:
        self.hold(effect)
        self._log(effect=effect, delta=self.delta)


@dataclass
class P1(EffectTarget):
    """Effect held at delta; delta += eta * (tau - surprise), so a bored
    model (surprise under tau) gets more energy and a confused one less."""

    tau: float = 3.0  # nats
    eta: float = 1e-3

    def update(self, effect: float | None, surprise: float | None = None, **_) -> None:
        if surprise is not None and math.isfinite(surprise):
            self.delta = max(0.0, self.delta + self.eta * (self.tau - surprise))
        self.hold(effect)
        self._log(effect=effect, delta=self.delta, surprise=surprise)


@dataclass
class P1b(Controller):
    """Surprise held at tau by moving the dose directly (no effect meter)."""

    tau: float = 3.0

    def update(self, surprise: float | None = None, **_) -> None:
        if surprise is not None and math.isfinite(surprise):
            self._move(up=surprise < self.tau)
        self._log(surprise=surprise)


@dataclass
class P2(EffectTarget):
    """Effect held at delta; delta moved by LC engagement from learning
    progress: delta += eta * (E0 - engagement), E0 = ENGAGEMENT_NEUTRAL.
    Low engagement (no real progress) raises delta: the re-melt.

    windows: (short, long) for two timescales; one window = P2a, where
    engagement is logistic(u) against E0 = logistic(U_SIGNIFICANT).
    amygdala: P2+A, eta scaled by unsigned surprise, fast over slow average.
    """

    eta: float = 1e-3
    windows: tuple[int, ...] = (16, 256)
    amygdala: bool = False
    _abs_fast: float | None = None
    _abs_slow: float | None = None
    _surprise_mean: float | None = None

    def engagement(self, meter: ProgressMeter) -> tuple[float, float]:
        """(engagement, neutral value). Utility over w is progress in noise
        units, rate * w**1.5 / std, so the short window's rate relative to
        the long one, in short-window noise units, is u_s - u_l * (w_s/w_l)**1.5."""
        if len(self.windows) == 1:
            return logistic(meter.utility(self.windows[0])), logistic(U_SIGNIFICANT)
        short, long = self.windows
        u_long = meter.utility(long)
        u_rel = meter.utility(short) - u_long * (short / long) ** 1.5
        return engagement(u_rel, u_long), ENGAGEMENT_NEUTRAL

    def associability(self, surprise: float | None) -> float:
        """Pearce-Hall: recent unsigned surprise relative to its long average."""
        if surprise is None or not math.isfinite(surprise):
            return 1.0
        if self._surprise_mean is None:
            self._surprise_mean = surprise
        dev = abs(surprise - self._surprise_mean)
        self._surprise_mean += (surprise - self._surprise_mean) / self.windows[-1]
        self._abs_fast = dev if self._abs_fast is None else self._abs_fast + (dev - self._abs_fast) / self.windows[0]
        self._abs_slow = dev if self._abs_slow is None else self._abs_slow + (dev - self._abs_slow) / self.windows[-1]
        return self._abs_fast / self._abs_slow if self._abs_slow > 0 else 1.0

    def update(self, effect: float | None, progress: ProgressMeter | None = None,
               surprise: float | None = None, **_) -> None:
        rate = self.associability(surprise) if self.amygdala else 1.0
        e = e0 = None
        if progress is not None and progress.ready(self.windows[-1]):
            e, e0 = self.engagement(progress)
            self.delta = max(0.0, self.delta + self.eta * rate * (e0 - e))
        self.hold(effect)
        self._log(effect=effect, delta=self.delta, engagement=e, rate=rate)


def make_controller(name: str, knob: str = "cost", **kw) -> Controller:
    """P1, P1a, P1b, P2, P2a, P2+A; knob 'cost' | 'noise' | 'drive'."""
    if name == "P1":
        return P1(knob=knob, **kw)
    if name == "P1a":
        return EffectTarget(knob=knob, **kw)
    if name == "P1b":
        return P1b(knob=knob, **kw)
    if name == "P2":
        return P2(knob=knob, **kw)
    if name == "P2a":
        kw.setdefault("windows", (256,))
        return P2(knob=knob, **kw)
    if name == "P2+A":
        return P2(knob=knob, amygdala=True, **kw)
    raise ValueError(f"unknown controller {name!r}")


# -- Mirostat (P0b) ---------------------------------------------------------------


class Mirostat:
    """Basu et al. 2021, algorithm 1 (Mirostat v1). Surprise in bits, as in
    the paper. mu starts at 2 * tau; m tokens estimate the Zipf exponent."""

    def __init__(self, tau: float = 3.0, eta: float = 0.1, m: int = 100):
        self.tau, self.eta, self.m = tau, eta, m
        self.mu = 2.0 * tau
        self.surprises: list[float] = []

    def _zipf_s(self, probs: Tensor) -> float:
        p = probs[: self.m]
        i = torch.arange(1, len(p), dtype=torch.float64, device=p.device)
        t = torch.log((i + 1) / i)
        b = torch.log(p[:-1].double() / p[1:].double())
        return float((t * b).sum() / (t * t).sum())

    def sample(self, logits: Tensor, generator: torch.Generator | None = None) -> int:
        """One token from one row of logits (1-D)."""
        probs = logits.float().softmax(-1)
        sorted_p, order = probs.sort(descending=True)
        n = probs.numel()
        s = max(self._zipf_s(sorted_p), 1.0 + 1e-6)
        eps = s - 1.0
        k = ((eps * 2.0 ** self.mu) / (1.0 - n ** (-eps))) ** (1.0 / s)
        k = int(min(max(round(k), 1), n))
        top = sorted_p[:k] / sorted_p[:k].sum()
        j = int(torch.multinomial(top, 1, generator=generator))
        surprise = -math.log2(float(sorted_p[j]))  # original probability, as the paper's code
        self.surprises.append(surprise)
        self.mu -= self.eta * (surprise - self.tau)
        return int(order[j])
