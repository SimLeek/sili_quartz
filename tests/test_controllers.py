"""Meters and dose controllers (controllers.py), and NeuronEnergy's live dose."""

import math
import os
import random

import pytest
import torch
from torch import nn

from sili_quartz.controllers import (
    ENGAGEMENT_AT_ZERO, ENGAGEMENT_NEUTRAL, LEVEL_LIMIT, EffectMeter, Mirostat, ProgressMeter, SurpriseMeter,
    effect_kl, engagement, make_controller,
)
from sili_quartz.energy_hooks import EnergySite, NeuronEnergy, mean_abs_outputs

DEVICE = os.environ.get("SQ_DEVICE", "cpu")
DIM, VOCAB = 24, 50


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(DIM, DIM)
        self.act = nn.SiLU()
        self.out = nn.Linear(DIM, VOCAB)

    def forward(self, x):
        return self.out(self.act(self.inp(x)))


def _setup(cost, seed=0, tokens=30):
    torch.manual_seed(seed)
    model = _TinyLM().to(DEVICE)
    x = torch.randn(4, tokens, DIM, device=DEVICE)
    (mean,) = mean_abs_outputs([model.act], lambda: model(x))
    energy = NeuronEnergy([EnergySite(model.act, mean)], cost=cost, seed=seed)
    return model, x, energy


# -- P.1 effect meter ---------------------------------------------------------------


def test_effect_kl_is_zero_for_identical_logits_and_positive_otherwise():
    a = torch.randn(5, VOCAB)
    assert torch.allclose(effect_kl(a, a), torch.zeros(5), atol=1e-6)
    assert (effect_kl(a, a + torch.randn(5, VOCAB)) > 0).all()


def test_effect_is_zero_with_energy_disabled():
    model, x, energy = _setup(cost=0.5)
    with energy.disabled():
        on = model(x)
    off = model(x)  # energy enabled, but not yet run, so check the context restored it
    assert energy.enabled
    with energy.disabled(), torch.no_grad():
        assert float(effect_kl(on, model(x)).max()) == pytest.approx(0.0, abs=1e-6)
    assert not torch.allclose(on, off)  # sanity: energy does change the output


def test_effect_rises_with_cost():
    kls = []
    for cost in (0.5, 1.0, 2.0):  # energy needs time to reach its thresholds: long sequence
        model, x, energy = _setup(cost=cost, tokens=200)
        with torch.no_grad():
            on = model(x)
            with energy.disabled():
                off = model(x)
        kls.append(float(effect_kl(on, off).mean()))
        energy.remove()
    assert kls[0] < kls[1] < kls[2], kls


def test_effect_meter_measures_every_n_steps_and_times_the_clean_pass():
    meter = EffectMeter(every=3)
    a = torch.randn(2, VOCAB)
    calls = []

    def clean():
        calls.append(1)
        return a + 1.0

    readings = [meter.measure(a, clean) for _ in range(7)]
    assert len(calls) == 3 and len(meter.seconds) == 3
    assert readings[0] == pytest.approx(0.0, abs=1e-6)  # a + const has the same softmax
    assert all(r is not None for r in readings)


def test_set_dose_rescales_live_states():
    model, x, energy = _setup(cost=0.2)
    model(x)
    ed = energy.states[0][0]
    cost0, drive0, expl0 = ed.activation_cost, ed.drive, ed.exploration
    energy.set_dose(cost_scale=2.0)
    assert ed.activation_cost == pytest.approx(2 * cost0) and ed.drive == pytest.approx(2 * drive0)
    assert ed.exploration == pytest.approx(expl0)
    energy.set_dose(noise_scale=3.0)
    assert ed.activation_cost == pytest.approx(cost0) and ed.exploration == pytest.approx(3 * expl0)
    energy.set_dose(drive_scale=1.5)
    assert ed.activation_cost == pytest.approx(cost0) and ed.drive == pytest.approx(1.5 * drive0)
    assert ed.decay == pytest.approx(ed.drive / energy.decay_div)


# -- P.2 surprise and learning progress ------------------------------------------------


def test_surprise_meter_tracks_the_chosen_tokens_neg_log_prob():
    meter = SurpriseMeter(window=1)  # no averaging
    logits = torch.log(torch.tensor([[0.5, 0.25, 0.25]]))
    assert meter.update(logits, torch.tensor([0])) == pytest.approx(math.log(2))
    assert meter.update(logits, torch.tensor([1])) == pytest.approx(math.log(4))


def test_surprise_meter_averages_over_its_window():
    meter = SurpriseMeter(window=10)
    logits = torch.log(torch.tensor([[0.5, 0.5]]))
    for _ in range(200):
        meter.update(logits, torch.tensor([0]))
    assert meter.mean == pytest.approx(math.log(2))


def test_progress_is_positive_while_loss_falls_and_zero_when_flat():
    rng = random.Random(0)
    meter = ProgressMeter(windows=(8, 32))
    for t in range(64):
        meter.update(3.0 - 0.02 * t + 0.01 * rng.gauss(0, 1))
    assert meter.progress(8) > 0 and meter.progress(32) > 0
    assert meter.utility(32) > 3  # many noise units
    for _ in range(64):
        meter.update(1.0 + 0.01 * rng.gauss(0, 1))
    assert abs(meter.utility(32)) < 3


def test_progress_is_zero_until_two_windows_are_full():
    meter = ProgressMeter(windows=(4,))
    for _ in range(7):
        meter.update(1.0)
    assert not meter.ready(4) and meter.utility(4) == 0.0


# -- P.3 controllers ------------------------------------------------------------------


def test_engagement_matches_aston_jones_cohen():
    assert ENGAGEMENT_AT_ZERO == pytest.approx(0.25)
    assert engagement(-3, 3) > engagement(0, 3) > engagement(0, -3)
    assert ENGAGEMENT_AT_ZERO < ENGAGEMENT_NEUTRAL < engagement(0, 5)


@pytest.mark.parametrize("knob", ["cost", "noise", "drive"])
def test_p1b_raises_the_dose_when_bored_and_lowers_it_when_confused(knob):
    c = make_controller("P1b", knob=knob, tau=3.0)
    for _ in range(50):
        c.update(surprise=1.0)
    assert c.level > 1.5
    scales = c.scales()
    assert scales[f"{knob}_scale"] == c.level and sum(v != 1.0 for v in scales.values()) == 1
    for _ in range(150):
        c.update(surprise=6.0)
    assert c.level < 1.0


def test_p1_target_rises_when_bored_and_falls_when_confused():
    c = make_controller("P1", delta=0.01, tau=3.0, eta=1e-3)
    for _ in range(20):
        c.update(effect=0.0, surprise=1.0)
    assert c.delta > 0.01 and c.level > 1.0  # effect under target -> more energy
    d = c.delta
    for _ in range(100):
        c.update(effect=1.0, surprise=6.0)
    assert c.delta < d and c.level < 1.0
    assert c.delta >= 0.0


def test_p1a_holds_the_effect_at_a_fixed_target():
    c = make_controller("P1a", delta=0.05)
    level = 1.0
    for _ in range(300):  # a fake plant: effect proportional to the dose level
        c.update(effect=0.05 * c.level / 2.0)
        level = c.level
    assert level == pytest.approx(2.0, rel=0.05)
    assert c.delta == 0.05


def test_levels_stay_bounded_and_finite_on_extreme_or_bad_input():
    for name in ("P1", "P1a", "P1b", "P2", "P2a", "P2+A"):
        c = make_controller(name)
        meter = ProgressMeter(windows=(4, 16))
        for t in range(3000):
            meter.update(float("nan") if t == 5 else 0.0)
            c.update(effect=0.0 if t % 7 else float("nan"), surprise=-1e9 if t % 2 else float("inf"),
                     progress=meter)
        assert 1 / LEVEL_LIMIT <= c.level <= LEVEL_LIMIT and math.isfinite(c.level), name
        assert math.isfinite(getattr(c, "delta", 0.0)), name


def _p2_run(name):
    """Loss falls for 600 steps, then sits on a floor. Effect tracks the dose."""
    rng = random.Random(1)
    c = make_controller(name, delta=0.05, eta=1e-3, windows=(16, 128) if name != "P2a" else (128,))
    meter = ProgressMeter(windows=c.windows)
    deltas = []
    for t in range(1800):
        loss = (3.0 - 0.004 * t if t < 600 else 0.6) + 0.02 * rng.gauss(0, 1)
        meter.update(loss)
        c.update(effect=0.05 * c.level, progress=meter, surprise=2.0 + 0.1 * rng.gauss(0, 1))
        deltas.append(c.delta)
    return deltas


@pytest.mark.parametrize("name", ["P2", "P2a", "P2+A"])
def test_p2_lowers_the_target_while_learning_and_re_melts_on_a_loss_floor(name):
    deltas = _p2_run(name)
    assert deltas[590] < deltas[300]  # progressing: focus (target falls)
    assert deltas[-1] > deltas[700]  # floor: re-melt (target rises)


def test_p2_plus_a_rate_rises_with_a_surprise_burst():
    c = make_controller("P2+A")
    rates = []
    for t in range(600):
        rates.append(c.associability(2.0 if t < 400 else (2.0 if t % 2 else 8.0)))
    assert max(rates[400:]) > 2 * (sum(rates[300:400]) / 100)


def test_unknown_controller_or_knob_is_an_error():
    with pytest.raises(ValueError):
        make_controller("P9")
    with pytest.raises(ValueError):
        make_controller("P1", knob="volume")


# -- P.4 Mirostat ------------------------------------------------------------------------


def test_mirostat_surprise_tracks_tau_on_a_zipf_distribution():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    ranks = torch.arange(1, 5001, dtype=torch.float64)
    logits = (-1.1 * torch.log(ranks)).float()  # Zipf, s = 1.1
    m = Mirostat(tau=3.0, eta=0.1)
    for _ in range(3000):
        m.sample(logits, generator=gen)
    late = m.surprises[1000:]
    assert sum(late) / len(late) == pytest.approx(3.0, abs=0.3)


def test_mirostat_zipf_estimate_recovers_the_exponent():
    ranks = torch.arange(1, 1001, dtype=torch.float64)
    p = ranks ** -1.3
    p = (p / p.sum()).float()
    assert Mirostat()._zipf_s(p) == pytest.approx(1.3, abs=0.01)
