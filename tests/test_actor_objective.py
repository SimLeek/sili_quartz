import math
import os

import pytest
import torch

from sili_quartz.actor_objective import (
    Dual, PredictiveInfo, gaussian_bc, participation_ratio, pointwise_mi, turn_positions,
)


def _correlated(n=4000, d=4, rho=0.8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = rho * x + math.sqrt(1 - rho**2) * torch.randn(n, d, generator=g)
    return x, y, -0.5 * d * math.log(1 - rho**2)


def test_gaussian_bc_recovers_known_mi_and_is_differentiable():
    x, y, true = _correlated()
    x.requires_grad_(True)
    mi = gaussian_bc(x, y)
    assert abs(float(mi.detach()) - true) < 0.05
    mi.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_pointwise_mi_averages_to_mi():
    x, y, true = _correlated()
    pmi = pointwise_mi(x, y, x, y)
    assert abs(float(pmi.mean()) - true) < 0.1


def test_participation_ratio_isotropic_vs_rank_one():
    g = torch.Generator().manual_seed(0)
    iso = torch.randn(2000, 16, generator=g)
    rank1 = torch.randn(2000, 1, generator=g) * torch.ones(1, 16)
    assert float(participation_ratio(iso)) > 14
    assert float(participation_ratio(rank1)) < 1.01


def test_dual_rises_only_below_floor():
    dual = Dual(frac=0.8, eta=0.1, warmup=3)
    for _ in range(3):
        dual.update(10.0)
    assert dual.floor == pytest.approx(8.0)
    dual.update(9.0)
    assert dual.lam == 0.0
    dual.update(4.0)
    assert dual.lam == pytest.approx(0.05)
    for _ in range(20):
        dual.update(12.0)
    assert dual.lam == 0.0


@pytest.mark.parametrize("arm,dx", [("A1", 64), ("A1-act", 32), ("A2", 32), ("A3", 32)])
def test_predictive_info_pairs_and_bank(arm, dx):
    pi = PredictiveInfo(arm, d_model=16, d=32, bank=512)
    feats = {k: torch.randn(6, 16) for k in ("s", "a", "rA", "rB")}
    out = pi(feats)
    assert out["mi"] is None and out["pairs"] == 5
    for _ in range(60):
        out = pi({k: torch.randn(6, 16) for k in ("s", "a", "rA", "rB")})
    assert out["mi"] is not None and out["reward"].shape == (5,)
    assert pi._project(pi.bank)[0].shape[1] == dx


def test_noisy_code_bounds_mi_for_a_static_stream():
    pi = PredictiveInfo("A1-act", d_model=16, d=8, bank=2048)
    for _ in range(200):
        v = torch.randn(1, 16).repeat(6, 1)  # identical states within a window
        out = pi({k: v for k in ("s", "a", "rA", "rB")})
    assert float(out["mi"]) <= 8 * 0.5 * math.log(2) + 0.3


MODEL = os.path.expanduser("~/models/GameScaling-0.8B-work")


@pytest.mark.skipif(not os.path.isdir(MODEL), reason="needs the GameScaling tokenizer")
def test_turn_positions_on_rendered_conversation():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = [{"role": "system", "content": "sys"}]
    acts = ["<|action_start|>-12 5 0 ; W ; W ; ; ; ; LB<|action_end|>",
            "<|action_start|>3 0 0 ; ; ; ; ; ; <|action_end|>"]
    for act in acts:
        msgs += [{"role": "user", "content": "current game screen"}, {"role": "assistant", "content": act}]
    ids = tok.apply_chat_template(msgs, tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    turns = turn_positions(list(ids), tok, prefill="<|action_start|")
    assert len(turns) == 2
    for t, act in zip(turns, acts):
        assert tok.decode([ids[t["state"]]]) == "<|im_end|>"
        assert ("<|action_start|" + tok.decode([ids[i] for i in t["action"]])) == act


def _out_energy(n_choice: int):
    from sili_quartz.energy_scan import ScanEnergy, ScanSite

    energy = ScanEnergy([ScanSite(torch.nn.Identity(), 1.0, 1.0 / n_choice, 0.0009)])
    energy.out_index = 0
    return energy


def test_output_energy_passes_at_dose_zero_and_keeps_gradient():
    from sili_quartz.actor_objective import output_energy

    energy = _out_energy(4)
    energy.set_dose(cost_scale=0.0, noise_scale=0.0)
    logits = torch.randn(5, 6, requires_grad=True)
    cols = torch.tensor([1, 2, 3, 4])
    out = output_energy(energy, logits, cols)
    assert torch.allclose(out.log_softmax(-1), logits.log_softmax(-1), atol=1e-5)
    out.log_softmax(-1)[:, 2].sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_output_energy_shuts_off_an_always_chosen_token():
    """A collapsed policy (one choice token at p ~ 1 every step) loses that token
    once its energy is drained; format columns outside `cols` are untouched."""
    from sili_quartz.actor_objective import output_energy

    energy = _out_energy(4)
    energy.set_dose(cost_scale=50.0, noise_scale=0.0)
    logits = torch.full((1, 6), -10.0)
    logits[0, 1] = 10.0  # the collapsed choice
    logits[0, 0] = 0.0  # a format token, not in cols
    cols = torch.tensor([1, 2, 3, 4])
    probs = [output_energy(energy, logits, cols).softmax(-1)[0] for _ in range(60)]
    assert probs[0][1] > 0.99
    assert probs[-1][1] < 0.01  # shut off
    assert probs[-1][0] > 0.9  # the mass goes to what is left; format token logit unchanged
    assert torch.isclose(output_energy(energy, logits, cols)[0, 0], logits[0, 0])


def test_flat_token_site_steps_over_tokens_not_rows():
    from sili_quartz.energy_scan import ScanEnergy, ScanSite

    mod = torch.nn.Identity()
    energy = ScanEnergy([ScanSite(mod, 1.0, 1.0, 0.03, flat_tokens=True)])
    energy.set_dose(cost_scale=1.0, noise_scale=0.0)
    out = mod(torch.ones(7, 3))  # (tokens, n), e.g. the vision merger
    assert out.shape == (7, 3)
    assert energy.states[0].numel() == 3  # one state per neuron, not per (token, neuron)
