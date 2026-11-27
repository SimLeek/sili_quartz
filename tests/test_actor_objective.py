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


def test_input_tether_trains_the_state_or_head_and_never_the_target():
    from sili_quartz.actor_objective import InputTether

    torch.manual_seed(0)
    for mode in ("head", "col"):
        tet = InputTether(mode, d_model=16, bank=64)
        state = torch.randn(6, 16, requires_grad=True)
        target = torch.randn(6, 16, requires_grad=True)
        out = tet(state, target)
        out["tether_loss"].backward()
        assert state.grad is not None and torch.isfinite(state.grad).all() and state.grad[:-1].abs().sum() > 0
        assert target.grad is None  # detached: the input cannot move toward the prediction
        if mode == "head":
            assert tet.head.weight.grad is not None
        assert tet.targets.shape[0] == 5 and math.isfinite(out["tether_copy_mse"])


def test_input_tether_col_mode_matches_column_averages_and_leaves_the_rest_free():
    from sili_quartz.actor_objective import InputTether

    torch.manual_seed(1)
    inp = torch.randn(40, 16)
    reduce = InputTether("col", d_model=16, col_size=8).reduce
    cols = inp @ reduce  # (40, 2): the value each column's average must match
    # state_t: columns averaging to input_t+1's values, plus anything that averages to 0
    free = torch.randn(40, 2, 8)
    free = free - free.mean(-1, keepdim=True)
    nxt = torch.cat([cols[1:], torch.zeros(1, 2)])
    state = (nxt[:, :, None] + free).reshape(40, 16)
    assert InputTether("col", d_model=16, bank=1000, col_size=8)(state, inp)["tether_mse"] < 1e-6
    shuffled = inp[torch.randperm(40)]
    assert InputTether("col", d_model=16, bank=1000, col_size=8)(state, shuffled)["tether_mse"] > 0.1


def test_image_means_takes_each_turns_own_image_run():
    from sili_quartz.actor_objective import image_means

    IMG = 99
    ids = [1, IMG, IMG, 2, 5, 3, IMG, IMG, IMG, 2, 5]
    emb = torch.arange(len(ids), dtype=torch.float)[:, None].repeat(1, 2)
    m = image_means(ids, emb, IMG, states=[3, 9])
    assert m[:, 0].tolist() == [1.5, 7.0]


def test_two_sided_dual_pulls_back_from_above_the_ceiling_and_rests_in_the_band():
    dual = Dual(frac=0.8, eta=0.1, warmup=3, two_sided=True)
    for _ in range(3):
        dual.update(1.0)
    assert dual.floor == pytest.approx(0.8) and dual.ceiling == pytest.approx(1.25)
    for _ in range(5):
        dual.update(2.0)  # ran away upward
    assert dual.lam == 0.0 and dual.lam_hi > 0
    h = torch.tensor(2.0, requires_grad=True)
    dual.penalty(h).backward()
    assert h.grad > 0  # descending the penalty lowers the entropy
    for _ in range(200):
        dual.update(1.0)  # back inside the band: both multipliers decay to 0
    assert dual.lam == 0.0 and dual.lam_hi == 0.0
    assert float(dual.penalty(torch.tensor(1.0))) == 0.0


def _chunk_states(n_layers=4, chunk=2, T=30, d=6, seed=0):
    """hs with frames as single image tokens at even positions and the state
    after acting at odd positions; chunk outputs average to the next frame's
    chunk input, with free variation that averages to 0 across the chunk."""
    g = torch.Generator().manual_seed(seed)
    L = 2 * T
    hs = [torch.randn(1, L, d, generator=g) for _ in range(n_layers + 1)]
    for c in range(n_layers // chunk):
        lo = c * chunk
        for t in range(T - 1):
            nxt = hs[lo][0, 2 * (t + 1)]
            free = torch.randn(chunk, d, generator=g)
            free = free - free.mean(0)
            for k in range(chunk):
                hs[lo + k + 1][0, 2 * t + 1] = nxt + free[k]
    return hs, [2 * t + 1 for t in range(T)], [(2 * t, 2 * t + 1) for t in range(T)]


def test_chunk_tether_is_zero_when_each_chunks_layer_average_is_the_next_input():
    from sili_quartz.actor_objective import ChunkTether

    hs, after, runs = _chunk_states()
    out = ChunkTether(4, chunk=2)(hs, after, runs)
    assert out["tether_mse"] < 1e-6 and len(out["tether_chunk_mse"]) == 2
    hs2, _, _ = _chunk_states(seed=1)
    assert ChunkTether(4, chunk=2)(hs2[:1] + hs[1:], after, runs)["tether_mse"] > 0.1  # inputs changed


def test_chunk_tether_trains_the_layers_and_never_the_chunk_input():
    from sili_quartz.actor_objective import ChunkTether

    hs, after, runs = _chunk_states()
    hs = [h.clone().requires_grad_(True) for h in hs]
    ChunkTether(4, chunk=2)(hs, after, runs)["tether_loss"].backward()
    assert hs[1].grad is not None and hs[1].grad.abs().sum() > 0
    assert hs[0].grad is None or hs[0].grad.abs().sum() == 0  # chunk 0's input: detached target only


def test_input_share_is_one_for_a_copying_chunk_and_lower_otherwise():
    from sili_quartz.actor_objective import input_share

    x = torch.randn(1, 10, 8)
    assert torch.allclose(input_share([x, x, x], chunk=2), torch.ones(1))
    s = input_share([x, torch.randn(1, 10, 8), torch.randn(1, 10, 8)], chunk=2)
    assert 0 <= float(s) < 0.5


def test_row_norm_band_pulls_rows_back_into_the_band_and_cycles_layers():
    from sili_quartz.continual import RowNormBand

    torch.manual_seed(0)
    lins = {f"l{i}": torch.nn.Linear(4, 3) for i in range(4)}
    band = RowNormBand(lins, frac=0.8, per_call=2)
    with torch.no_grad():
        lins["l0"].weight[0] *= 3.0
        lins["l1"].weight[1] *= 0.1
        lins["l3"].weight[2] *= 5.0
    changed, rows = band.step()  # visits l0, l1
    assert changed == ["l0.weight", "l1.weight"] and rows == 2
    n0 = lins["l0"].weight.norm(dim=1)
    assert torch.allclose(n0[0], band.base["l0"][0] / 0.8)
    assert torch.allclose(lins["l1"].weight.norm(dim=1)[1], band.base["l1"][1] * 0.8)
    changed, rows = band.step()  # visits l2, l3
    assert changed == ["l3.weight"] and rows == 1
