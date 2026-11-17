import os

import numpy as np
import pytest
import torch

from sili_quartz.models import FAMILIES, TitansMAC, Transformer
from sili_quartz.models.common import recall_loss, segment_mask
from sili_quartz.tasks import RecallSpec, accuracy, constant_predictor_baseline, make_batch

DEV = os.environ.get("SQ_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def test_segment_mask_structure():
    m = segment_mask(3, 3, 2, "cpu")
    assert m.shape == (8, 8)
    assert not m[3:6, :3].any()  # chunk sees all prefix
    assert m[0, 1] and not m[1, 0]  # prefix causal among itself
    assert m[3, 4] and not m[4, 3]  # chunk causal
    assert not m[6:, :].any()  # suffix sees everything


@pytest.mark.parametrize("family", list(FAMILIES))
def test_forward_shapes(family):
    torch.manual_seed(0)
    model = FAMILIES[family](vocab=40, d=32, n_heads=4, chunk=8)
    toks = torch.randint(0, 40, (3, 21))  # not a multiple of chunk
    logits, state = model(toks)
    assert logits.shape == (3, 21, 40)
    assert (state is None) == (family == "transformer")


@pytest.mark.parametrize("family", ["rmt", "titans", "combined"])
def test_state_detached_at_chunk_boundary(family):
    model = FAMILIES[family](vocab=20, d=16, n_heads=2, chunk=4, tbptt_chunks=1)
    _, state = model(torch.randint(0, 20, (2, 8)))
    leaf = state.mem if family == "rmt" else (state.M if family == "titans" else state.mem.M)
    assert not leaf.requires_grad


@pytest.mark.parametrize("family", ["rmt", "titans", "combined"])
def test_state_carries_across_calls(family):
    torch.manual_seed(0)
    model = FAMILIES[family](vocab=20, d=16, n_heads=2, chunk=4)
    toks = torch.randint(0, 20, (2, 8))
    _, s1 = model(toks[:, :4])
    _, s2 = model(toks[:, 4:], s1)
    a, b = (
        (s1.mem, s2.mem) if family == "rmt" else (s1.M, s2.M) if family == "titans" else (s1.mem.M, s2.mem.M)
    )
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("family", ["rmt", "titans", "combined"])
def test_memory_write_changes_later_logits_only(family):
    # Writes must not leak into the chunk that produced them: same first chunk, different second.
    torch.manual_seed(0)
    model = FAMILIES[family](vocab=20, d=16, n_heads=2, chunk=4)
    a = torch.randint(0, 20, (1, 8))
    b = a.clone()
    b[:, 4:] = (b[:, 4:] + 1) % 20
    la, _ = model(a)
    lb, _ = model(b)
    assert torch.allclose(la[:, :4], lb[:, :4], atol=1e-6)


def _train(model, spec, steps, lr=3e-3, batch=32, seed=0):
    model = model.to(DEV)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        toks, tgts = make_batch(spec, batch, rng)
        logits, _ = model(toks.to(DEV))
        loss = recall_loss(logits, tgts.to(DEV))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return _evaluate(model, spec)


@torch.no_grad()
def _evaluate(model, spec, n=256):
    toks, tgts = make_batch(spec, n, np.random.default_rng(12345))
    logits, _ = model(toks.to(DEV))
    return accuracy(logits, tgts.to(DEV)), constant_predictor_baseline(tgts)


# Every switch combination must keep the old capability (beyond-window recall).
SWITCHES = {
    "base": {},
    "omega": {"window": 8},
}


@pytest.mark.integration
def test_learns_in_context_recall():
    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=0)
    model = Transformer(vocab=spec.vocab, d=64, n_heads=4, chunk=spec.length)
    acc, base = _train(model, spec, steps=300)
    assert acc > base + 0.3, (acc, base)


@pytest.mark.integration
def test_no_memory_cannot_do_long_gap():
    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=16)
    model = Transformer(vocab=spec.vocab, d=64, n_heads=4, chunk=8)
    acc, base = _train(model, spec, steps=200)
    assert acc < base + 0.15, (acc, base)  # leak check: attention alone must stay at chance


@pytest.mark.integration
@pytest.mark.parametrize("name", list(SWITCHES))
def test_memory_learns_long_gap_recall(name):
    """No regression: every switch setting still learns beyond-window recall."""
    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=16)
    # Verified config: write keyed on the previous token, backprop through the
    # memory across the whole episode. Without either, this sits near 0.2-0.3.
    model = TitansMAC(
        vocab=spec.vocab,
        d=64,
        n_heads=4,
        chunk=8,
        write_key_from_prev=True,
        tbptt_chunks=8,
        **SWITCHES[name],
    )
    acc, base = _train(model, spec, steps=800)
    assert acc > base + 0.5, (name, acc, base)


def _capacity(name, train_pairs=8, eval_pairs=32, steps=1500, seed=0):
    """Train once, then count bindings recovered at eval_pairs without retraining."""
    torch.manual_seed(seed)
    spec = RecallSpec(n_keys=64, n_values=64, n_pairs=train_pairs, n_queries=train_pairs, gap=16)
    model = TitansMAC(
        vocab=spec.vocab,
        d=64,
        n_heads=4,
        chunk=8,
        write_key_from_prev=True,
        tbptt_chunks=64,
        **SWITCHES[name],
    )
    acc_train, _ = _train(model, spec, steps=steps, batch=64)
    acc_eval, _ = _evaluate(
        model, RecallSpec(n_keys=64, n_values=64, n_pairs=eval_pairs, n_queries=8, gap=16)
    )
    print(f"[capacity] {name}: train_acc={acc_train:.3f}  recovered@{eval_pairs}={acc_eval * eval_pairs:.1f}")
    return acc_train, acc_eval * eval_pairs


@pytest.mark.integration
def test_omega_keeps_capacity():
    """Omega (window=chunk) must not cost bindings relative to the online rule."""
    base_train, base_rec = _capacity("base")
    omega_train, omega_rec = _capacity("omega")
    assert omega_train > 0.8, omega_train
    assert omega_rec > base_rec - 2, (base_rec, omega_rec)


# ---- family matrix: B, C, B+C must learn beyond the window; A must not ----


# RMT (B) measured 2026-11-15: 0.33 +- 0.01 beyond-window accuracy for m=4 and m=16, with
# and without a 0->4->8->16 gap curriculum; per-pair accuracy uniform (0.31-0.36) across
# write order. It carries the SET of written values but does not bind key->value. Strict
# xfail: if B ever clears the bar this fails loudly and the ledger must be updated.
RMT_XFAIL = pytest.mark.xfail(
    strict=True, reason="RMT plateaus at ~0.33: bag of values, no key->value binding"
)


@pytest.mark.integration
@pytest.mark.parametrize("family", [pytest.param("rmt", marks=RMT_XFAIL), "combined"])
def test_family_learns_long_gap_recall(family):
    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=16)
    model = FAMILIES[family](vocab=spec.vocab, d=64, n_heads=4, chunk=8, tbptt_chunks=64)
    acc, base = _train(model, spec, steps=800)
    assert acc > base + 0.5, (family, acc, base)


def _family_capacity(family, train_pairs=8, eval_pairs=32, steps=1500, seed=0):
    torch.manual_seed(seed)
    spec = RecallSpec(n_keys=64, n_values=64, n_pairs=train_pairs, n_queries=train_pairs, gap=16)
    model = FAMILIES[family](vocab=spec.vocab, d=64, n_heads=4, chunk=8, tbptt_chunks=64)
    acc_train, _ = _train(model, spec, steps=steps, batch=64)
    acc_eval, _ = _evaluate(
        model, RecallSpec(n_keys=64, n_values=64, n_pairs=eval_pairs, n_queries=8, gap=16)
    )
    rec = acc_eval * eval_pairs
    print(f"[family-capacity] {family}: train_acc={acc_train:.3f}  recovered@{eval_pairs}={rec:.1f}")
    return acc_train, acc_eval * eval_pairs


@pytest.mark.integration
@pytest.mark.parametrize("family", [pytest.param("rmt", marks=RMT_XFAIL), "titans", "combined"])
def test_family_capacity_probe(family):
    """Records bindings recovered per family (the ledger's B-vs-C question); only requires learning."""
    acc_train, _rec = _family_capacity(family)
    assert acc_train > 0.8, (family, acc_train)


# Ablations: each required setting, switched off alone, must break long-gap
# recall that the verified config (test_memory_learns_long_gap_recall) learns.
# Measured (seed 0, 800 steps, base 0.074): full config 1.000; same-token keys
# 0.383, detach every chunk 0.061, alpha 0.5 0.061. Same line as the positive test.
ABLATION_CEILING = 0.5
ABLATIONS = {
    "key_on_same_token": {"write_key_from_prev": False},  # ledger row 14
    "detach_every_chunk": {"tbptt_chunks": 1},  # row 15: no gradient across the gap
    "alpha_half": {"alpha": 0.5},  # row 24: random-init-like forget gate
}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(ABLATIONS))
def test_ablation_breaks_long_gap_recall(name):
    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=16)
    kw = dict(write_key_from_prev=True, tbptt_chunks=8) | ABLATIONS[name]
    model = TitansMAC(vocab=spec.vocab, d=64, n_heads=4, chunk=8, **kw)
    acc, base = _train(model, spec, steps=800)
    print(f"ablation {name}: acc {acc:.3f} base {base:.3f}")
    assert acc < base + ABLATION_CEILING, (name, acc, base)


def test_default_gates_are_not_random():
    """Row 24: gates come from config, not from the RNG, and do not forget."""
    mems = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        mems.append(TitansMAC(vocab=32, d=64, n_heads=4, chunk=8).memory)
    for m in mems:
        assert m.alpha == 0.0 and m.eta == 0.0, (m.alpha, m.eta)
    assert not any("alpha" in n or "eta" in n for n, _ in mems[0].named_parameters())
