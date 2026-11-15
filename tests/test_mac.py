import os

import numpy as np
import pytest
import torch

from sili_quartz.tasks import RecallSpec, accuracy, constant_predictor_baseline, make_batch
from sili_quartz.titans import TitansMAC
from sili_quartz.titans.mac import _mac_mask, recall_loss

DEV = os.environ.get("SQ_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def test_mac_mask_structure():
    m = _mac_mask(3, "cpu")
    assert m.shape == (6, 6)
    assert not m[3:, :3].any()  # chunk sees all retrievals
    assert m[0, 1] and not m[1, 0]  # retrievals causal among themselves
    assert m[3, 4] and not m[4, 3]  # chunk causal


@pytest.mark.parametrize("memory", [True, False])
def test_forward_shapes(memory):
    torch.manual_seed(0)
    model = TitansMAC(vocab=40, d=32, n_heads=4, chunk=8, memory=memory)
    toks = torch.randint(0, 40, (3, 21))  # not a multiple of chunk
    logits, state = model(toks)
    assert logits.shape == (3, 21, 40)
    assert (state is None) == (not memory)


def test_state_detached_at_chunk_boundary():
    model = TitansMAC(vocab=20, d=16, n_heads=2, chunk=4, tbptt_chunks=1)
    _, state = model(torch.randint(0, 20, (2, 8)))
    assert not state.M.requires_grad


def test_state_carries_across_calls():
    torch.manual_seed(0)
    model = TitansMAC(vocab=20, d=16, n_heads=2, chunk=4)
    toks = torch.randint(0, 20, (2, 8))
    _, s1 = model(toks[:, :4])
    _, s2 = model(toks[:, 4:], s1)
    assert not torch.allclose(s1.M, s2.M)


def test_memory_write_changes_later_logits_only():
    # Writes must not leak into the chunk that produced them: same first chunk, different second.
    torch.manual_seed(0)
    model = TitansMAC(vocab=20, d=16, n_heads=2, chunk=4)
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
    "poly": {"features": "poly2"},
    "omega": {"window": 8},
    "both": {"features": "poly2", "window": 8},
}


@pytest.mark.integration
def test_learns_in_context_recall():
    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=0)
    model = TitansMAC(vocab=spec.vocab, d=64, n_heads=4, chunk=spec.length, memory=False)
    acc, base = _train(model, spec, steps=300)
    assert acc > base + 0.3, (acc, base)


@pytest.mark.integration
def test_no_memory_cannot_do_long_gap():
    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=16)
    model = TitansMAC(vocab=spec.vocab, d=64, n_heads=4, chunk=8, memory=False)
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
        memory=True,
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
        memory=True,
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
def test_poly2_raises_capacity():
    """New capability: poly2 recovers more bindings than the base matrix at the same d."""
    base_train, base_rec = _capacity("base")
    poly_train, poly_rec = _capacity("poly")
    assert base_train > 0.8 and poly_train > 0.8, (base_train, poly_train)
    assert poly_rec > base_rec + 4, (base_rec, poly_rec)


@pytest.mark.integration
def test_omega_keeps_capacity():
    """Omega (window=chunk) must not cost bindings relative to the online rule."""
    base_train, base_rec = _capacity("base")
    omega_train, omega_rec = _capacity("omega")
    assert omega_train > 0.8, omega_train
    assert omega_rec > base_rec - 2, (base_rec, omega_rec)
