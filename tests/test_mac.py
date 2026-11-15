import numpy as np
import pytest
import torch

from sili_quartz.tasks import RecallSpec, accuracy, constant_predictor_baseline, make_batch
from sili_quartz.titans import TitansMAC
from sili_quartz.titans.mac import _mac_mask, recall_loss


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
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        toks, tgts = make_batch(spec, batch, rng)
        logits, _ = model(toks)
        loss = recall_loss(logits, tgts)
        opt.zero_grad()
        loss.backward()
        opt.step()
    toks, tgts = make_batch(spec, 256, np.random.default_rng(12345))
    with torch.no_grad():
        logits, _ = model(toks)
    return accuracy(logits, tgts), constant_predictor_baseline(tgts)


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
def test_memory_learns_long_gap_recall():
    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=16)
    # Verified config: write keyed on the previous token, backprop through the
    # memory across the whole episode. Without either, this sits near 0.2-0.3.
    model = TitansMAC(
        vocab=spec.vocab, d=64, n_heads=4, chunk=8, memory=True, write_key_from_prev=True, tbptt_chunks=8
    )
    acc, base = _train(model, spec, steps=800)
    assert acc > base + 0.5, (acc, base)
