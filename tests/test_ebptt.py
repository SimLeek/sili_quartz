"""e-BPTT fake backprop (models/ebptt.py): the mechanism, then ledger row 15."""

import numpy as np
import pytest
import torch

from sili_quartz.models import TitansMAC
from sili_quartz.models.ebptt import credit, substitute_saved
from sili_quartz.models.memory import TitansMemory


def _starmap(T, side, rng, stars=5):
    """T frames of a side x side sky, a few random unit stars each."""
    x = np.zeros((T, side * side), np.float32)
    for t in range(T):
        x[t, rng.choice(side * side, stars, replace=False)] = 1.0
    return torch.from_numpy(x)


def _blob_walk(T, side, rng, sigma=1.5):
    """A 2D gaussian blob stepping across the grid, z = t, plus gaussian noise."""
    yy, xx = np.mgrid[0:side, 0:side]
    c = rng.uniform(0, side, 2)
    v = rng.normal(0, 1, 2)
    out = []
    for _ in range(T):
        c = (c + v) % side
        f = np.exp(-((yy - c[0]) ** 2 + (xx - c[1]) ** 2) / (2 * sigma**2))
        out.append(f.ravel() + 0.05 * rng.normal(size=side * side))
    return torch.tensor(np.array(out), dtype=torch.float32)


PATTERNS = {
    "starmap": _starmap,
    "blob_walk": _blob_walk,
    "noise": lambda T, side, rng: torch.from_numpy(rng.normal(size=(T, side * side)).astype(np.float32)),
}


@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_substituted_saved_input_recomputes_weight_grad(pattern):
    """Replace a linear layer's saved input during backward: the weight grad is
    recomputed from the replacement, the input grad does not change."""
    rng = np.random.default_rng(0)
    side, T = 6, 10
    x = torch.randn(T, side * side, requires_grad=True)
    x_sub = PATTERNS[pattern](T, side, rng)
    lin = torch.nn.Linear(side * side, 8, bias=False)
    g = torch.randn(T, 8)

    y = lin(x)
    y.backward(g)
    gx_plain = x.grad.clone()

    lin.zero_grad()
    x.grad = None
    with substitute_saved(x, x_sub):
        y = lin(x)
    torch.testing.assert_close(y, x @ lin.weight.T)  # forward untouched
    y.backward(g)
    torch.testing.assert_close(lin.weight.grad, g.T @ x_sub)
    torch.testing.assert_close(x.grad, gx_plain)


def test_credit_node_is_identity_forward_and_routes_g():
    B, d, di = 2, 4, 3
    M = torch.randn(B, d, d)
    wv, wk = torch.randn(d, di, requires_grad=True), torch.randn(d, di, requires_grad=True)
    E_V, E_K = torch.randn(B, d, di), torch.randn(B, d, di)
    out = credit(M, wv, wk, E_V, E_K)
    torch.testing.assert_close(out, M)
    G = torch.randn(B, d, d)
    out.backward(G)
    torch.testing.assert_close(wv.grad, sum(G[b] @ E_V[b] for b in range(B)))
    torch.testing.assert_close(wk.grad, sum(G[b].T @ E_K[b] for b in range(B)))


@pytest.mark.parametrize("mode", ["trace", "buffer"])
def test_credit_matches_bptt_for_one_write(mode):
    """One write from M = 0, then a read across a detach: the W_V gradient from
    credit equals true backprop exactly (the identity-Jacobian approximation is
    exact when nothing else touched M). W_K differs only by the key
    normalizer's own gradient, which credit leaves out."""
    torch.manual_seed(0)
    B, d = 3, 8
    x, q = torch.randn(B, 1, d), torch.randn(B, 2, d)
    grads = {}
    for name, cr in (("bptt", "bptt"), ("credit", mode)):
        torch.manual_seed(1)
        mem = TitansMemory(d, credit=cr)
        st = mem.write(x, mem.init_state(B))
        if name == "credit":
            st = mem.with_credit(st)
        mem.read(q, st).pow(2).sum().backward()
        grads[name] = mem.W_V.weight.grad.clone()
    torch.testing.assert_close(grads["credit"], grads["bptt"])


@pytest.mark.parametrize("mode", ["trace", "buffer"])
def test_credit_matches_bptt_wv_over_many_writes(mode):
    """Several writes and windows in a row, keys detached from W_V: the carried
    trace is the exact W_V gradient, not an approximation."""
    torch.manual_seed(0)
    B, d, L = 2, 8, 6
    x, q = torch.randn(B, L, d), torch.randn(B, 3, d)
    grads = {}
    for cr in ("bptt", mode):
        torch.manual_seed(1)
        mem = TitansMemory(d, credit=cr, window=2, credit_buffer=L, credit_tol=0.0)
        st = mem.init_state(B)
        for s in range(0, L, 3):  # two write calls, three tokens each
            st = mem.write(x[:, s : s + 3], st)
        if cr != "bptt":
            st = mem.with_credit(st)
        mem.read(q, st).pow(2).sum().backward()
        grads[cr] = mem.W_V.weight.grad.clone()
    torch.testing.assert_close(grads[mode], grads["bptt"], rtol=1e-4, atol=1e-5)


def test_sparse_credit_matches_dense_on_kept_dims():
    """density < 1 computes the same contraction restricted to the top-|G| dims."""
    torch.manual_seed(0)
    B, d, di = 2, 8, 5
    M, E_V, E_K = torch.randn(B, d, d), torch.randn(B, d, di), torch.randn(B, d, di)
    G = torch.randn(B, d, d)
    wv, wk = torch.zeros(d, di, requires_grad=True), torch.zeros(d, di, requires_grad=True)
    credit(M, wv, wk, E_V, E_K, density=0.5).backward(G)
    ev, ek = torch.zeros(d, di), torch.zeros(d, di)
    for b in range(B):
        j = G[b].norm(dim=0).topk(4).indices
        i = G[b].norm(dim=1).topk(4).indices
        ev += G[b][:, j] @ E_V[b][j]
        ek += G[b][i].T @ E_K[b][i]
    torch.testing.assert_close(wv.grad, ev)
    torch.testing.assert_close(wk.grad, ek)
    wv.grad = wk.grad = None
    credit(M, wv, wk, E_V, E_K, density=1.0).backward(G)
    torch.testing.assert_close(wv.grad, sum(G[b] @ E_V[b] for b in range(B)))


@pytest.mark.parametrize("family", ["titans", "combined"])
def test_stream_backward_matches_full_backward(family):
    """Per-chunk backward gives the episode-summed loss's gradient exactly."""
    from sili_quartz.models import FAMILIES
    from sili_quartz.models.common import recall_loss
    from sili_quartz.models.ebptt import stream_backward
    from sili_quartz.tasks import make_batch

    spec = RecallSpec(n_pairs=2, n_queries=2, gap=8)
    toks, tgts = make_batch(spec, 4, np.random.default_rng(0))
    grads = []
    for mode in ("full", "stream"):
        torch.manual_seed(0)
        m = FAMILIES[family](vocab=spec.vocab, d=16, n_heads=2, chunk=4, tbptt_chunks=1,
                             credit="trace", credit_density=0.5, credit_score="noise")
        torch.manual_seed(1)  # same noise draws in both modes
        if mode == "full":
            recall_loss(m(toks)[0], tgts).backward()
        else:
            stream_backward(m, toks, tgts)
        grads.append({k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None})
    # a parameter reached only through a detached state has no grad in one
    # mode and an all-zero grad in the other; both mean zero
    for k in grads[0].keys() | grads[1].keys():
        g0, g1 = grads[0].get(k), grads[1].get(k)
        g0 = torch.zeros_like(g1) if g0 is None else g0
        g1 = torch.zeros_like(g0) if g1 is None else g1
        torch.testing.assert_close(g1, g0, rtol=1e-4, atol=1e-6, msg=k)


def test_buffer_keeps_last_nonzero_writes():
    torch.manual_seed(0)
    d, n = 4, 3
    mem = TitansMemory(d, credit="buffer", credit_buffer=n)
    st = mem.init_state(1)
    x = torch.randn(1, 6, d)
    x[:, 3] = 0.0  # a zero input writes nothing: must not take a slot
    st = mem.write(x, st)
    assert st.buf[1].shape == (1, n, d)
    torch.testing.assert_close(st.buf[1][0], x[0, [2, 4, 5]])


# --- ledger row 15: long-gap recall with the state detached every chunk ----

from sili_quartz.tasks import RecallSpec  # noqa: E402
from tests.test_mac import _train  # noqa: E402

PASS_MARGIN = 0.8  # same bar as test_memory_learns_long_gap_recall


@pytest.mark.integration
def test_ebptt_noise_sparse_per_chunk_backward_learns_long_gap():
    """Row 15 at constant memory: state detached every chunk, each chunk's loss
    backpropagated as soon as it ends, credit over a noise-chosen 1/4 of the
    memory dims (docs/results/2026-11-17-ebptt-credit-sweep.md)."""
    from sili_quartz.models.ebptt import stream_backward
    from sili_quartz.tasks import make_batch
    from tests.test_mac import DEV, _evaluate

    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=16)
    model = TitansMAC(vocab=spec.vocab, d=64, n_heads=4, chunk=8, tbptt_chunks=1,
                      credit="trace", credit_density=0.25, credit_score="noise").to(DEV)
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(1500):
        toks, tgts = make_batch(spec, 32, rng)
        opt.zero_grad()
        stream_backward(model, toks.to(DEV), tgts.to(DEV))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    acc, base = _evaluate(model, spec)
    print(f"ebptt noise_p25_stream: acc {acc:.3f} base {base:.3f}")
    assert acc > base + PASS_MARGIN, (acc, base)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["trace", "buffer"])
def test_ebptt_learns_long_gap_with_state_detached_every_chunk(mode):
    """Row 15 says gradient must reach the write across the gap; detaching
    every chunk drops recall to chance (test_ablation_breaks_long_gap_recall).
    e-BPTT keeps the detach and credits the write from remembered inputs."""
    torch.manual_seed(0)
    spec = RecallSpec(n_pairs=3, n_queries=3, gap=16)
    model = TitansMAC(vocab=spec.vocab, d=64, n_heads=4, chunk=8, tbptt_chunks=1, credit=mode)
    acc, base = _train(model, spec, steps=800)
    print(f"ebptt {mode}: acc {acc:.3f} base {base:.3f}")
    assert acc > base + PASS_MARGIN, (mode, acc, base)
