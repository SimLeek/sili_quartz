import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from sili_quartz.models import MemoryState, TitansMemory


def _mem(d=8, **kw):
    torch.manual_seed(0)
    return TitansMemory(d, **kw).double()


def test_one_shot_store():
    # From M=0 with alpha=eta=0 and theta=0.5/||k||^2, one write stores one pair exactly.
    m = _mem()
    x = torch.randn(3, 1, 8, dtype=torch.double)
    st0 = m.init_state(3, dtype=torch.double)
    st1 = m.write(x, st0)
    k, v = m.W_K(x[:, 0]), m.W_V(x[:, 0])
    assert torch.allclose(torch.einsum("bij,bj->bi", st1.M, k), v, atol=1e-9)


def test_write_is_an_outer_product_update():
    m = _mem()
    x = torch.randn(2, 1, 8, dtype=torch.double)
    st1 = m.write(x, m.init_state(2, dtype=torch.double))
    assert torch.linalg.matrix_rank(st1.M[0]).item() == 1


def test_momentum_and_forget_change_the_update():
    x = torch.randn(2, 4, 8, dtype=torch.double)
    base = _mem().write(x, _mem().init_state(2, dtype=torch.double)).M
    mom = _mem(eta=0.9).write(x, _mem(eta=0.9).init_state(2, dtype=torch.double)).M
    fgt = _mem(alpha=0.5).write(x, _mem(alpha=0.5).init_state(2, dtype=torch.double)).M
    assert not torch.allclose(base, mom) and not torch.allclose(base, fgt)


def test_write_then_read_gradcheck():
    m = _mem(d=5)
    x = torch.randn(2, 3, 5, dtype=torch.double, requires_grad=True)
    q = torch.randn(2, 5, dtype=torch.double, requires_grad=True)

    def f(x, q):
        st = m.write(x, m.init_state(2, dtype=torch.double))
        return m.read(q, st)

    assert torch.autograd.gradcheck(f, (x, q), eps=1e-6, atol=1e-5)


def test_detach_cuts_the_graph():
    m = _mem()
    x = torch.randn(2, 2, 8, dtype=torch.double, requires_grad=True)
    st = m.write(x, m.init_state(2, dtype=torch.double))
    assert st.M.requires_grad and not st.detach().M.requires_grad


def test_kernel_matched_gates_follow_law5():
    # alpha = 1/L, eta = 1 - 1/L, theta scaled by (1 - eta) = 1/L.
    m = _mem(gates="kernel_matched")
    st = m.init_state(2, dtype=torch.double)
    st.lag = torch.tensor([4.0, 10.0], dtype=torch.double)
    a, e, th = m._gates(st)
    assert torch.allclose(a.flatten(), torch.tensor([0.25, 0.1], dtype=torch.double))
    assert torch.allclose(e.flatten(), torch.tensor([0.75, 0.9], dtype=torch.double))
    assert torch.allclose(th.flatten(), torch.tensor([0.25, 0.1], dtype=torch.double))


def test_lag_estimate_tracks_true_lag():
    # Write a key at tick 0, pad 9 ticks, query with that key: measured lag -> 10.
    m = _mem(gates="kernel_matched", lag_beta=1.0)
    m.W_Q.weight.data.copy_(m.W_K.weight.data)  # estimator needs q/k alignment, as retrieval does
    x = torch.randn(1, 10, 8, dtype=torch.double)
    st = m.write(x, m.init_state(1, dtype=torch.double))
    m.read(x[:, 0], st)
    assert st.lag.item() == pytest.approx(10.0)


@settings(max_examples=25, deadline=None)
@given(B=st.integers(1, 4), L=st.integers(1, 6), d=st.integers(2, 12))
def test_shapes_and_finiteness(B, L, d):
    m = TitansMemory(d)
    x = torch.randn(B, L, d)
    st0 = m.init_state(B)
    st1 = m.write(x, st0)
    y = m.read(x, st1)
    assert st1.M.shape == (B, d, d) and y.shape == (B, L, d)
    assert torch.isfinite(st1.M).all() and torch.isfinite(y).all()


def test_state_is_per_batch_element():
    m = _mem()
    x = torch.randn(2, 1, 8, dtype=torch.double)
    st = m.write(x, m.init_state(2, dtype=torch.double))
    assert not torch.allclose(st.M[0], st.M[1])
    assert isinstance(st, MemoryState)


# ---- ATLAS switches: each must add its property without breaking the ones above ----


def _store_exact(m, x, k_src=None):
    st = m.write(x, m.init_state(x.shape[0], dtype=torch.double), k_src=k_src)
    k = m.phi(m.W_K(x if k_src is None else k_src))
    return torch.einsum("bij,bnj->bni", st.M, k), m.W_V(x)


def test_poly2_feature_dimension():
    from sili_quartz.models.memory import poly2

    k = torch.randn(2, 3, 8)
    assert poly2(k).shape == (2, 3, 8 + 36)
    m = TitansMemory(8, features="poly2")
    assert m.init_state(1).M.shape == (1, 8, 44)


def test_poly2_one_shot_store():
    m = _mem(features="poly2")
    got, want = _store_exact(m, torch.randn(3, 1, 8, dtype=torch.double))
    assert torch.allclose(got, want, atol=1e-9)


def test_window_one_matches_reference_delta_rule():
    # window=1 must reproduce the plain sequential 0.5/||k||^2 delta rule exactly.
    m = _mem()
    x = torch.randn(2, 5, 8, dtype=torch.double)
    st = m.write(x, m.init_state(2, dtype=torch.double))
    K, V = m.W_K(x), m.W_V(x)
    M = torch.zeros(2, 8, 8, dtype=torch.double)
    for t in range(5):
        k, v = K[:, t], V[:, t]
        err = torch.einsum("bij,bj->bi", M, k) - v
        M = M - (err.unsqueeze(-1) * k.unsqueeze(-2)) / k.pow(2).sum(-1).view(-1, 1, 1)
    assert torch.allclose(st.M, M, atol=1e-6)  # eps in the normalizer


def test_window_stores_orthogonal_keys_exactly_in_one_step():
    # Ledger row 1 generalized: a window of independent (here orthogonal) keys is stored exactly.
    m = _mem(window=4)
    m.W_K.weight.data.copy_(torch.eye(8, dtype=torch.double))
    x = torch.zeros(1, 4, 8, dtype=torch.double)
    for t in range(4):
        x[0, t, 2 * t] = 1.0 + t  # orthogonal keys of different norms
    got, want = _store_exact(m, x)
    assert torch.allclose(got, want, atol=1e-9)


def test_window_inner_steps_converge_for_correlated_keys():
    m1 = _mem(window=4, inner_steps=1)
    m8 = _mem(window=4, inner_steps=8)
    m8.load_state_dict(m1.state_dict())
    x = torch.randn(2, 4, 8, dtype=torch.double)
    e1 = (lambda g, w: (g - w).norm())(*_store_exact(m1, x))
    e8 = (lambda g, w: (g - w).norm())(*_store_exact(m8, x))
    assert e8 < 0.5 * e1


def test_poly2_window_gradcheck():
    m = _mem(d=4, features="poly2", window=2)
    x = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
    q = torch.randn(1, 4, dtype=torch.double, requires_grad=True)

    def f(x, q):
        return m.read(q, m.write(x, m.init_state(1, dtype=torch.double)))

    assert torch.autograd.gradcheck(f, (x, q), eps=1e-6, atol=1e-5)


def test_poly2_raises_independent_key_capacity():
    # d=6 identity memory cannot hold 12 random pairs; poly2 (d_feat=27) can, given inner steps.
    torch.manual_seed(1)
    x = torch.randn(1, 12, 6, dtype=torch.double)
    errs = {}
    for feat in ("identity", "poly2"):
        m = TitansMemory(6, features=feat, window=12, inner_steps=60).double()
        got, want = _store_exact(m, x)
        errs[feat] = ((got - want).norm() / want.norm()).item()
    assert errs["poly2"] < 0.1 < errs["identity"], errs
