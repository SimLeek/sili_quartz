import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from sili_quartz.titans import MemoryState, TitansMemory


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
