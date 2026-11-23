"""EnergyDynamics torch port: parity with sili, and the properties the port must keep.

The parity tests need sili__new importable (PYTHONPATH=<dir containing sili/>)
and skip without it. SQ_DEVICE=cuda runs the torch side on a GPU.
"""

import os

import numpy as np
import pytest
import torch

from sili_quartz.energy import EnergyDynamics, abs_slope1_at_zero

DEVICE = os.environ.get("SQ_DEVICE", "cpu")
N = 128
BASE = {"drive": 0.15, "activation_cost": 0.08, "precision": 0.04, "density": 0.05, "p": 0.5}
CONFIGS = {
    "base": {},
    "low_p": {"p": 0.1, "density": 0.02},
    "p_is_one": {"p": 1.0},
    "decay": {"decay": 0.05},
    "reactive": {"reactivity": 0.5, "setpoint": 0.2},
    "fire_cost": {"fire_cost": 1.5},
    "fire_reset": {"p": 1.0, "fire_reset_to_zero": True},  # p=1: reset energies tie in top-p
    "wake_gradient": {"fire_wake_gradient": 0.3},
    "wake_gate": {"wake_gate_steps": 4},
    "wake_gate_staggered": {"wake_gate_steps": 6, "stagger_wake_init": True},
    "shutoff": {"p": 1.0, "drive": 0.02, "activation_cost": 1.0},
}


class _FixedNoise:
    """Stands in for numpy's Generator so sili gets the draw torch gets."""

    def __init__(self):
        self.value = None

    def normal(self, loc, scale, size):
        return self.value.reshape(size)

    def integers(self, low, high, size):  # staggered init: the draw torch makes from wake_seed=0
        return np.random.RandomState(0).randint(low, high, size)


def _inputs(kind, step, gen):
    h = gen.standard_normal(N).astype(np.float32)
    if kind == "zeros":
        h[:] = 0
    elif kind == "some_zeros":
        h[::4] = 0  # abs slope at exactly 0
    elif kind == "quiet_and_loud":
        h[: N // 2] *= 1e-5  # inside the dead zone: climbs to fire
        h[N // 2 :] *= 8  # drains to shutoff
    elif kind == "changing":
        h *= 0.05 if step % 10 < 5 else 3.0
    return h


@pytest.mark.parametrize("kind", ["gaussian", "zeros", "some_zeros", "quiet_and_loud", "changing"])
@pytest.mark.parametrize("name", CONFIGS)
def test_matches_sili_over_a_trajectory(name, kind):
    sili_energy = pytest.importorskip("sili.energy", reason="needs sili__new on PYTHONPATH")
    from sili.tensor import Tensor as SiliTensor

    cfg = BASE | CONFIGS[name]
    if kind == "zeros":
        cfg["p"] = 1.0  # all |h| tie, and ties may resolve differently
    fixed = _FixedNoise()
    theirs = sili_energy.EnergyDynamics(**cfg, rng=fixed)
    ours = EnergyDynamics(**cfg)
    gen = np.random.default_rng(0)

    for step in range(60):
        h = _inputs(kind, step, gen)
        noise = gen.normal(0, 0.001, N).astype(np.float32)
        weights = gen.standard_normal(N).astype(np.float32)
        fixed.value = noise

        hs = SiliTensor(h.copy())
        out_s, aux_s, p_s = theirs(hs)
        (aux_s + (out_s * weights).sum()).backward()

        ht = torch.tensor(h, device=DEVICE, requires_grad=True)
        out_t, aux_t, p_t = ours(ht, noise=torch.as_tensor(noise))
        (aux_t + (out_t * torch.as_tensor(weights, device=DEVICE)).sum()).backward()

        where = f"{name}/{kind} step {step}"
        np.testing.assert_allclose(ours.energy.cpu(), theirs.energy, rtol=0, atol=1e-6, err_msg=where)
        np.testing.assert_allclose(out_t.detach().cpu(), out_s.data, rtol=0, atol=1e-6, err_msg=where)
        np.testing.assert_allclose(ht.grad.cpu(), hs.grad, rtol=1e-5, atol=1e-6, err_msg=where)
        assert aux_t.item() == pytest.approx(float(np.asarray(aux_s.data).item()), rel=1e-5, abs=1e-6), where
        assert p_t == pytest.approx(p_s), where
        np.testing.assert_array_equal(ours.kept_indices.cpu(), theirs.kept_indices, err_msg=where)
        np.testing.assert_array_equal(ours.steps_since_fired.cpu(), theirs.steps_since_fired, err_msg=where)


def test_abs_has_slope_one_at_exactly_zero():
    x = torch.tensor([-2.0, -0.0, 0.0, 3.0], device=DEVICE, requires_grad=True)
    y = abs_slope1_at_zero(x)
    y.sum().backward()
    assert y.tolist() == [2.0, 0.0, 0.0, 3.0]
    assert x.grad.tolist() == [-1.0, 1.0, 1.0, 1.0]


def test_zero_activations_get_a_gradient():
    h = torch.zeros(N, device=DEVICE, requires_grad=True)
    _, aux, _ = EnergyDynamics(**BASE)(h)
    aux.backward()
    assert (h.grad != 0).all()


def test_quiet_neurons_fire_and_loud_neurons_shut_off():
    ed = EnergyDynamics(**(BASE | {"p": 1.0, "exploration": 0.0}))
    h = torch.zeros(N, device=DEVICE)
    h[N // 2 :] = 10.0
    fired = shut = False
    for _ in range(30):
        out, _, _ = ed(h)
        fired |= bool((out[: N // 2] == 2.0).all())
        shut |= bool((ed.energy[N // 2 :] == -2.0).all())
    assert fired and shut


def test_top_p_is_a_hard_ceiling():
    ed = EnergyDynamics(**(BASE | {"p": 0.1, "density": 0.02}))
    for _ in range(40):
        out, _, _ = ed(torch.randn(N, device=DEVICE))
        assert (out != 0).sum() <= round(0.1 * N)


def test_awake_overflow_does_not_poison_the_loss():
    ed = EnergyDynamics(**BASE, wake_gate_steps=5)
    h = torch.full((N,), 1e30, device=DEVICE, requires_grad=True)
    out, aux, _ = ed(h)
    aux.backward()
    assert torch.isfinite(aux) and torch.isfinite(h.grad).all()
    assert torch.equal(out, h)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_output_keeps_the_model_dtype_and_energy_stays_fp32(dtype):
    ed = EnergyDynamics(**BASE)
    out, aux, _ = ed(torch.randn(N, device=DEVICE, dtype=dtype))
    assert out.dtype == dtype and ed.energy.dtype == aux.dtype == torch.float32


def test_state_from_inference_mode_can_be_trained_on():
    ed = EnergyDynamics(**BASE)
    with torch.inference_mode():
        ed(torch.randn(N, device=DEVICE))
    h = torch.randn(N, device=DEVICE, requires_grad=True)
    _, aux, _ = ed(h)
    aux.backward()
    assert h.grad is not None and ed.aux_loss.grad_fn is None


def test_state_dict_round_trip():
    a, b = EnergyDynamics(**BASE), EnergyDynamics(**BASE)
    for _ in range(5):
        a(torch.randn(N, device=DEVICE))
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.energy, b.energy) and torch.equal(a.steps_since_fired, b.steps_since_fired)
