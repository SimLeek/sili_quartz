"""Energy on MLP, attention and residual outputs (energy_hooks.py)."""

import os

import pytest
import torch
from torch import nn

from sili_quartz.energy_hooks import EnergySite, NeuronEnergy, mean_abs_outputs

DEVICE = os.environ.get("SQ_DEVICE", "cpu")
DIM = 24


class _TupleAttention(nn.Module):
    """Full attention returns (hidden, weights)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, x):
        return self.proj(x), "weights"


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _TupleAttention()
        self.gate = nn.Linear(DIM, DIM)
        self.act = nn.SiLU()
        self.down = nn.Linear(DIM, DIM)

    def forward(self, x):
        hidden, weights = self.attn(x)
        assert weights == "weights"
        x = x + hidden
        return x + self.down(self.act(self.gate(x)))


def _setup(seed=0):
    torch.manual_seed(seed)
    layer = _Layer().to(DEVICE)
    x = torch.randn(3, 20, DIM, device=DEVICE)
    modules = [layer.act, layer.attn, layer]  # mlp, attention, residual
    means = mean_abs_outputs(modules, lambda: layer(x))
    sites = [EnergySite(m, means[0], mean / means[0]) for m, mean in zip(modules, means, strict=True)]
    return layer, x, sites


def test_means_are_of_each_modules_output():
    layer, x, _ = _setup()
    got = mean_abs_outputs([layer.attn, layer], lambda: layer(x))
    assert got == pytest.approx([layer.attn(x)[0].abs().mean().item(), layer(x).abs().mean().item()])
    assert not layer._forward_hooks and not layer.attn._forward_hooks


@pytest.mark.parametrize("which", [[0], [1], [2], [0, 1, 2]], ids=["mlp", "attention", "residual", "all"])
def test_energy_changes_the_output_at_every_site(which):
    layer, x, sites = _setup()
    plain = layer(x)
    energy = NeuronEnergy([sites[i] for i in which], cost=2.0, decay_div=40)  # an event within 20 tokens
    assert not torch.allclose(layer(x), plain)
    assert energy.changed_fraction() > 0
    energy.remove()
    assert torch.equal(layer(x), plain)


def test_whole_sequence_equals_token_by_token():
    """Training sees a sequence, generation one token at a time: energy must not differ."""
    layer, x, sites = _setup()
    whole = NeuronEnergy(sites, cost=0.5, decay_div=40)
    at_once = layer(x)
    whole.remove()
    stepped = NeuronEnergy(sites, cost=0.5, decay_div=40)
    by_token = torch.cat([layer(x[:, t : t + 1]) for t in range(x.shape[1])], 1)
    stepped.remove()
    torch.testing.assert_close(at_once, by_token, atol=1e-4, rtol=1e-4)


def test_scale_makes_a_site_of_another_size_behave_the_same():
    small, big = nn.Identity(), nn.Identity()
    h = torch.randn(2, 30, DIM, device=DEVICE)
    a = NeuronEnergy([EnergySite(small, 0.8)], cost=0.5)
    b = NeuronEnergy([EnergySite(big, 0.8, scale=8.0)], cost=0.5)
    torch.testing.assert_close(small(h) * 8.0, big(h * 8.0), atol=1e-4, rtol=1e-4)
    assert a.changed_fraction() == pytest.approx(b.changed_fraction(), abs=0.01)


def test_disabled_is_identity_and_freezes_the_state():
    layer, x, sites = _setup()
    plain = layer(x)
    energy = NeuronEnergy(sites, cost=1.0)
    layer(x)
    before = [s[0].energy.clone() for s in energy.states]
    energy.enabled = False
    assert torch.equal(layer(x), plain)
    for old, state in zip(before, energy.states, strict=True):
        assert torch.equal(old, state[0].energy)


def test_parts_keep_separate_streams():
    layer, x, sites = _setup()
    energy = NeuronEnergy(sites, cost=0.5, parts=2)
    layer(x[:1])
    first = energy.states[0][0].energy.clone()
    energy.part = 1
    layer(x[1:])  # a different batch size must not reset part 0
    assert torch.equal(energy.states[0][0].energy, first)
    assert energy.states[0][1].energy.shape == (2, DIM)


def test_aux_reaches_the_weights_behind_attention_and_residual():
    layer, x, sites = _setup()
    energy = NeuronEnergy(sites[1:], cost=0.1, train=True)
    layer(x)
    torch.stack(energy.aux).sum().backward()
    assert layer.attn.proj.weight.grad.abs().sum() > 0
    assert layer.down.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("decay_div,reaches", [(4, False), (40, True), (0, True)])
def test_small_decay_div_lets_energy_settle_instead_of_building_up(decay_div, reaches):
    """A neuron 20% louder than average is inside the ±50% band of decay_div=4."""
    site = nn.Identity()
    energy = NeuronEnergy([EnergySite(site, 1.0)], cost=0.1, decay_div=decay_div)
    site(torch.full((1, 800, 4), 1.2, device=DEVICE))
    shut_off = (energy.states[0][0].energy <= -2.0).all()
    assert bool(shut_off) is reaches


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_scaled_site_keeps_the_model_dtype(dtype):
    site = nn.Identity()
    NeuronEnergy([EnergySite(site, 0.8, scale=0.1)], cost=0.1)
    assert site(torch.randn(2, 5, DIM, device=DEVICE, dtype=dtype)).dtype == dtype
