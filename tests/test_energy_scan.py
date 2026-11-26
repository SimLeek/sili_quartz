import pytest
import torch
from torch import nn

from sili_quartz.energy import _apply_energy_dynamics
from sili_quartz.energy_scan import ScanEnergy, ScanSite, energy_scan, energy_scan_reference


def _inputs(T=7, M=50, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    h = (torch.randn(T, M, generator=g) * 1.5).to(device)
    noise = torch.randn(T, M, generator=g).to(device)
    e = (torch.rand(M, generator=g) * 3.6 - 1.8).to(device)
    dose = torch.tensor([0.4, 0.3, 0.01, 0.001], device=device)
    return h, e, noise, dose


def test_reference_matches_energy_py():
    h, e, noise, dose = _inputs()
    drive, cost, decay, expl = dose.tolist()
    e_ref = e.clone()
    gate, const = energy_scan_reference(h, e.clone(), noise, dose)
    out = h * gate + const
    for t in range(h.shape[0]):
        h_out, e_ref, *_ = _apply_energy_dynamics(
            h[t], e_ref, drive, cost, precision=0.04, density=0.5, p=1.0, exploration=expl,
            decay=decay, fire_reset_to_zero=True, noise=expl * noise[t])
        torch.testing.assert_close(out[t], h_out)


def test_scale_maps_site_to_unit_dose():
    h, e, noise, dose = _inputs()
    g1, c1 = energy_scan_reference(h, e.clone(), noise, dose, scale=1.0)
    g3, c3 = energy_scan_reference(h * 3, e.clone(), noise, dose, scale=3.0)
    torch.testing.assert_close(g1, g3)
    torch.testing.assert_close(c1 * 3, c3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_matches_reference():
    h, e, noise, dose = _inputs(T=33, M=3000, device="cuda")
    e1, e2 = e.clone(), e.clone()
    g1, c1 = energy_scan(h, e1, noise, dose)
    g2, c2 = energy_scan_reference(h, e2, noise, dose)
    torch.testing.assert_close(g1, g2)
    torch.testing.assert_close(c1, c2)
    torch.testing.assert_close(e1, e2)


def test_hook_state_carries_across_calls_like_one_call():
    """With exploration off, a sequence split over two calls (a decision, then the
    next) gives exactly what one call over the whole sequence gives."""
    torch.manual_seed(0)
    h = torch.randn(1, 10, 20) * 4
    outs = []
    for split in (None, 4):
        lin = nn.Identity()
        en = ScanEnergy([ScanSite(lin, mean=1.0, cost=0.5)])
        en.set_dose(noise_scale=0.0)
        outs.append(lin(h) if split is None else torch.cat([lin(h[:, :split]), lin(h[:, split:])], 1))
        en.remove()
    torch.testing.assert_close(outs[0], outs[1])
    assert not torch.equal(outs[0], h)  # energy did act


def test_set_dose_changes_live_dose_and_grad_passes_kept():
    lin = nn.Identity()
    en = ScanEnergy([ScanSite(lin, mean=1.0, cost=0.03)])
    h = (torch.randn(1, 5, 8) * 0.5).requires_grad_(True)
    lin(h).sum().backward()
    assert h.grad is not None and torch.all((h.grad == 0) | (h.grad == 1))
    en.set_dose(cost_scale=2.0)
    assert abs(float(en.doses[0][1]) - 0.06) < 1e-7
