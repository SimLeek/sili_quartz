import time

import torch
from torch import nn

from sili_quartz.continual import ContinualLoRA, SyncApplier, WeightSync, direct_params, sync_tensors, target_linears


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


def _targets(m):
    return target_linears(m, r"^[02]$")


def test_adapter_starts_as_identity_and_trains_only_adapters():
    m = _model()
    for p in m.parameters():
        p.requires_grad_(False)
    x = torch.randn(4, 16)
    y0 = m(x)
    lora = ContinualLoRA(_targets(m), rank=4)
    assert torch.equal(m(x), y0)  # B = 0
    m(x).pow(2).sum().backward()
    assert all(p.grad is None for p in m.parameters())
    assert all(b.grad is not None for b in lora.B.values())


def test_merge_preserves_function_and_resets_adapter():
    m = _model()
    lora = ContinualLoRA(_targets(m), rank=4)
    with torch.no_grad():
        for b in lora.B.values():
            b.normal_(0, 0.1)
    x = torch.randn(4, 16)
    before = m(x)
    changed = lora.merge()
    assert sorted(changed) == ["0.weight", "2.weight"]
    assert all(not torch.any(b) for b in lora.B.values())
    torch.testing.assert_close(m(x), before, atol=1e-5, rtol=1e-5)
    assert lora.merge() == []  # nothing to merge after a reset


def test_sync_keeps_actor_equal_to_fp16_of_trainer(tmp_path):
    trainer = _model()
    actor = _model().half()
    actor.load_state_dict({k: v.half() for k, v in trainer.state_dict().items()})
    lora = ContinualLoRA(_targets(trainer), rank=4)
    direct = direct_params(trainer, _targets(trainer))
    sync = WeightSync(sync_tensors(_targets(trainer), direct), tmp_path)
    applier = SyncApplier(actor, tmp_path, per_call=1)
    try:
        for i in range(12):
            with torch.no_grad():
                # mix of tiny updates (below one fp16 step) and larger ones
                s = 1e-6 if i % 3 else 1e-2
                for b in lora.B.values():
                    b.normal_(0, s)
                for p in direct.values():
                    p.add_(torch.randn_like(p) * s)
            sync.write(lora.merge() + list(direct), meta={"merge": i})
        deadline = time.time() + 5
        while applier.next_index <= sync.count and time.time() < deadline:
            time.sleep(0.05)
        applier.drain()
        actor_p = dict(actor.named_parameters())
        for name, p in trainer.named_parameters():
            assert torch.equal(actor_p[name], p.detach().half()), name
        assert applier.applied_updates == sync.count
    finally:
        applier.stop()


def test_tiny_updates_accumulate_instead_of_vanishing(tmp_path):
    trainer = _model()
    lin = trainer[0]
    sync = WeightSync({"0.weight": lin.weight}, tmp_path)
    w16 = lin.weight.detach().half().clone()
    step = torch.full_like(lin.weight, 1e-5)  # far below fp16 resolution of |w| ~ 0.1
    for _ in range(200):
        with torch.no_grad():
            lin.weight.add_(step)
        sync.packet(["0.weight"])
    # naive per-update fp16 apply would have stayed at w16
    assert not torch.equal(sync.sent["0.weight"], w16)
    torch.testing.assert_close(sync.sent["0.weight"].float(), lin.weight.detach(), atol=1e-3, rtol=0)


def test_dense_packet_when_most_entries_change(tmp_path):
    m = _model()
    sync = WeightSync(sync_tensors(_targets(m)), tmp_path, dense_above=0.25)
    with torch.no_grad():
        m[0].weight.add_(1.0)
        m[2].weight[0, 0] += 1.0
    pkt = sync.packet()
    assert pkt["0.weight"][0] == "dense" and pkt["2.weight"][0] == "sparse"


def test_direct_params_cover_everything_but_adapted_weights():
    m = _model()
    lin = _targets(m)
    direct = direct_params(m, lin)
    assert sorted(direct) == ["0.bias", "2.bias"]
    assert len(sync_tensors(lin, direct)) == len(list(m.parameters()))
