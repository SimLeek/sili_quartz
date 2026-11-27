"""Continual LoRA training merged into the weights, and its sync to a live actor (track A, 2.4a).

Trainer side (fp32, any GPU):
- `ContinualLoRA` adds a rank-r adapter to each target `nn.Linear` with a
  forward hook (no module replacement, so the model's own code is untouched).
- `merge()` folds `scale * B @ A` into the weight, sets B = 0 and redraws A
  (ReLoRA-style), so the adapter is only a cheap way to compute an update:
  the model itself is what changes.
- `direct_params` lists everything else (norms, convolutions, gates, position
  embeddings, biases): trained directly, synced the same way.
- `WeightSync` turns each merge into a packet for the actor: per matrix, the
  entries whose fp16 value changed and their new fp16 values. The trainer
  keeps the fp16 copy it last sent, so the actor's fp16 weights always equal
  fp16(trainer's fp32 weights) exactly. Updates too small to change an fp16
  value accumulate in fp32 until they do, instead of being lost to rounding
  on the actor.

Actor side (fp16, the GPU that plays):
- `SyncApplier` loads packets from a directory in a background thread and
  writes a few matrices per call in place (`W.view(-1)[idx] = vals`), so play
  never stalls and CUDA graphs that read the weights stay valid.

Packets are files `update_000001.pt`, ... written atomically (tmp + rename).
"""

from __future__ import annotations

import math
import queue
import re
import threading
from pathlib import Path

import torch
from torch import nn

ALL_LINEARS = r".*"
# The language model's projections only (testing; the training runs use ALL_LINEARS).
LM_TARGETS = (r"language_model\.layers\.\d+\."
              r"(mlp\.(gate|up|down)_proj|self_attn\.[qkvo]_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj))$")


def target_linears(model: nn.Module, pattern: str = ALL_LINEARS) -> dict[str, nn.Linear]:
    rx = re.compile(pattern)
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and rx.search(n)}


def direct_params(model: nn.Module, linears: dict[str, nn.Linear]) -> dict[str, nn.Parameter]:
    """Every parameter not covered by a LoRA'd Linear (norms, convolutions, gates,
    position embeddings): trained directly and synced as weight diffs. A weight
    tied to an adapted Linear (the token embedding tied to lm_head) is left out:
    the merge already changes it."""
    covered = {id(m.weight) for m in linears.values()} | {id(m.bias) for m in linears.values() if m.bias is not None}
    out = {}
    for n, p in model.named_parameters():
        if id(p) not in covered:
            out[n] = p
    # biases of adapted Linears are trained directly too (LoRA only covers the weight)
    for n, m in linears.items():
        if m.bias is not None:
            out[n + ".bias"] = m.bias
    return out


def sync_tensors(linears: dict[str, nn.Linear], direct: dict[str, nn.Parameter] | None = None) -> dict:
    """Parameter name -> tensor, for WeightSync: adapted weights plus direct params."""
    out = {n + ".weight": m.weight for n, m in linears.items()}
    out.update(direct or {})
    return out


class ContinualLoRA(nn.Module):
    """Rank-r adapters on `targets` (name -> Linear), merged into the weights on `merge()`."""

    def __init__(self, targets: dict[str, nn.Linear], rank: int = 8, alpha: float = 16.0, seed: int = 0):
        super().__init__()
        self.rank, self.scale = rank, alpha / rank
        self.targets = targets
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        self._hooks = []
        for name, lin in targets.items():
            key = name.replace(".", "_")
            w = lin.weight
            self.A[key] = nn.Parameter(torch.empty(rank, w.shape[1], device=w.device, dtype=torch.float32))
            self.B[key] = nn.Parameter(torch.zeros(w.shape[0], rank, device=w.device, dtype=torch.float32))
            self._init_A(key)
            self._hooks.append(lin.register_forward_hook(self._hook(key)))
        self.merges = 0

    def _init_A(self, key: str) -> None:
        a = self.A[key]
        bound = 1.0 / math.sqrt(a.shape[1])  # kaiming-uniform(a=sqrt(5)) bound, as nn.Linear / LoRA
        a.data.copy_((torch.rand(a.shape, generator=self.gen) * 2 - 1).mul_(bound).to(a.device))

    def _hook(self, key: str):
        def fn(mod, inp, out):
            x = inp[0]
            delta = (x.to(torch.float32) @ self.A[key].t()) @ self.B[key].t()
            return out + (self.scale * delta).to(out.dtype)
        return fn

    @torch.no_grad()
    def merge(self) -> list[str]:
        """W += scale * B @ A for every adapter with a nonzero B; reset B = 0, redraw A.
        Returns the parameter names ("<linear>.weight") whose weights changed."""
        changed = []
        for name, lin in self.targets.items():
            key = name.replace(".", "_")
            b = self.B[key]
            if not torch.any(b):
                continue
            lin.weight.add_((self.scale * (b @ self.A[key])).to(lin.weight.dtype))
            b.zero_()
            self._init_A(key)
            changed.append(name + ".weight")
        self.merges += 1
        return changed

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def reset_adam_state(opt: torch.optim.Optimizer, params) -> None:
    """Drop optimizer state for `params` (ReLoRA resets it at each merge: the
    moments belong to the old A, B)."""
    for p in params:
        opt.state.pop(p, None)


class RowNormBand:
    """Amortized magnitude health: each Linear's row norms stay within
    [frac, 1/frac] x their starting values (a band, as the health floors are
    frac x the early value). `step()` visits `per_call` layers in turn and
    rescales rows outside the band to its edge; direction is left alone."""

    def __init__(self, linears: dict[str, nn.Linear], frac: float = 0.8, per_call: int | None = None):
        self.lin, self.names, self.frac = linears, list(linears), frac
        self.base = {n: m.weight.detach().float().norm(dim=1) for n, m in linears.items()}
        self.per_call = per_call or max(1, len(self.names) // 8)
        self.cursor = 0

    @torch.no_grad()
    def step(self) -> tuple[list[str], int]:
        """Returns the parameter names changed and the number of rows rescaled."""
        changed, rows = [], 0
        for _ in range(min(self.per_call, len(self.names))):
            n = self.names[self.cursor]
            self.cursor = (self.cursor + 1) % len(self.names)
            w, b = self.lin[n].weight, self.base[n]
            norm = w.float().norm(dim=1)
            target = torch.minimum(torch.maximum(norm, b * self.frac), b / self.frac)
            scale = torch.where(b > 0, target / norm.clamp(min=1e-12), torch.ones_like(norm))
            moved = (scale - 1).abs() > 1e-6
            if moved.any():
                w.mul_(scale[:, None].to(w.dtype))
                changed.append(n + ".weight")
                rows += int(moved.sum())
        return changed, rows


@torch.no_grad()
def load_checkpoint(model: nn.Module, path) -> int:
    """Load a trainer checkpoint (parameter name -> tensor, as the trainer saves
    every --checkpoint-every seconds) into `model` in place, in its own dtypes.
    Returns the number of tensors loaded; raises on a name the model lacks."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters(remove_duplicate=False))
    missing = [n for n in sd if n not in params]
    if missing:
        raise KeyError(f"{len(missing)} checkpoint tensors not in the model, e.g. {missing[:3]}")
    for n, t in sd.items():
        params[n].copy_(t.to(params[n].device, params[n].dtype))
    return len(sd)


class WeightSync:
    """Trainer side: per merge, a packet of the fp16 entries that changed."""

    def __init__(self, tensors: dict[str, torch.Tensor], out_dir: Path, dense_above: float = 0.25):
        """tensors: parameter name (as in the actor's named_parameters) -> trainer tensor."""
        self.tensors = tensors
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dense_above = dense_above
        self.sent = {n: t.detach().to(torch.float16).clone() for n, t in tensors.items()}
        self.count = 0

    @torch.no_grad()
    def packet(self, names=None) -> dict:
        pkt = {}
        for n in names if names is not None else self.tensors:
            new16 = self.tensors[n].detach().to(torch.float16)
            flat_new, flat_old = new16.view(-1), self.sent[n].view(-1)
            idx = (flat_new != flat_old).nonzero().view(-1)
            if idx.numel() == 0:
                continue
            if idx.numel() > self.dense_above * flat_new.numel():
                pkt[n] = ("dense", new16.cpu())
            else:
                pkt[n] = ("sparse", idx.to(torch.int32).cpu(), flat_new[idx].cpu())
            self.sent[n].copy_(new16)
        return pkt

    def write(self, names=None, meta: dict | None = None) -> Path | None:
        pkt = self.packet(names)
        if not pkt:
            return None
        self.count += 1
        path = self.out / f"update_{self.count:06d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save({"meta": meta or {}, "weights": pkt}, tmp)
        tmp.rename(path)
        return path


class SyncApplier:
    """Actor side: apply packets from `sync_dir` in place, `per_call` matrices per `step()`."""

    def __init__(self, model: nn.Module, sync_dir: Path, per_call: int = 4, on_applied=None,
                 delete_applied: bool = False):
        """on_applied(name): called after each tensor is written (e.g. to refresh copies)."""
        # remove_duplicate=False: a tied weight is reachable under every name
        self.params = dict(model.named_parameters(remove_duplicate=False))
        self.on_applied = on_applied
        self.delete_applied = delete_applied
        self._current: int | None = None
        self.dir = Path(sync_dir)
        self.per_call = per_call
        self.next_index = 1
        self.pending: list = []  # (name, entry)
        self.loaded: queue.Queue = queue.Queue()
        self.applied_updates = 0
        self.applied_matrices = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loader, daemon=True)
        self._thread.start()

    def _loader(self) -> None:
        while not self._stop.is_set():
            path = self.dir / f"update_{self.next_index:06d}.pt"
            if path.exists():
                pkt = torch.load(path, map_location="cpu", weights_only=False)
                if self.delete_applied:
                    path.unlink()  # loaded into memory; the trainer keeps periodic checkpoints
                self.loaded.put((self.next_index, pkt))
                self.next_index += 1
            else:
                self._stop.wait(0.5)

    @torch.no_grad()
    def step(self) -> int:
        """Apply up to per_call matrices; returns how many were applied."""
        if not self.pending:
            try:
                _, pkt = self.loaded.get_nowait()
            except queue.Empty:
                return 0
            self.pending = list(pkt["weights"].items())
            self.applied_updates += 1
        n = 0
        while self.pending and n < self.per_call:
            name, entry = self.pending.pop(0)
            w = self.params[name]
            if entry[0] == "dense":
                w.copy_(entry[1].to(w.device, w.dtype))
            else:
                _, idx, vals = entry
                w.view(-1)[idx.to(w.device, torch.long)] = vals.to(w.device, w.dtype)
            if self.on_applied is not None:
                self.on_applied(name)
            n += 1
        self.applied_matrices += n
        return n

    def drain(self) -> None:
        """Apply everything loaded so far (tests, shutdown)."""
        while self.step():
            pass

    def stop(self) -> None:
        self._stop.set()
