"""Model collapse on two GPUs, with and without neuron energy (TODO 2c).

A small copy of the published method, for "some model" (any local Qwen3.5 dir):

- Shumailov et al. 2024: the teacher continues 64-token blocks of real text by
  64 tokens, and the next model is trained on those continuations.
- Drayson et al. 2025: loss on the continuation only, AdamW lr 1e-5, batch 8,
  one epoch per generation.
- Forgetting is the rise in perplexity on fixed real test text, energy off.

Each generation the teacher (GPU 1, fp16) writes, the student (GPU 0, fp32)
trains, then the teacher takes the student's weights. Every generation uses
fresh blocks of real text as prompts (and as the real control's data): a few
hundred reused blocks overfit, which looks like forgetting. The student is
never reset, so this measures forgetting as well as collapse. With energy, both
models run it on every layer's MLP neurons, and their energy accumulates,
never reset within an arm.

    python examples/model_collapse.py --model DIR --data DIR --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from sili_quartz.energy import EnergyDynamics

BLOCK = 64  # prompt tokens, and continuation tokens
BATCH = 8
MICRO = 4  # the P40 backprops half a batch at a time: same gradient, half the graph


@dataclass(frozen=True)
class Arm:
    name: str
    temperature: float  # 0 = greedy
    cost: float = 0.0  # energy activation_cost, 0 = no energy
    real: bool = False  # train on the real continuations instead


ARMS = (
    Arm("real_control", 0.0, real=True),
    Arm("real_energy_0.1", 0.0, cost=0.1, real=True),  # does energy just block learning?
    Arm("greedy", 0.0),
    Arm("greedy_energy_0.001", 0.0, cost=0.001),
    Arm("greedy_energy_0.01", 0.0, cost=0.01),
    Arm("greedy_energy_0.1", 0.0, cost=0.1),
    Arm("t0.9", 0.9),
    Arm("t0.9_energy_0.001", 0.9, cost=0.001),
    Arm("t0.9_energy_0.01", 0.9, cost=0.01),
    Arm("t0.9_energy_0.1", 0.9, cost=0.1),
    Arm("t1.0", 1.0),
)


def check_temperature(limit: int) -> None:
    cmd = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]
    temps = [int(t) for t in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()]
    if max(temps) >= limit:
        raise SystemExit(f"GPU at {max(temps)}C, limit {limit}C: stopping")


def continuation_loss(logits: torch.Tensor, ids: torch.Tensor, first: int) -> torch.Tensor:
    return F.cross_entropy(logits[:, first - 1 : -1].float().flatten(0, 1), ids[:, first:].flatten())


class NeuronEnergy:
    """EnergyDynamics on every layer's MLP activations, stepped over tokens, one state per stream."""

    def __init__(self, model, means: list[float], cost: float, seed: int, train: bool):
        self.train = train
        self.enabled = True
        self.part = 0  # which micro-batch is running: each has its own streams
        self.aux: list[torch.Tensor] = []
        self.changed = 0.0
        self.count = 0
        self.handles = []
        parts = BATCH // MICRO if train else 1
        device = next(model.parameters()).device
        for i, (layer, mean) in enumerate(zip(model.model.layers, means, strict=True)):
            drive = cost * mean  # neutral on average
            eds = [
                EnergyDynamics(
                    drive,
                    cost,
                    precision=0.04,
                    density=0.5,
                    p=1.0,  # pretrained and dense: energy only fires and shuts off
                    exploration=min(0.001, drive / 4),
                    decay=drive / 4,
                    fire_reset_to_zero=True,
                    rng=torch.Generator(device).manual_seed(seed + 1000 * part + i),
                )
                for part in range(parts)
            ]
            self.handles.append(layer.mlp.act_fn.register_forward_hook(self._hook(eds)))

    def _hook(self, eds: list[EnergyDynamics]):
        def hook(_module, _args, output):
            if not self.enabled:
                return output
            ed = eds[self.part]
            if ed.energy is None:  # start at rest: sili's near-threshold start fires a whole layer at once
                shape = output[:, 0].shape
                ed.energy = torch.zeros(shape, device=output.device)
                ed.steps_since_fired = torch.zeros(shape, dtype=torch.long, device=output.device)
            outs = []
            for t in range(output.shape[1]):
                h = output[:, t]
                out, aux, _ = ed(h)
                if self.train:
                    self.aux.append(aux)
                self.changed = self.changed + ((out - h).detach().abs() > 1e-3).sum()
                self.count += h.numel()
                outs.append(out)
            return torch.stack(outs, 1)

        return hook

    def changed_fraction(self) -> float:
        frac = float(self.changed / max(self.count, 1))
        self.changed, self.count = 0.0, 0
        return frac

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def activation_means(model, ids: torch.Tensor) -> list[float]:
    means: list[float] = []
    record = lambda _m, _a, out: means.append(out.float().abs().mean().item())  # noqa: E731
    handles = [layer.mlp.act_fn.register_forward_hook(record) for layer in model.model.layers]
    model(input_ids=ids, use_cache=False)
    for handle in handles:
        handle.remove()
    return means


@torch.no_grad()
def generate(model, prompts: torch.Tensor, temperature: float, rng, max_temp: int) -> torch.Tensor:
    device = next(model.parameters()).device
    rows = []
    for batch in prompts.split(BATCH):
        check_temperature(max_temp)
        out = model(input_ids=batch.to(device), use_cache=True)
        made = []
        for i in range(BLOCK):
            logits = out.logits[:, -1].float()
            if temperature:
                token = torch.multinomial((logits / temperature).softmax(-1), 1, generator=rng)
            else:
                token = logits.argmax(-1, keepdim=True)
            made.append(token)
            if i + 1 < BLOCK:
                out = model(input_ids=token, past_key_values=out.past_key_values, use_cache=True)
        rows.append(torch.cat([batch, torch.cat(made, 1).cpu()], 1))
    return torch.cat(rows)


def train_epoch(model, opt, data: torch.Tensor, energy, aux_weight: float, order, max_temp: int) -> float:
    device = next(model.parameters()).device
    total, batches = 0.0, 0
    for idx in torch.randperm(len(data), generator=order).split(BATCH):
        check_temperature(max_temp)
        opt.zero_grad(set_to_none=True)
        for part, micro in enumerate(idx.split(MICRO)):
            ids = data[micro].to(device)
            if energy:
                energy.part = part
            lm = continuation_loss(model(input_ids=ids, use_cache=False).logits, ids, BLOCK)
            loss = lm
            if energy:
                loss = lm + aux_weight * torch.stack(energy.aux).sum() / ids.numel()
                energy.aux.clear()
            share = len(micro) / len(idx)
            (share * loss).backward()
            total += share * lm.item()
        opt.step()
        batches += 1
    opt.zero_grad(set_to_none=True)
    return total / batches


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, first: int) -> float:
    device = next(model.parameters()).device
    losses = []
    for batch in ids.split(BATCH):
        batch = batch.to(device)
        losses.append(continuation_loss(model(input_ids=batch, use_cache=False).logits, batch, first).item())
    return math.exp(sum(losses) / len(losses))


def sample_stats(made: torch.Tensor, rare: torch.Tensor) -> dict:
    distinct = []
    for row in made.tolist():
        grams = [tuple(row[i : i + 3]) for i in range(len(row) - 2)]
        distinct.append(len(set(grams)) / len(grams))
    return {
        "rare_mass": rare[made].float().mean().item(),
        "vocab_used": made.unique().numel(),
        "distinct3": sum(distinct) / len(distinct),
        "looping": sum(d < 0.5 for d in distinct) / len(distinct),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model directory; never written to")
    ap.add_argument("--data", required=True, help="directory with wikitext2_{train,test}.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--test-samples", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--aux-weight", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default=",".join(a.name for a in ARMS))
    ap.add_argument("--max-temp", type=int, default=80)
    args = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForCausalLM  # optional dependency

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    tok = AutoTokenizer.from_pretrained(args.model)
    config = Qwen3_5Config.from_pretrained(args.model).text_config

    def blocks(split: str, count: int) -> tuple[torch.Tensor, list[int]]:
        text = (Path(args.data) / f"wikitext2_{split}.txt").read_text()
        ids = tok(text[:2_000_000], add_special_tokens=False).input_ids
        return torch.tensor(ids[: count * 2 * BLOCK]).view(count, 2 * BLOCK), ids

    real_blocks, train_tokens = blocks("train", args.samples * (args.generations + 1))
    real = real_blocks.split(args.samples)  # one fresh set per generation
    test, _ = blocks("test", args.test_samples)
    counts = torch.bincount(torch.tensor(train_tokens), minlength=config.vocab_size).float()
    by_count = counts.argsort(descending=True)
    head = int((counts[by_count].cumsum(0) < 0.9 * counts.sum()).sum())
    rare = torch.ones(config.vocab_size, dtype=torch.bool)  # rare = outside the 90% head of real text
    rare[by_count[:head]] = False

    def load(device: str, dtype: torch.dtype):
        return Qwen3_5ForCausalLM.from_pretrained(args.model, config=config, dtype=dtype).to(device).eval()

    student, teacher = load("cuda:0", torch.float32), load("cuda:1", torch.float16)
    original = {k: v.detach().cpu().clone() for k, v in student.named_parameters()}
    means = activation_means(teacher, test[:BATCH].to("cuda:1"))

    def set_weights(model, weights) -> None:
        with torch.no_grad():
            for key, param in model.named_parameters():
                param.copy_(weights[key].to(param.device, param.dtype))

    rows, kept = [], {}
    wanted = set(args.arms.split(","))
    for arm in [a for a in ARMS if a.name in wanted]:
        torch.manual_seed(args.seed)
        order = torch.Generator().manual_seed(args.seed)
        rng = torch.Generator("cuda:1").manual_seed(args.seed)
        set_weights(student, original)
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
        train_epoch(student, opt, real[0], None, 0.0, order, args.max_temp)  # generation 0: real text
        base = perplexity(student, test, BLOCK)
        rows.append({"arm": arm.name, "generation": 0, "test_perplexity": base})
        print(json.dumps(rows[-1]), flush=True)

        for gen in range(1, args.generations + 1):
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats("cuda:0")
            check_temperature(args.max_temp)
            set_weights(teacher, dict(student.named_parameters()))
            if gen == 1:
                teach_energy, learn_energy = None, None
                if arm.cost:
                    teach_energy = NeuronEnergy(teacher, means, arm.cost, args.seed, train=False)
                    learn_energy = NeuronEnergy(student, means, arm.cost, args.seed + 100, train=True)
            data = (
                real[gen]
                if arm.real
                else generate(teacher, real[gen][:, :BLOCK], arm.temperature, rng, args.max_temp)
            )
            lm = train_epoch(student, opt, data, learn_energy, args.aux_weight, order, args.max_temp)
            row = {"arm": arm.name, "generation": gen, "train_lm": lm}
            if arm.cost:
                row["teacher_changed"] = teach_energy.changed_fraction()
                row["student_changed"] = learn_energy.changed_fraction()
                learn_energy.enabled = False  # evaluation is always energy off
            row["test_perplexity"] = perplexity(student, test, BLOCK)
            if arm.cost:
                learn_energy.enabled = True
            row |= sample_stats(data[:, BLOCK:], rare)
            row["seconds"] = time.time() - t0
            row["p40_peak_gib"] = torch.cuda.max_memory_allocated("cuda:0") / 2**30
            kept[arm.name, gen] = data
            rows.append(row)
            print(json.dumps(row), flush=True)
            with (out / "samples.jsonl").open("a") as f:
                texts = [tok.decode(r) for r in data[:3, BLOCK:]]
                f.write(json.dumps({"arm": arm.name, "generation": gen, "texts": texts}) + "\n")
            write_csv(out / "metrics.csv", rows)
        for energy in (teach_energy, learn_energy):
            if energy:
                energy.remove()
        del opt
        torch.cuda.empty_cache()

    # structure check: the original model's perplexity on every generation's samples
    set_weights(teacher, original)
    for row in rows:
        key = (row["arm"], row["generation"])
        if key in kept:
            row["original_on_samples"] = perplexity(teacher, kept[key], BLOCK)
    write_csv(out / "metrics.csv", rows)
    print("DONE", flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
