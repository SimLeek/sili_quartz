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
models run it on every layer (which outputs is the arm's `sites`: MLP neurons,
attention, residual), and their energy accumulates, never reset within an arm.
`free_*` is the student running free, greedy, energy off: perplexity is
teacher-forced and cannot see repetition.

    python examples/model_collapse.py --model DIR --data DIR --out DIR [--suite sites]
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

from sili_quartz.energy_hooks import EnergySite, NeuronEnergy, mean_abs_outputs

BLOCK = 64  # prompt tokens, and continuation tokens
BATCH = 8
MICRO = 4  # the P40 backprops half a batch at a time: same gradient, half the graph


@dataclass(frozen=True)
class Arm:
    name: str
    temperature: float  # 0 = greedy
    cost: float = 0.0  # energy activation_cost, 0 = no energy
    real: bool = False  # train on the real continuations instead
    sites: tuple[str, ...] = ("mlp",)
    decay_div: float = 4  # decay = drive / decay_div
    noise: float = 0.0  # instead of energy: white noise of this x site mean (damage-matched control)
    site_costs: tuple[float, ...] = ()  # one cost per site, else `cost` everywhere
    penalty: float = 1.0  # repetition penalty on the teacher's sampling (a token-level method)


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

ALL_SITES = ("mlp", "attn", "resid")
# Second sweep: where energy sits, with a decay that lets it build up. Doses from
# docs/results/2026-11-22-energy-vs-loops.md: each breaks short loops at ~1-1.4 nats
# of damage to real text; noise .35 costs about as much and does not.
SITE_ARMS = (
    Arm("real_control", 0.0, real=True),
    Arm("real_mlp_0.05", 0.0, 0.05, True, ("mlp",), 40),
    Arm("real_attn_0.02", 0.0, 0.02, True, ("attn",), 40),
    Arm("real_noise_0.35", 0.0, real=True, sites=ALL_SITES, noise=0.35),
    Arm("greedy", 0.0),
    Arm("greedy_mlp_0.05", 0.0, 0.05, sites=("mlp",), decay_div=40),
    Arm("greedy_attn_0.02", 0.0, 0.02, sites=("attn",), decay_div=40),
    Arm("greedy_resid_0.01", 0.0, 0.01, sites=("resid",), decay_div=40),
    Arm("greedy_all_0.003", 0.0, 0.003, sites=ALL_SITES, decay_div=40),
    Arm("greedy_mlp0.03_attn0.01", 0.0, 0.03, sites=("mlp", "attn"), decay_div=40, site_costs=(0.03, 0.01)),
    Arm("greedy_noise_0.35", 0.0, sites=ALL_SITES, noise=0.35),
    Arm("greedy_penalty_1.5", 0.0, penalty=1.5),
    Arm("real_mlp0.03_attn0.01", 0.0, 0.03, True, ("mlp", "attn"), 40, site_costs=(0.03, 0.01)),
    Arm("t0.9", 0.9),
    Arm("t1.0", 1.0),
    Arm("t0.9_mlp_0.05", 0.9, 0.05, sites=("mlp",), decay_div=40),
    Arm("t0.9_attn_0.02", 0.9, 0.02, sites=("attn",), decay_div=40),
    Arm("t0.9_mlp0.03_attn0.01", 0.9, 0.03, sites=("mlp", "attn"), decay_div=40, site_costs=(0.03, 0.01)),
)
SUITES = {"pilot": ARMS, "sites": SITE_ARMS}


def check_temperature(limit: int) -> None:
    cmd = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]
    temps = [int(t) for t in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.split()]
    if max(temps) >= limit:
        raise SystemExit(f"GPU at {max(temps)}C, limit {limit}C: stopping")


def continuation_loss(logits: torch.Tensor, ids: torch.Tensor, first: int) -> torch.Tensor:
    return F.cross_entropy(logits[:, first - 1 : -1].float().flatten(0, 1), ids[:, first:].flatten())


def site_modules(model, kind: str) -> list:
    layers = model.model.layers
    if kind == "mlp":
        return [layer.mlp.act_fn for layer in layers]
    if kind == "attn":  # what attention adds to the residual (a linear or a full layer)
        return [getattr(layer, "linear_attn", None) or layer.self_attn for layer in layers]
    if kind == "resid":  # what the next layer receives
        return list(layers)
    raise ValueError(f"unknown energy site {kind!r}")


def site_means(model, ids: torch.Tensor) -> dict[str, list[float]]:
    modules = [m for kind in ALL_SITES for m in site_modules(model, kind)]
    means = mean_abs_outputs(modules, lambda: model(input_ids=ids, use_cache=False))
    n = len(model.model.layers)
    return {kind: means[i * n : (i + 1) * n] for i, kind in enumerate(ALL_SITES)}


def energy_sites(model, means: dict[str, list[float]], kinds: tuple[str, ...], costs=()) -> list[EnergySite]:
    """Every site is scaled to its layer's MLP activations, so a cost is the same dose everywhere."""
    sites = []
    for i, kind in enumerate(kinds):
        cost = costs[i] if costs else None
        for module, mean, ref in zip(site_modules(model, kind), means[kind], means["mlp"], strict=True):
            sites.append(EnergySite(module, ref, scale=1.0 if kind == "mlp" else mean / ref, cost=cost))
    return sites


class SiteNoise:
    """White noise of sigma x (site mean) at the same sites: the damage-matched control for energy."""

    def __init__(self, sites: list[EnergySite], sigma: float, seed: int):
        self.enabled, self.part, self.aux = True, 0, []
        self.handles = [
            s.module.register_forward_hook(self._hook(s, sigma, seed + i)) for i, s in enumerate(sites)
        ]
        self.gens: dict = {}

    def _hook(self, site: EnergySite, sigma: float, seed: int):
        def hook(_module, _args, output):
            if not self.enabled:
                return None
            h = output[0] if isinstance(output, tuple) else output
            gen = self.gens.setdefault((seed, h.device), torch.Generator(h.device).manual_seed(seed))
            noise = torch.randn(h.shape, generator=gen, device=h.device) * (sigma * site.mean * site.scale)
            noisy = h + noise.to(h.dtype)
            return (noisy, *output[1:]) if isinstance(output, tuple) else noisy

        return hook

    def changed_fraction(self) -> float:
        return 1.0

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def generate(model, prompts: torch.Tensor, temperature: float, rng, max_temp: int, penalty: float = 1.0):
    device = next(model.parameters()).device
    rows = []
    for batch in prompts.split(BATCH):
        check_temperature(max_temp)
        out = model(input_ids=batch.to(device), use_cache=True)
        made = []
        seen = torch.zeros(len(batch), out.logits.shape[-1], dtype=torch.bool, device=device)
        seen[torch.arange(len(batch), device=device)[:, None], batch.to(device)] = True
        for i in range(BLOCK):
            logits = out.logits[:, -1].float()
            if penalty != 1.0:  # CTRL / Hugging Face repetition penalty on every token already used
                logits = torch.where(
                    seen, torch.where(logits > 0, logits / penalty, logits * penalty), logits
                )
            if temperature:
                token = torch.multinomial((logits / temperature).softmax(-1), 1, generator=rng)
            else:
                token = logits.argmax(-1, keepdim=True)
            made.append(token)
            seen[torch.arange(len(batch), device=device), token[:, 0]] = True
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
            if energy and energy.aux:
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
    ap.add_argument("--suite", choices=SUITES, default="pilot")
    ap.add_argument("--arms", help="comma-separated subset of the suite")
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
    means = site_means(teacher, test[:BATCH].to("cuda:1"))

    def set_weights(model, weights) -> None:
        with torch.no_grad():
            for key, param in model.named_parameters():
                param.copy_(weights[key].to(param.device, param.dtype))

    rows, kept = [], {}
    arms = SUITES[args.suite]
    if args.arms:
        wanted = set(args.arms.split(","))
        assert wanted <= {a.name for a in arms}, (
            f"not in suite {args.suite}: {wanted - {a.name for a in arms}}"
        )
        arms = [a for a in arms if a.name in wanted]
    for arm in arms:
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
                if arm.noise:
                    teach_energy = SiteNoise(energy_sites(teacher, means, arm.sites), arm.noise, args.seed)
                    learn_energy = SiteNoise(
                        energy_sites(student, means, arm.sites), arm.noise, args.seed + 100
                    )
                elif arm.cost:
                    teach_energy = NeuronEnergy(
                        energy_sites(teacher, means, arm.sites, arm.site_costs),
                        arm.cost,
                        args.seed,
                        decay_div=arm.decay_div,
                    )
                    learn_energy = NeuronEnergy(
                        energy_sites(student, means, arm.sites, arm.site_costs),
                        arm.cost,
                        args.seed + 100,
                        train=True,
                        parts=BATCH // MICRO,
                        decay_div=arm.decay_div,
                    )
            data = (
                real[gen]
                if arm.real
                else generate(teacher, real[gen][:, :BLOCK], arm.temperature, rng, args.max_temp, arm.penalty)
            )
            lm = train_epoch(student, opt, data, learn_energy, args.aux_weight, order, args.max_temp)
            row = {"arm": arm.name, "generation": gen, "train_lm": lm}
            if arm.cost or arm.noise:
                row["teacher_changed"] = teach_energy.changed_fraction()
                row["student_changed"] = learn_energy.changed_fraction()
                learn_energy.enabled = False  # evaluation is always energy off
            row["test_perplexity"] = perplexity(student, test, BLOCK)
            free = generate(student, test[: 2 * BATCH, :BLOCK], 0.0, None, args.max_temp)[:, BLOCK:]
            row |= {f"free_{k}": v for k, v in sample_stats(free, rare).items()}
            if arm.cost or arm.noise:
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
