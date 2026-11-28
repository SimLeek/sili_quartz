"""P-1, the loop test (docs/energy-controller/2-methodology.md, section 2.3).

GameScaling-0.8B's text head writes FREE tokens greedily (Mirostat samples)
after each prompt. Every arm is scored the same way, on its own output:
- loops (3 prompts that end in a phrase repeated many times): distinct-3,
  under 0.5 = still repeating;
- real text (3 WikiText-2 test segments): distinct-3, and the unmodified
  model's perplexity of what was written (low = repetitive, high =
  incoherent; the real continuation is the reference);
- the effect KL per token (energy arms), measured every EFFECT_EVERY steps
  with a clean pass on a copy of the cache; and seconds per token.

Arms: P0a no energy, P0b Mirostat, P0c fixed dose (MLP 0.03 + attention
0.01, the 2c dose), P1, P1a, P1b, P1 x R (noise), P1 x D (drive). Every
controller starts at P0c's dose. Starting values are measured, not tuned:
delta = P0c's mean effect KL on the real prompts; tau = the model's mean
surprise on WikiText-2 *train* text (Mirostat gets the same tau in bits).

    python examples/loop_test.py --device cuda:0 --out loop_test.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5Config, Qwen3_5ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_collapse as mc  # noqa: E402

from sili_quartz.controllers import EffectMeter, Mirostat, SurpriseMeter, make_controller  # noqa: E402
from sili_quartz.energy_hooks import NeuronEnergy  # noqa: E402

WARM, TAIL = 128, 256
P0C_SITES, P0C_COSTS, DECAY_DIV = ("mlp", "attn"), (0.03, 0.01), 40
PHRASES = [
    " He was badly injured in the war .",
    " The team lost again",
    "\n 2 . The album reached number one on the chart in France and Germany , selling many copies .",
]
REAL_STARTS = {"realA": 0, "realB": 2000, "realC": 8000}


def distinct3(row):
    grams = [tuple(row[i : i + 3]) for i in range(len(row) - 2)]
    return len(set(grams)) / max(len(grams), 1)


class Harness:
    def __init__(self, args):
        self.args = args
        self.dev = args.device
        torch.set_grad_enabled(False)
        self.tok = AutoTokenizer.from_pretrained(args.model)
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            args.model, config=Qwen3_5Config.from_pretrained(args.model).text_config, dtype=torch.float32
        ).to(self.dev).eval()
        data = Path(args.data)
        real = self.tok((data / "wikitext2_test.txt").read_text()[:40000], add_special_tokens=False).input_ids
        train = self.tok((data / "wikitext2_train.txt").read_text()[:200000], add_special_tokens=False).input_ids
        self.prompts, self.reference = {}, {}
        for i, (start, p) in enumerate(zip((0, 6000, 7000), PHRASES, strict=True)):
            ids = self.tok(p, add_special_tokens=False).input_ids
            self.prompts[f"loop{i + 1}"] = (real[start : start + WARM] + ids * TAIL)[: WARM + TAIL]
        for name, start in REAL_STARTS.items():
            self.prompts[name] = real[start : start + WARM + TAIL]
            self.reference[name] = real[start + WARM + TAIL : start + WARM + TAIL + args.free]
        self.means = mc.site_means(self.model, torch.tensor(train[4000:5024], device=self.dev).view(8, 128))
        self.tau = self.text_surprise(train[10000 : 10000 + 8 * 256])

    def text_surprise(self, ids) -> float:
        """Mean -log p (nats) of real text under the model: the tau for P1/P1b."""
        x = torch.tensor(ids, device=self.dev).view(8, -1)
        logits = self.model(input_ids=x, use_cache=False).logits
        return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), x[:, 1:].flatten()).item()

    def original_ppl(self, prompt, made) -> float:
        ids = torch.tensor([prompt + made], device=self.dev)
        logits = self.model(input_ids=ids, use_cache=False).logits
        return math.exp(F.cross_entropy(logits[0, len(prompt) - 1 : -1].float(), ids[0, len(prompt) :]).item())

    def energy(self, seed):
        sites = mc.energy_sites(self.model, self.means, P0C_SITES, P0C_COSTS)
        return NeuronEnergy(sites, P0C_COSTS[0], seed, decay_div=DECAY_DIV)

    def free_run(self, prompt, energy=None, controller=None, mirostat=None, seed=0):
        """Returns (tokens, per-step log)."""
        gen = torch.Generator(self.dev).manual_seed(seed)
        out = self.model(input_ids=torch.tensor([prompt], device=self.dev), use_cache=True)
        effect = EffectMeter(every=self.args.effect_every) if energy is not None else None
        surprise = SurpriseMeter(window=self.args.surprise_window)
        made, log = [], []
        for i in range(self.args.free):
            logits = out.logits[0, -1].float()
            token = mirostat.sample(logits, gen) if mirostat else int(logits.argmax())
            made.append(token)
            s = surprise.update(logits[None], torch.tensor([token], device=self.dev))
            if i + 1 == self.args.free:
                break
            ids = torch.tensor([[token]], device=self.dev)
            cache = out.past_key_values
            snapshot = copy.deepcopy(cache) if effect is not None and effect.due() else None
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
            d = None
            if effect is not None:
                def clean(snapshot=snapshot):
                    with energy.disabled():
                        return self.model(input_ids=ids, past_key_values=snapshot, use_cache=True).logits[0, -1]
                d = effect.measure(out.logits[0, -1], clean) if snapshot is not None else effect.measure(None, None)
            if controller is not None:
                controller.update(effect=d, surprise=s)
                controller.apply(energy)
            log.append({"surprise": s, "effect": d, "level": controller.level if controller else None,
                        "delta": getattr(controller, "delta", None)})
        return made, log, (effect.seconds if effect else [])

    def run_arm(self, name, make):
        rows, t0, tokens = {}, time.time(), 0
        effects, clean_s = [], []
        for pname, prompt in self.prompts.items():
            energy, controller, mirostat = make()
            made, log, secs = self.free_run(prompt, energy, controller, mirostat, seed=self.args.seed)
            if energy is not None:
                energy.remove()
            tokens += len(made)
            clean_s += secs
            effects += [r["effect"] for r in log if r["effect"] is not None]
            row = {"distinct3": round(distinct3(made), 3), "text": self.tok.decode(made[:28])}
            if pname.startswith("real"):
                row["orig_ppl"] = round(self.original_ppl(prompt, made), 1)
            if controller is not None:
                row["level_end"] = round(controller.level, 3)
                row["delta_end"] = round(getattr(controller, "delta", float("nan")), 4)
            if log:
                row["surprise_mean"] = round(sum(r["surprise"] for r in log) / len(log), 3)
            rows[pname] = row
        seconds = time.time() - t0
        return {
            "arm": name, **rows,
            "effect_kl_mean": round(sum(effects) / len(effects), 5) if effects else None,
            "seconds_per_token": round(seconds / tokens, 4),
            "clean_pass_seconds": round(sum(clean_s) / len(clean_s), 4) if clean_s else None,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--data", default=os.path.expanduser("~/collapse/data"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--free", type=int, default=128)
    ap.add_argument("--effect-every", type=int, default=4)
    ap.add_argument("--surprise-window", type=int, default=16)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    h = Harness(args)

    def emit(row):
        line = json.dumps(row)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a") as fh:
                fh.write(line + "\n")

    ref = {n: round(h.original_ppl(h.prompts[n], h.reference[n]), 1) for n in h.reference}
    emit({"reference_orig_ppl": ref, "reference_distinct3": {n: round(distinct3(r), 3) for n, r in h.reference.items()},
          "tau_nats": round(h.tau, 3), "free": args.free, "effect_every": args.effect_every})

    p0c = h.run_arm("P0c fixed dose", lambda: (h.energy(args.seed), None, None))
    delta0 = p0c["effect_kl_mean"]
    emit({**p0c, "delta0": delta0})
    ctrl = lambda name, knob="cost", **kw: (  # noqa: E731
        lambda: (h.energy(args.seed), make_controller(name, knob, delta=delta0, eta=args.eta, **kw), None))
    arms = {
        "P0a no energy": lambda: (None, None, None),
        "P0b Mirostat": lambda: (None, None, Mirostat(tau=h.tau / math.log(2))),
        "P1": ctrl("P1", tau=h.tau),
        "P1a": ctrl("P1a"),
        "P1b": lambda: (h.energy(args.seed), make_controller("P1b", tau=h.tau), None),
        "P1 x R": ctrl("P1", "noise", tau=h.tau),
        "P1 x D": ctrl("P1", "drive", tau=h.tau),
    }
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
    for name, make in arms.items():
        if wanted and name not in wanted:
            continue
        emit(h.run_arm(name, make))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
