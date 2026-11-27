"""How collapsed is a checkpoint's policy? (evaluation only, energy off)

    python tools/repertoire_probe.py --set world:RUN:0:10 --states 30 --k 4 --out r.jsonl base CK1 CK2 ...

For each model (the base, or a trainer checkpoint loaded into it), on the same
states (each with its 5 preceding decisions as history, as the actor plays),
K actions are sampled exactly as the actor samples them (grammar mask,
temperature 1, the CUDA-graph decoder). Energy is off, so this measures the
weights' own repertoire, not energy's noise. Per model:
- distinct keys used (of the 74), key entropy (bits), LB share;
- distinct actions / actions, median mouse move;
- mean log-prob per sampled token (higher = more certain);
- input coherence: mean key-set Jaccard similarity between the K actions of
  the *same* state minus that between actions of *different* states. High =
  the keys depend consistently on the screen; ~0 = noise, or one action
  everywhere.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from gamescaling_trainer import assistant_content, read_rows  # noqa: E402

from sili_quartz.continual import load_checkpoint  # noqa: E402
from sili_quartz.gamescaling import (  # noqa: E402
    ActionGrammar, GameScalingPolicy, MINECRAFT_KEYS, MINECRAFT_RULES, build_system_prompt,
)


def parse(text: str):
    body = re.sub(r"<\|action_(start|end)\|>", "", text)
    parts = body.split(";")
    try:
        n = parts[0].split()
        mouse = abs(float(n[0])) + abs(float(n[1]))
    except (ValueError, IndexError):
        mouse = None
    return mouse, [k for p in parts[1:] for k in p.split()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="'base' or trainer checkpoint paths")
    ap.add_argument("--set", action="append", required=True, help="label:RUN:from_min:to_min")
    ap.add_argument("--states", type=int, default=30, help="states per set")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0, help="which states are drawn")
    ap.add_argument("--sample-seed", type=int, default=None,
                    help="seed of the action sampling (default: --seed); vary it on fixed states to measure the "
                         "probe's own sampling noise")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    rng = random.Random(a.seed)
    proc = AutoProcessor.from_pretrained(a.model)
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)
    states = []  # (run, rows, index)
    for spec in a.set:
        label, run, lo, hi = spec.split(":")
        run = Path(run)
        rows = read_rows(run / "actor.jsonl", run / "replay")
        t0 = rows[0]["t"]
        idx = [i for i, r in enumerate(rows) if float(lo) * 60 <= r["t"] - t0 < float(hi) * 60 and i >= 5]
        states += [(run, rows, i) for i in sorted(rng.sample(idx, min(a.states, len(idx))))]

    def frame(run, r):
        return Image.open(run / "replay" / f"{r['n']:06d}.jpg").convert("RGB")

    for name in a.models:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
        if name != "base":
            load_checkpoint(model, name)
        pol = GameScalingPolicy(model, proc, system, history_len=5, temperature=1.0, fast_decode=True)
        pol.grammar = ActionGrammar(pol.tok, pol.vocab, pol.max_new_tokens)
        pol.generator = torch.Generator(device=a.device).manual_seed(a.seed if a.sample_seed is None else a.sample_seed)
        keys, actions, mice, lps = collections.Counter(), [], [], []
        per_state: list[list[frozenset]] = []
        for run, rows, i in states:
            hist = [(frame(run, r), assistant_content(r)) for r in rows[i - 5:i]]
            cur = frame(run, rows[i])
            per_state.append([])
            for _ in range(a.k):
                pol.reset()
                pol.turns = list(hist)
                action, completion = pol.act(cur)
                m, ks = parse(action.text)
                keys.update(ks)
                per_state[-1].append(frozenset(ks))
                actions.append(action.text)
                if m is not None:
                    mice.append(m)
                if pol.sampled_logp:
                    lps.append(sum(pol.sampled_logp) / len(pol.sampled_logp))
        def jac(x, y):
            return 1.0 if not x and not y else len(x & y) / len(x | y)
        within = [jac(x, y) for st in per_state for i, x in enumerate(st) for y in st[i + 1:]]
        across = [jac(x, y) for si, st in enumerate(per_state) for st2 in per_state[si + 1:] for x in st for y in st2]
        coherence = (sum(within) / len(within) - sum(across) / len(across)) if within and across else 0.0
        n = sum(keys.values())
        ent = -sum(v / n * math.log2(v / n) for v in keys.values()) if n else 0.0
        mice.sort()
        row = {"model": name, "sample_seed": a.seed if a.sample_seed is None else a.sample_seed, "actions": len(actions), "distinct_actions": len(set(actions)),
               "distinct_keys": len(keys), "key_entropy_bits": round(ent, 3),
               "lb_share": round(sum("LB" in parse(t)[1] for t in actions) / len(actions), 3),
               "mouse_median": mice[len(mice) // 2] if mice else None,
               "token_logp": round(sum(lps) / len(lps), 3) if lps else None,
               "coherence": round(coherence, 4),
               "top_keys": keys.most_common(6),
               "top_actions": collections.Counter(actions).most_common(2)}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        del pol, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
