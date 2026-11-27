"""How far a checkpoint's policy and per-layer states have moved from the base
model on the same frames (evaluation only).

    python tools/policy_drift.py --run RUN --minutes 32:37 --checkpoint CK --states 12 --out d.jsonl

For --states decisions from that stretch of the run, each with its 5
preceding decisions as history (as the actor plays, direct mode):
- K actions sampled by the base model and by the checkpoint: share with a left
  click (LB), median mouse move, the keys used;
- per layer, the cosine between the base and checkpoint hidden states at the
  last prefix token (where the first action token is predicted), mean over
  states. Low cosine in the last layers means the action head reads different
  features than it was trained on.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from gamescaling_trainer import assistant_content, read_rows  # noqa: E402

from sili_quartz.continual import load_checkpoint  # noqa: E402
from sili_quartz.gamescaling import (  # noqa: E402
    ACTION_END, MINECRAFT_KEYS, MINECRAFT_RULES, PREFILL_DIRECT, USER_TEXT, build_system_prompt,
)


def parse(text: str):
    body = text.split(ACTION_END)[0]
    parts = body.lstrip(">").split(";")
    try:
        n = parts[0].split()
        mouse = abs(float(n[0])) + abs(float(n[1]))
    except (ValueError, IndexError):
        mouse = None
    keys = [k for p in parts[1:] for k in p.split()]
    return mouse, keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--minutes", required=True, help="from:to")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    rng = random.Random(a.seed)
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)
    rows = read_rows(a.run / "actor.jsonl", a.run / "replay")
    t0 = rows[0]["t"]
    lo, hi = (float(x) * 60 for x in a.minutes.split(":"))
    idx = [i for i, r in enumerate(rows) if lo <= r["t"] - t0 < hi and i >= 5]
    picks = sorted(rng.sample(idx, min(a.states, len(idx))))

    def frame(r):
        return Image.open(a.run / "replay" / f"{r['n']:06d}.jpg").convert("RGB")

    def inputs(i):
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        hist = rows[i - 5:i]
        for r in hist:
            msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT}, {"type": "image", "image": None}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_content(r)}]})
        msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT}, {"type": "image", "image": None}]})
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": PREFILL_DIRECT}]})
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True)
        return proc(text=[text], images=[frame(r) for r in hist] + [frame(rows[i])], return_tensors="pt").to(a.device)

    results = {}
    for name in ("base", "checkpoint"):
        torch.manual_seed(a.seed)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
        if name == "checkpoint":
            load_checkpoint(model, a.checkpoint)
        states, acts = [], []
        with torch.no_grad():
            for i in picks:
                inp = inputs(i)
                hs = model(**inp, output_hidden_states=True, logits_to_keep=1).hidden_states
                states.append(torch.stack([h[0, -1].float().cpu() for h in hs]))
                for _ in range(a.k):
                    out = model.generate(**inp, max_new_tokens=64, do_sample=True, temperature=1.0, use_cache=True)
                    acts.append(tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=False))
        results[name] = (torch.stack(states), acts)
        del model
        torch.cuda.empty_cache()

    cos = torch.nn.functional.cosine_similarity(results["base"][0], results["checkpoint"][0], dim=-1).mean(0)
    summary = {"run": a.run.name, "minutes": a.minutes, "checkpoint": str(a.checkpoint), "states": len(picks),
               "layer_cosine": [round(float(c), 3) for c in cos]}
    for name in ("base", "checkpoint"):
        parsed = [parse(x) for x in results[name][1]]
        mice = sorted(m for m, _ in parsed if m is not None)
        keys = {}
        for _, ks in parsed:
            for k in ks:
                keys[k] = keys.get(k, 0) + 1
        summary[name] = {"actions": len(parsed), "lb_share": round(sum("LB" in ks for _, ks in parsed) / len(parsed), 3),
                         "mouse_median": mice[len(mice) // 2] if mice else None,
                         "top_keys": sorted(keys.items(), key=lambda kv: -kv[1])[:6],
                         "examples": [x.split(ACTION_END)[0][:70] for x in results[name][1][:6]]}
    print(json.dumps(summary), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
