"""Offline check of the state-empowerment estimate (evaluation only; run before
using it as a reward). Static screens should score near 0, the open world
higher, and only once the tether has trained: on an untrained tether the
probe predictions also carry the probe's own action text, so every screen
reads as spurious control.

    python tools/empowerment_check.py --calib RUN --first 600 \
        --set world:RUN_A:0:10 --set pause:RUN_B:40:100 [--checkpoint CK] --out e.jsonl

1. Calibration: the chunk tether's banks are filled by running the model over
   the first --first decisions of --calib (forward only), giving chunk 1's
   standardization and its per-dimension error sigma2.
2. Each --set label:RUN:from_min:to_min: --states states sampled from that
   stretch of that run, each with its 5 preceding decisions as history.
3. The same --probes probe actions (drawn once from --calib) for every state.
Prints per set: states, mean, median, min and max empowerment (nats).
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

from sili_quartz.actor_objective import ChunkTether, image_runs, turn_positions  # noqa: E402
from sili_quartz.continual import load_checkpoint  # noqa: E402
from sili_quartz.empowerment import state_empowerment  # noqa: E402
from sili_quartz.gamescaling import (  # noqa: E402
    ACTION_END, MINECRAFT_KEYS, MINECRAFT_RULES, USER_TEXT, build_system_prompt,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--first", type=int, default=600)
    ap.add_argument("--set", action="append", required=True, help="label:RUN:from_min:to_min")
    ap.add_argument("--states", type=int, default=24)
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--checkpoint", type=Path, default=None)
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
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    if a.checkpoint is not None:
        load_checkpoint(model, a.checkpoint)
    n_layers = len(model.model.language_model.layers)
    tether = ChunkTether(n_layers, chunk=a.chunk)
    image_token = model.config.image_token_id

    def frame(run: Path, r: dict):
        return Image.open(run / "replay" / f"{r['n']:06d}.jpg").convert("RGB")

    calib = read_rows(a.calib / "actor.jsonl", a.calib / "replay")[:a.first]
    with torch.no_grad():
        for i in range(0, len(calib) - a.window + 1, a.window):
            win = calib[i:i + a.window]
            msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
            for r in win:
                msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT},
                                                         {"type": "image", "image": frame(a.calib, r)}]})
                msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_content(r)}]})
            inp = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt").to(a.device)
            ids = inp["input_ids"][0].tolist()
            turns = turn_positions(ids, tok, prefill="<|action_start|")
            if len(turns) != len(win):
                continue
            hs = model(**inp, output_hidden_states=True, logits_to_keep=1).hidden_states
            tether(hs, [t["action"][-1] for t in turns], image_runs(ids, image_token, [t["state"] for t in turns]))
    st = tether.chunk0_stats()
    pool = list(dict.fromkeys(r["raw"] for r in calib if r.get("parsed") and (r.get("raw") or "").rstrip().endswith(ACTION_END)))
    probes = rng.sample(pool, a.probes)
    head = {"checkpoint": str(a.checkpoint) if a.checkpoint else "base", "calib_pairs": int(tether.bx[0].shape[0]),
            "chunk0_mse": round(st["mse"], 4), "chunk0_copy_mse": round(st["copy_mse"], 4), "probes": probes}
    print(json.dumps(head), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(head) + "\n")
    for spec in a.set:
        label, run, lo, hi = spec.split(":")
        run = Path(run)
        rows = read_rows(run / "actor.jsonl", run / "replay")
        t0 = rows[0]["t"]
        idx = [i for i, r in enumerate(rows) if float(lo) * 60 <= r["t"] - t0 < float(hi) * 60 and i >= a.window - 1]
        es = []
        for i in sorted(rng.sample(idx, min(a.states, len(idx)))):
            hist = [(frame(run, r), assistant_content(r)) for r in rows[i - a.window + 1:i]]
            e, _ = state_empowerment(model, proc, system, hist, frame(run, rows[i]), probes, a.chunk, st)
            es.append(e)
        es.sort()
        row = {"set": label, "run": run.name, "minutes": [float(lo), float(hi)], "states": len(es),
               "mean": round(sum(es) / len(es), 3), "median": round(es[len(es) // 2], 3),
               "min": round(es[0], 3), "max": round(es[-1], 3)}
        print(json.dumps(row), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
