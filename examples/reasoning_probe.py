"""Read what GameScaling thinks while it plays (evaluation only).

Every --every seconds: copy the frame robonet last sent the AI (the brain's
monitor frame.png, drawn cursor included), let a separate copy of the
untouched model think before acting on it (reasoning mode), and append
{t, frame, thought, action} to --out. The playing model is not affected.

    python examples/reasoning_probe.py --monitor ~/robonet_logs/monitor/frame.png \\
        --out RUN/probes.jsonl --every 60 --seconds 7200
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from PIL import Image

from sili_quartz.gamescaling import MINECRAFT_KEYS, MINECRAFT_RULES, build_system_prompt, reasoning_probe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--monitor", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--every", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--samples", type=int, default=4, help="samples per frame (most skip thinking)")
    a = ap.parse_args()
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(a.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)
    frames = a.out.parent / "probe_frames"
    frames.mkdir(parents=True, exist_ok=True)
    t_start, i = time.time(), 0
    while not a.seconds or time.time() - t_start < a.seconds:
        t0 = time.time()
        i += 1
        dst = frames / f"{i:05d}.png"
        try:
            shutil.copy(a.monitor, dst)
            frame = Image.open(dst).convert("RGB")
        except (OSError, ValueError):
            time.sleep(5)
            continue
        for k in range(a.samples):
            res = reasoning_probe(model, proc, system, frame, a.max_new)
            row = {"t": round(t0, 1), "i": i, "sample": k, "frame": dst.name, **res,
                   "seconds": round(time.time() - t0, 1)}
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            if row["thought"]:
                print(json.dumps({"i": i, "thought": row["thought"][:160]}), flush=True)
        time.sleep(max(0.0, a.every - (time.time() - t0)))


if __name__ == "__main__":
    main()
