"""GameScaling-0.8B playing Minecraft through robonet (track A, step A.7).

Needs, on the same machine: the game (tools/minecraft.py launch), the robonet
desktop endpoint and the headless brain with ai_res 16:9 (854x480, the
"480p" model input of GameScaling's own Minecraft evaluation). Then:

    python examples/gamescaling_actor.py --device cuda:0 --decisions 50 --log actor.jsonl

Each decision: the latest frame -> the policy (5 past turns, no reasoning,
temperature 1) -> 6 steps of 33 ms through robonet. One JSON line per
decision: the raw completion, whether it parsed, seconds, prompt tokens.
Ctrl+C (or --decisions / --seconds) releases every key and hands control back.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sili_quartz.gamescaling import (
    MINECRAFT_KEYS, MINECRAFT_RULES, GameScalingPolicy, RobonetExecutor, build_system_prompt,
)


def load_policy(model_dir: str, device: str, dtype: torch.dtype, history: int, temperature: float):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)
    return GameScalingPolicy(model, processor, system, history_len=history, temperature=temperature)


def make_client():
    from robonet.bridge.ai_client_base import RobonetAIClient

    class FrameClient(RobonetAIClient):
        """Keeps the latest video frame (BGR, from the bridge) as RGB."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.frame = None
            self.frame_time = 0.0

        def on_robonet_start(self): pass
        def on_robonet_shutdown(self): self.stop()
        def on_robonet_connect(self, name): pass
        def on_robonet_disconnect(self): pass

        def on_channel_data(self, label, data):
            info = self.channel_info.get(label)
            if label != "video" or not info:
                return
            a = np.frombuffer(data, dtype=info["dtype"])
            if a.size == np.prod(info["shape"]):
                self.frame = a.reshape(info["shape"])[:, :, ::-1].copy()
                self.frame_time = time.time()

    return FrameClient()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--history", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--decisions", type=int, default=0, help="stop after this many (0 = no limit)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = no limit)")
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args()

    policy = load_policy(a.model, a.device, getattr(torch, a.dtype), a.history, a.temperature)
    client = make_client()
    if not client.connect(timeout_s=30):
        raise SystemExit("robonet bridge not reachable")
    threading.Thread(target=client.run, daemon=True).start()
    while client.frame is None:
        time.sleep(0.05)
    executor = RobonetExecutor(client)
    client.set_want_control("ai")
    t_start, n, parsed = time.time(), 0, 0
    try:
        while (not a.decisions or n < a.decisions) and (not a.seconds or time.time() - t_start < a.seconds):
            frame = Image.fromarray(client.frame)
            frame_age = time.time() - client.frame_time
            action, completion = policy.act(frame)
            executor.execute(action)
            n += 1
            parsed += action.parsed
            row = {"t": round(time.time(), 3), "n": n, "parsed": action.parsed, "truncated": action.truncated,
                   "text": action.text, "raw": completion, "dropped": action.dropped,
                   "seconds": round(policy.last_seconds, 3), "prompt_tokens": policy.last_prompt_tokens,
                   "frame_age": round(frame_age, 3)}
            if a.log:
                with open(a.log, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            if n <= 3 or n % 10 == 0:
                print(json.dumps({k: row[k] for k in ("n", "parsed", "text", "seconds", "prompt_tokens")}), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        executor.release_all()
        client.set_want_control("human")
        time.sleep(0.5)
        client.stop()
        client.close()
    print(json.dumps({"decisions": n, "parse_rate": round(parsed / max(n, 1), 3),
                      "seconds": round(time.time() - t_start, 1)}), flush=True)


if __name__ == "__main__":
    main()
