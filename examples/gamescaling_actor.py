"""GameScaling-0.8B playing Minecraft through robonet (track A, step A.7).

Needs, on the same machine: the game (tools/minecraft.py launch), the robonet
desktop endpoint and the headless brain with ai_res 16:9 (854x480, the
"480p" model input of GameScaling's own Minecraft evaluation). Then:

    python examples/gamescaling_actor.py --device cuda:0 --decisions 50 --log actor.jsonl

Each decision: the latest frame -> the policy (5 past turns, no reasoning,
temperature 1; StreamingPolicy -- new frame only, sliding KV window,
CUDA-graph decode -- unless --slow) -> 6 steps of
33 ms through robonet. Pipelined (unless --no-pipeline): the next decision is
computed while the current chunk plays, its frame captured so it lands as
the chunk ends; reaction time (frame -> first step of its chunk) and the
interval between chunks go to LOG.timing.jsonl. One JSON line per decision
in LOG: the raw completion, whether it parsed, seconds, prompt tokens.
Ctrl+C (or --decisions / --seconds) releases every key and hands control back.

Training (A.9+): --replay-dir saves each decision's frame for the trainer
(examples/gamescaling_trainer.py), --sync-dir applies the trainer's merged
weight updates in place a few tensors per decision, --energy puts energy on
the model (MLP activations and attention outputs, sili_quartz/energy_scan.py)
at the dose the trainer writes to --dose-file (0 until it does).

--random-from LOG is the A-0 random-policy baseline: no model; each decision
is an action drawn uniformly (seeded) from a previous run's LOG, after a
compute time drawn from the same LOG, so keys, mouse and cadence match that
run but ignore the screen.
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
    MINECRAFT_KEYS, MINECRAFT_RULES, GameScalingPolicy, PipelinedExecutor, RobonetExecutor, StreamingPolicy,
    build_system_prompt, parse_action,
)


class ReplayRandomPolicy:
    """Random baseline: actions and compute times drawn independently from a logged run."""

    def __init__(self, log: Path, seed: int):
        rows = [json.loads(line) for line in open(log)]
        self.texts = [r["text"] for r in rows if r.get("parsed")]
        self.seconds = [r["seconds"] for r in rows[10:]]  # skip warm-up
        self.rng = np.random.default_rng(seed)
        self.last_seconds = self.last_prefill_s = self.last_decode_s = 0.0
        self.last_prompt_tokens, self.last_reused = 0, False

    def act(self, frame):
        t0 = time.perf_counter()
        text = self.texts[self.rng.integers(len(self.texts))]
        time.sleep(max(0.0, self.seconds[self.rng.integers(len(self.seconds))] - (time.perf_counter() - t0)))
        self.last_seconds = time.perf_counter() - t0
        return parse_action(text), text


def load_policy(model_dir: str, device: str, dtype: torch.dtype, history: int, temperature: float,
                fast: bool = True, session_turns: int = 30):
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)
    if fast:
        return StreamingPolicy(model, processor, system, history_len=history, temperature=temperature)
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

    client = FrameClient()
    # The heartbeat (run thread) and the executor thread both send; a
    # multiprocessing Connection is not safe for concurrent sends.
    lock = threading.Lock()
    raw_send = client.bridge.send

    def locked_send(message):
        with lock:
            return raw_send(message)

    client.bridge.send = locked_send
    return client


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
    ap.add_argument("--slow", action="store_true",
                    help="the model card's setup: re-read the last 5 turns every call, transformers decode")
    ap.add_argument("--no-pipeline", action="store_true", help="compute, then execute, in turn")
    ap.add_argument("--random-from", type=Path, default=None, help="random baseline: replay draws from this LOG")
    ap.add_argument("--seed", type=int, default=0, help="seed for --random-from")
    ap.add_argument("--replay-dir", type=Path, default=None, help="save each decision's frame here (JPEG)")
    ap.add_argument("--sync-dir", type=Path, default=None, help="apply the trainer's weight updates from here")
    ap.add_argument("--energy", action="store_true", help="energy on the model, dose from --dose-file")
    ap.add_argument("--dose-file", type=Path, default=None)
    a = ap.parse_args()

    if a.random_from:
        policy = ReplayRandomPolicy(a.random_from, a.seed)
    else:
        policy = load_policy(a.model, a.device, getattr(torch, a.dtype), a.history, a.temperature,
                             fast=not a.slow, session_turns=0 if a.slow else 30)
    client = make_client()
    if not client.connect(timeout_s=30):
        raise SystemExit("robonet bridge not reachable")
    threading.Thread(target=client.run, daemon=True).start()
    while client.frame is None:
        time.sleep(0.05)
    energy = applier = saver = None
    if a.energy:
        from sili_quartz.actor_objective import decoder_attn_post, make_energy, measure_site_means

        frame0 = Image.fromarray(client.frame)
        text = policy._render(policy.messages(frame0, turns=[]))
        inputs = policy.processor(text=[text], images=[frame0], return_tensors="pt").to(policy.model.device)
        mlp_means, attn_means = measure_site_means(policy.model, lambda: policy.model(**inputs))
        energy = make_energy(policy.model, mlp_means, attn_means)
        energy.set_dose(cost_scale=0.0, noise_scale=0.0)
        policy.decoder.attn_post = decoder_attn_post(energy, len(mlp_means))
        if a.dose_file:
            means_path = a.dose_file.parent / "energy_means.json"
            means_path.write_text(json.dumps({"mlp": mlp_means, "attn": attn_means}))
    if a.sync_dir:
        from sili_quartz.continual import SyncApplier

        tied = {"lm_head.weight", "model.language_model.embed_tokens.weight"}
        applier = SyncApplier(policy.model, a.sync_dir, per_call=16, delete_applied=True,
                              on_applied=lambda name: policy.decoder.refresh_head() if name in tied else None)
    if a.replay_dir:
        from concurrent.futures import ThreadPoolExecutor

        a.replay_dir.mkdir(parents=True, exist_ok=True)
        saver = ThreadPoolExecutor(max_workers=1)
    dose = 0.0
    pipelined = not a.no_pipeline
    executor = PipelinedExecutor(client) if pipelined else RobonetExecutor(client)
    client.set_want_control("ai")
    t_start, n, parsed = time.time(), 0, 0
    compute_est = 0.3  # running estimate of seconds per decision
    try:
        while (not a.decisions or n < a.decisions) and (not a.seconds or time.time() - t_start < a.seconds):
            if pipelined:
                # capture the frame late enough that the decision lands when the current chunk ends
                wait = executor.chunk_end - compute_est - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            if applier is not None:
                applier.step()
            if energy is not None and a.dose_file and n % 20 == 0 and a.dose_file.exists():
                try:
                    new = float(json.loads(a.dose_file.read_text())["dose"])
                except (ValueError, KeyError, OSError):
                    new = dose
                if new != dose:
                    dose = new
                    energy.set_dose(cost_scale=dose, noise_scale=dose)
            frame_time = time.perf_counter()
            frame = Image.fromarray(client.frame)
            frame_age = time.time() - client.frame_time
            action, completion = policy.act(frame)
            compute_est = 0.8 * compute_est + 0.2 * policy.last_seconds
            if pipelined:
                executor.submit(action, frame_time)
            else:
                executor.execute(action)
            n += 1
            parsed += action.parsed
            if saver is not None:
                saver.submit(frame.save, a.replay_dir / f"{n:06d}.jpg", quality=90)
            row = {"t": round(time.time(), 3), "n": n, "parsed": action.parsed, "truncated": action.truncated,
                   "text": action.text, "raw": completion, "dropped": action.dropped,
                   "seconds": round(policy.last_seconds, 3), "prompt_tokens": policy.last_prompt_tokens,
                   "prefill_s": round(policy.last_prefill_s, 3), "decode_s": round(policy.last_decode_s, 3),
                   "reused": policy.last_reused, "frame_age": round(frame_age, 3)}
            if getattr(policy, "sampled_tokens", None):
                row["sampled"] = list(policy.sampled_tokens)
                row["sampled_logp"] = [round(v, 5) for v in policy.sampled_logp]
            if applier is not None:
                row["synced_updates"] = applier.applied_updates
            if energy is not None:
                row["dose"] = dose
            if a.log:
                with open(a.log, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            if n <= 3 or n % 10 == 0:
                print(json.dumps({k: row[k] for k in ("n", "parsed", "text", "seconds", "prompt_tokens")}), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if pipelined:
            time.sleep(0.25)
            executor.stop()
            if a.log and executor.timings:
                with open(a.log.with_suffix(".timing.jsonl"), "w") as fh:
                    for t in executor.timings:
                        fh.write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}) + "\n")
        executor.release_all()
        client.set_want_control("human")
        time.sleep(0.5)
        client.stop()
        client.close()
    print(json.dumps({"decisions": n, "parse_rate": round(parsed / max(n, 1), 3),
                      "seconds": round(time.time() - t_start, 1)}), flush=True)


if __name__ == "__main__":
    main()
