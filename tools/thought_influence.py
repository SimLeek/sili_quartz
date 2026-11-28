"""Do the model's thoughts change its actions? (evaluation only)

    python tools/thought_influence.py --set world:RUN:0:10 --set pause:RUN:40:100 --states 16 --out t.jsonl

Per state (one frame from a run's replay, no history: with past turns the
model copies their empty thought blocks):
1. the model thinks (prefill "<think>", sampled until "</think>");
2. K actions sampled after its own thought (T), and K after an empty thought
   block (D, the direct mode the actor plays in);
3. every sampled action scored under both prefixes (teacher-forced sequence
   log-probability), giving KL(D || T) ~ mean over a ~ D of log pD(a) - log pT(a)
   and KL(T || D) the other way, in nats per action;
4. the keys pressed: Jaccard overlap of the key sets used by the T and D samples.
Thoughts that change nothing give KL ~ 0 and overlap ~ the within-D overlap.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from gamescaling_trainer import read_rows  # noqa: E402

from sili_quartz.gamescaling import (  # noqa: E402
    ACTION_END, MINECRAFT_KEYS, MINECRAFT_RULES, PREFILL_DIRECT, PREFILL_REASONING, USER_TEXT, build_system_prompt,
)


def keys_of(action_text: str) -> set[str]:
    body = action_text.split(ACTION_END)[0]
    return {k for part in body.split(";")[1:] for k in part.split()}


def jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, help="label:RUN:from_min:to_min")
    ap.add_argument("--states", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)

    def render(prefill: str) -> str:
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [{"type": "text", "text": USER_TEXT}, {"type": "image", "image": None}]},
                {"role": "assistant", "content": [{"type": "text", "text": prefill}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, continue_final_message=True)

    @torch.no_grad()
    def generate(prefix: str, frame, max_new: int) -> list[int]:
        inp = proc(text=[prefix], images=[frame], return_tensors="pt").to(a.device)
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=True, temperature=1.0, use_cache=True)
        return out[0, inp["input_ids"].shape[1]:].tolist()

    @torch.no_grad()
    def seq_logp(prefix: str, frame, action_ids: list[int]) -> float:
        inp = proc(text=[prefix], images=[frame], return_tensors="pt").to(a.device)
        n0 = inp["input_ids"].shape[1]
        add = torch.tensor([action_ids], device=a.device)
        inp["input_ids"] = torch.cat([inp["input_ids"], add], 1)
        inp["attention_mask"] = torch.ones_like(inp["input_ids"])
        inp["mm_token_type_ids"] = torch.cat([inp["mm_token_type_ids"], torch.zeros_like(add)], 1)
        logits = model(**inp).logits[0, n0 - 1:-1].float().log_softmax(-1)
        return float(logits.gather(1, add[0][:, None]).sum())

    def action_ids(completion_ids: list[int]) -> list[int]:
        """Cut a sampled completion after the action's end tag."""
        for j in range(1, len(completion_ids) + 1):
            if ACTION_END in tok.decode(completion_ids[:j], skip_special_tokens=False):
                return completion_ids[:j]
        return completion_ids

    for spec in a.set:
        label, run, lo, hi = spec.split(":")
        run = Path(run)
        rows = read_rows(run / "actor.jsonl", run / "replay")
        t0 = rows[0]["t"]
        idx = [i for i, r in enumerate(rows) if float(lo) * 60 <= r["t"] - t0 < float(hi) * 60]
        for i in sorted(rng.sample(idx, min(a.states, len(idx)))):
            frame = Image.open(run / "replay" / f"{rows[i]['n']:06d}.jpg").convert("RGB")
            comp = tok.decode(generate(render(PREFILL_REASONING), frame, 384), skip_special_tokens=False)
            thought, sep, _ = comp.partition("</think>")
            row = {"set": label, "decision": rows[i]["n"], "thought": thought.strip()[:400],
                   "thought_tokens": len(tok(thought, add_special_tokens=False).input_ids), "finished": bool(sep)}
            if not sep or not thought.strip():
                row["skipped"] = "no thought" if sep else "thought not finished"
            else:
                pre_t = render(PREFILL_REASONING + thought + "</think>\n\n<|action_start|")
                pre_d = render(PREFILL_DIRECT)
                acts = {c: [action_ids(generate(p, frame, 64)) for _ in range(a.k)] for c, p in (("T", pre_t), ("D", pre_d))}
                lp = {(c, j, q): seq_logp(p, frame, ids) for c in "TD" for j, ids in enumerate(acts[c])
                      for q, p in (("T", pre_t), ("D", pre_d))}
                kl_dt = sum(lp[("D", j, "D")] - lp[("D", j, "T")] for j in range(a.k)) / a.k
                kl_td = sum(lp[("T", j, "T")] - lp[("T", j, "D")] for j in range(a.k)) / a.k
                txt = {c: [tok.decode(ids, skip_special_tokens=False) for ids in acts[c]] for c in "TD"}
                kt = set().union(*[keys_of(x) for x in txt["T"]])
                kd = set().union(*[keys_of(x) for x in txt["D"]])
                within = [jaccard(keys_of(x), keys_of(y)) for j, x in enumerate(txt["D"]) for y in txt["D"][j + 1:]]
                across = [jaccard(keys_of(x), keys_of(y)) for x in txt["T"] for y in txt["D"]]
                row.update({"kl_d_t": round(kl_dt, 3), "kl_t_d": round(kl_td, 3),
                            "keys_T": sorted(kt), "keys_D": sorted(kd),
                            "jaccard_across": round(sum(across) / len(across), 3),
                            "jaccard_within_D": round(sum(within) / max(1, len(within)), 3),
                            "actions_T": [re.sub(r"\s+", " ", x)[:80] for x in txt["T"]],
                            "actions_D": [re.sub(r"\s+", " ", x)[:80] for x in txt["D"]]})
            print(json.dumps(row), flush=True)
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
