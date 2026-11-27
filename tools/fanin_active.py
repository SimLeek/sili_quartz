"""Active fan-in of the action-key tokens, per model, on the same replay windows
(evaluation only). The weights-only version (fanin_probe.py) showed RL barely
moves the static paths; this weights each path by the neuron's activation.

    python tools/fanin_active.py --ref RUN --minutes 0:10 --windows 24 --out f.jsonl base CK1 CK2 ...

For every position p in the windows whose next token is an action key, and
every output neuron j of every layer l (inputs of the attention output
projection and of the MLP down-projection):

    c[l, j, t](p) = a[l, j](p) * F[l, t, j] / rms(h_final(p)),
    F[l, t, j] = ((1 + g) * W_out_l[:, j]) . E_t   (E centered over the action vocab)

is neuron j's direct contribution to key t's logit at p. Per model and key t,
over the positions where t is the emitted token:
- drive: mean total contribution to t (sum over neurons and layers);
- spread: participation ratio of |c| over neurons (how many neurons carry t);
- overlap: mean |cosine| between t's contribution vector and the other keys'
  at the same positions (high = the neurons driving t also drive other keys).
Same windows and positions for every model, so the numbers compare directly.
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

from sili_quartz.actor_objective import turn_positions  # noqa: E402
from sili_quartz.continual import load_checkpoint  # noqa: E402
from sili_quartz.gamescaling import (  # noqa: E402
    KEY_VOCAB, MINECRAFT_KEYS, MINECRAFT_RULES, USER_TEXT, action_vocab, build_system_prompt,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--minutes", default="0:10")
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)
    vocab = action_vocab(tok)
    strings = [tok.decode([t]) for t in vocab]
    key_tok = {}  # token id -> key
    for k in sorted(KEY_VOCAB):
        for cand in (" " + k, k):
            if cand in strings:
                key_tok[vocab[strings.index(cand)]] = k
                break
    keys = sorted(set(key_tok.values()))
    kidx = {k: i for i, k in enumerate(keys)}
    key_vocab_rows = [strings.index(" " + k) if " " + k in strings else strings.index(k) for k in keys]

    rows = read_rows(a.ref / "actor.jsonl", a.ref / "replay")
    t0 = rows[0]["t"]
    lo, hi = (float(x) * 60 for x in a.minutes.split(":"))
    starts = [i for i in range(len(rows) - a.window) if lo <= rows[i]["t"] - t0 < hi]
    starts = sorted(random.Random(a.seed).sample(starts, min(a.windows, len(starts))))

    def inputs(i):
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}]
        for r in rows[i:i + a.window]:
            img = Image.open(a.ref / "replay" / f"{r['n']:06d}.jpg").convert("RGB")
            msgs.append({"role": "user", "content": [{"type": "text", "text": USER_TEXT}, {"type": "image", "image": img}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": assistant_content(r)}]})
        return proc.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt").to(a.device)

    for name in a.models:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
        if name != "base":
            load_checkpoint(model, name)
        lm = model.model.language_model
        with torch.no_grad():
            emb = model.get_input_embeddings().weight[vocab].float()
            emb = emb - emb.mean(0)
            e = emb[key_vocab_rows] * (1.0 + lm.norm.weight.float())  # (K, d)
        mods = []
        for layer in lm.layers:
            mods.append(layer.self_attn.o_proj if hasattr(layer, "self_attn") else layer.linear_attn.out_proj)
            mods.append(layer.mlp.down_proj)
        with torch.no_grad():
            F = [e @ m.weight.float() for m in mods]  # (K, n) per module
        want: list[int] = []
        acts: list[torch.Tensor] = []
        handles = [m.register_forward_pre_hook(lambda mod, args: acts.append(args[0][0, want].float()))
                   for m in mods]
        rms_box: list[torch.Tensor] = []
        handles.append(lm.norm.register_forward_pre_hook(
            lambda mod, args: rms_box.append(args[0][0, want].float().pow(2).mean(-1).add(1e-6).sqrt())))
        drive = torch.zeros(len(keys)); spread = torch.zeros(len(keys)); over = torch.zeros(len(keys))
        count = torch.zeros(len(keys))
        with torch.no_grad():
            for i in starts:
                inp = inputs(i)
                ids = inp["input_ids"][0].tolist()
                spans = {q for t in turn_positions(ids, tok, prefill="<|action_start|") for q in t["action"]}
                pos = [q - 1 for q in range(1, len(ids)) if q in spans and ids[q] in key_tok]
                if not pos:
                    continue
                want[:] = pos
                acts.clear(); rms_box.clear()
                model(**inp, logits_to_keep=1)
                rms = rms_box[0]  # (P,)
                for pi, q in enumerate(pos):
                    # contributions of every output neuron to every key at this position: (K, sum n)
                    c = torch.cat([acts[m][pi][None] * F[m] for m in range(len(mods))], 1) / rms[pi]
                    t = kidx[key_tok[ids[q + 1]]]
                    v = c[t]
                    drive[t] += float(v.sum())
                    av = v.abs()
                    spread[t] += float(av.sum() ** 2 / (av.pow(2).sum() + 1e-12))
                    others = torch.cat([c[:t], c[t + 1:]])
                    cosv = torch.nn.functional.cosine_similarity(others, v[None], dim=1).abs()
                    over[t] += float(cosv.mean())
                    count[t] += 1
        for h in handles:
            h.remove()
        seen = count > 0
        per = {keys[t]: {"n": int(count[t]), "drive": round(float(drive[t] / count[t]), 3),
                         "spread": round(float(spread[t] / count[t]), 1), "overlap": round(float(over[t] / count[t]), 4)}
               for t in range(len(keys)) if seen[t]}
        row = {"model": name, "positions": int(count.sum()), "keys_seen": int(seen.sum()),
               "mean_drive": round(float((drive[seen] / count[seen]).mean()), 3),
               "mean_spread": round(float((spread[seen] / count[seen]).mean()), 1),
               "mean_overlap": round(float((over[seen] / count[seen]).mean()), 4), "tokens": per}
        print(json.dumps({k: row[k] for k in ("model", "positions", "keys_seen", "mean_drive", "mean_spread", "mean_overlap")}), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
