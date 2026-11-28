"""Exclusive fan-in consolidation, offline (track A token memory, stage A).

    python tools/fanin_consolidate.py --ref RUN --minutes 0:10 --windows 16 \
        [--init CK] --rounds 4 --out-ckpt consolidated.pt --out log.jsonl

Keeps every action key *reachable* without setting how often it is used, and
without using which keys were pressed (no behaviour cloning):

1. Exclusive sets: the last chunk's output neurons (inputs of the attention
   output projection and the MLP down-projection, last --chunk layers) are
   partitioned among the keys, each neuron to the key it pushes hardest
   relative to every other key (exclusive drive F[t, j] - max_{u != t} F[u, j],
   F = direct path through the final norm and the tied head; with
   --competitors vocab the rivals are every action-vocabulary token that can
   win the slot, separators and spaces included). Each key keeps at
   most n_neurons / n_keys of them (a partition).
2. Capacity of key t: the 95th percentile, over all action-token positions of
   the replay windows, of the push its exclusive neurons give t's logit
   (activation * F / rms of the final residual).
3. Floor: --floor x the median capacity. The default 0.71 is the untouched
   model's weakest/median ratio of active drive (17 keys seen).
4. Update (one round): for each key below the floor, every exclusive neuron's
   write column moves a step --eta toward the key's direction (embedding row
   times the final-norm gain, centered over the action vocabulary), with the
   rivals' directions projected out (no pathway shared between tokens), then
   is renormalized to its old norm.

Writes the changed weights as a trainer-style checkpoint (--out-ckpt).
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
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--minutes", default="0:10")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--init", type=Path, default=None)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--floor", type=float, default=0.71)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--competitors", choices=("keys", "vocab"), default="vocab",
                    help="exclusivity against the other keys only (v1) or against every action-vocabulary token "
                         "that competes at the slot, separators and spaces included (v2, default)")
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    system = build_system_prompt("Minecraft", MINECRAFT_RULES, MINECRAFT_KEYS)
    vocab = action_vocab(tok)
    strings = [tok.decode([t]) for t in vocab]
    key_rows = []
    for k in sorted(KEY_VOCAB):
        for cand in (" " + k, k):
            if cand in strings:
                key_rows.append(strings.index(cand))
                break
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32).to(a.device).eval()
    if a.init is not None:
        load_checkpoint(model, a.init)
    lm = model.model.language_model
    layers = lm.layers[-a.chunk:]
    mods = []
    for layer in layers:
        mods.append(layer.self_attn.o_proj if hasattr(layer, "self_attn") else layer.linear_attn.out_proj)
        mods.append(layer.mlp.down_proj)

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

    batches = [inputs(i) for i in starts]
    action_pos = []
    for inp in batches:
        ids = inp["input_ids"][0].tolist()
        action_pos.append(sorted({q - 1 for t in turn_positions(ids, tok, prefill="<|action_start|") for q in t["action"]}))

    non_key = [i for i in range(len(vocab)) if i not in set(key_rows)]

    def vocab_dirs():
        """Directions of every action-vocab token (centered, through the final-norm gain): keys first."""
        with torch.no_grad():
            emb = model.get_input_embeddings().weight[vocab].float()
            emb = emb - emb.mean(0)
            emb = emb * (1.0 + lm.norm.weight.float())
            return emb[key_rows], emb[non_key]

    def key_dirs():
        return vocab_dirs()[0]

    def measure():
        """Activations at action positions, exclusive sets, capacities."""
        e, e_other = vocab_dirs()
        with torch.no_grad():
            F = torch.cat([e @ m.weight.float() for m in mods], 1)  # (K, N)
            G = torch.cat([e_other @ m.weight.float() for m in mods], 1) if a.competitors == "vocab" else None
        others = F.clone()
        excl = torch.empty_like(F)
        g_max = G.max(0).values if G is not None else None
        for t in range(F.shape[0]):
            others[t] = float("-inf")
            rival = others.max(0).values if g_max is None else torch.maximum(others.max(0).values, g_max)
            excl[t] = F[t] - rival
            others[t] = F[t]
        owner = excl.argmax(0)  # each neuron to the key it pushes hardest, relatively
        quota = F.shape[1] // F.shape[0]
        member = torch.zeros_like(F, dtype=torch.bool)
        for t in range(F.shape[0]):
            cand = (owner == t).nonzero().view(-1)
            if cand.numel():
                top = cand[excl[t, cand].argsort(descending=True)[:quota]]
                member[t, top] = True
        acts, rmss = [], []
        want = []
        cap_acts: list[torch.Tensor] = []
        hs = [m.register_forward_pre_hook(lambda mod, args: cap_acts.append(args[0][0, want].float())) for m in mods]
        h2 = lm.norm.register_forward_pre_hook(lambda mod, args: rmss.append(args[0][0, want].float().pow(2).mean(-1).add(1e-6).sqrt()))
        with torch.no_grad():
            for inp, pos in zip(batches, action_pos):
                want[:] = pos
                cap_acts.clear()
                model(**inp, logits_to_keep=1)
                acts.append(torch.cat(cap_acts, 1) / rmss[-1][:, None])
        for h in hs + [h2]:
            h.remove()
        A = torch.cat(acts)  # (P, N)
        push = A @ (F * member).T  # (P, K): push of each key's exclusive neurons
        cap = torch.quantile(push, 0.95, dim=0)
        return F, member, cap

    log = []
    for rnd in range(a.rounds + 1):
        F, member, cap = measure()
        med = float(cap.median())
        floor = a.floor * med
        below = (cap < floor).nonzero().view(-1).tolist()
        row = {"round": rnd, "median_capacity": round(med, 3), "floor": round(floor, 3), "below": len(below),
               "capacity_min": round(float(cap.min()), 3), "capacity_max": round(float(cap.max()), 3),
               "members_per_key": int(member.sum(1).float().mean())}
        print(json.dumps(row), flush=True)
        log.append(row)
        if rnd == a.rounds or not below:
            break
        e, e_other = vocab_dirs()
        eu = e / e.norm(dim=1, keepdim=True)
        ou = e_other / e_other.norm(dim=1, keepdim=True)
        with torch.no_grad():
            offs = 0
            for m in mods:
                n = m.weight.shape[1]
                for t in below:
                    cols = member[t, offs:offs + n].nonzero().view(-1)
                    if not cols.numel():
                        continue
                    others = torch.cat([eu[:t], eu[t + 1:]] + ([ou] if a.competitors == "vocab" else []))
                    q, _ = torch.linalg.qr(others.T)  # orthonormal basis of the other keys' directions
                    d = eu[t] - q @ (q.T @ eu[t])
                    d = d / d.norm()
                    W = m.weight[:, cols].float()
                    norms = W.norm(dim=0, keepdim=True)
                    W = W + a.eta * norms * d[:, None]
                    W = W / W.norm(dim=0, keepdim=True) * norms
                    m.weight[:, cols] = W.to(m.weight.dtype)
                offs += n
    emb = model.get_input_embeddings().weight
    sd = {n: p.detach().half().cpu() for n, p in model.named_parameters()
          if any(p is m.weight for m in mods) or p is emb}  # the (possibly scrambled) key rows travel with it
    torch.save(sd, a.out_ckpt)
    with open(a.out, "a") as fh:
        for row in log:
            fh.write(json.dumps({"init": str(a.init), **row}) + "\n")


if __name__ == "__main__":
    main()
