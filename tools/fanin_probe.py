"""Direct fan-in of the action tokens, per model (evaluation only, weights only, CPU).

    python tools/fanin_probe.py --out f.jsonl base CK1 CK2 ...

The output head is tied to the input embeddings, and every layer's output
neurons (MLP down-projection columns, attention output-projection columns)
write into the residual stream the head reads through the final norm. Token
t's direct fan-in from output neuron j of layer l is

    F[l, t, j] = ((1 + g) * W_out_l[:, j]) . E_t

with g the final norm's weight and E_t the token's embedding row, centered
over the action vocabulary (only relative logits matter for the choice).

Per model and per action-key token:
- strength: norm of its fan-in over all layers, and per chunk of 4 layers;
- overlap: mean |cosine| between its fan-in and each other key token's
  (high = its drivers also drive other keys: not exclusive), and the most
  overlapping partner;
- vs the first model given (normally the base): strength ratio, and cosine
  between its fan-in now and its fan-in in the first model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sili_quartz.continual import load_checkpoint  # noqa: E402
from sili_quartz.gamescaling import KEY_VOCAB, action_vocab  # noqa: E402


def write_matrices(model) -> list[torch.Tensor]:
    """Per layer, the (d_model, n) matrices whose columns write into the residual stream."""
    out = []
    for layer in model.model.language_model.layers:
        attn = layer.self_attn.o_proj if hasattr(layer, "self_attn") else layer.linear_attn.out_proj
        out.append(torch.cat([attn.weight.detach().float(), layer.mlp.down_proj.weight.detach().float()], 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="'base' or trainer checkpoint paths; the first is the reference")
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(a.model)
    vocab = action_vocab(tok)
    strings = [tok.decode([t]) for t in vocab]
    keys = sorted(KEY_VOCAB)
    key_rows = {}  # key -> row index in vocab (the " key" token if it is a single token)
    for k in keys:
        for cand in (" " + k, k):
            if cand in strings:
                key_rows[k] = strings.index(cand)
                break
    ref = None
    for name in a.models:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32)
        if name != "base":
            load_checkpoint(model, name)
        with torch.no_grad():
            emb = model.get_input_embeddings().weight[vocab].float()
            emb = emb - emb.mean(0)
            gain = 1.0 + model.model.language_model.norm.weight.float()
            mats = write_matrices(model)
            rows = torch.tensor([key_rows[k] for k in key_rows])
            e = emb[rows] * gain  # (K, d)
            per_layer = [e @ m for m in mats]  # (K, n_l) per layer
        full = torch.cat(per_layer, 1)  # (K, sum n_l)
        strength = full.norm(dim=1)
        chunk_strength = torch.stack([torch.cat(per_layer[c:c + a.chunk], 1).norm(dim=1)
                                      for c in range(0, len(per_layer), a.chunk)], 1)
        unit = full / full.norm(dim=1, keepdim=True)
        cos = (unit @ unit.T).abs()
        cos.fill_diagonal_(0)
        overlap = cos.sum(1) / (len(key_rows) - 1)
        partner = cos.argmax(1)
        names = list(key_rows)
        tokens = {}
        for i, k in enumerate(names):
            tokens[k] = {"strength": round(float(strength[i]), 3),
                         "chunks": [round(float(x), 3) for x in chunk_strength[i]],
                         "overlap": round(float(overlap[i]), 4),
                         "partner": names[int(partner[i])], "partner_cos": round(float(cos[i, partner[i]]), 4)}
            if ref is not None:
                tokens[k]["strength_vs_ref"] = round(float(strength[i] / ref["strength"][i]), 4)
                tokens[k]["cos_vs_ref"] = round(float(unit[i] @ ref["unit"][i]), 4)
        if ref is None:
            ref = {"strength": strength, "unit": unit}
        row = {"model": name, "keys": len(names), "mean_overlap": round(float(overlap.mean()), 4), "tokens": tokens}
        print(json.dumps({"model": name, "mean_overlap": row["mean_overlap"],
                          "LB": tokens.get("LB"), "W": tokens.get("W")}), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        del model


if __name__ == "__main__":
    main()
