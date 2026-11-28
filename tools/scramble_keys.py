"""Scramble the action keys' token rows, for the random-init token-memory check.

    python tools/scramble_keys.py --out scrambled.pt [--seed 0]

The output head is tied to the input embeddings, so one row per key token
holds both how the key is read and how it is chosen. Each single-token key
(" LB", " W", ...) gets a random direction with its own row norm kept, so
nothing about which key is which survives, but the scale does. The result is
a trainer-style checkpoint (just the embedding tensor), loadable with
`--init` or by any probe's checkpoint argument. The grammar mask still keeps
every action well-formed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sili_quartz.gamescaling import KEY_VOCAB, action_vocab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--init", type=Path, default=None, help="scramble this checkpoint's rows instead of the base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(a.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32)
    if a.init is not None:
        from sili_quartz.continual import load_checkpoint

        load_checkpoint(model, a.init)
    vocab = action_vocab(tok)
    strings = [tok.decode([t]) for t in vocab]
    ids = []
    for k in sorted(KEY_VOCAB):
        for cand in (" " + k, k):
            if cand in strings:
                ids.append(vocab[strings.index(cand)])
                break
    name = next(n for n, p in model.named_parameters() if p is model.get_input_embeddings().weight)
    w = model.get_input_embeddings().weight.detach().clone().float()
    g = torch.Generator().manual_seed(a.seed)
    rows = torch.randn(len(ids), w.shape[1], generator=g)
    rows = rows / rows.norm(dim=1, keepdim=True) * w[ids].norm(dim=1, keepdim=True)
    w[ids] = rows
    torch.save({name: w.half()}, a.out)
    print(json.dumps({"scrambled_tokens": len(ids), "tensor": name, "out": str(a.out)}))


if __name__ == "__main__":
    main()
