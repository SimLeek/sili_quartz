"""Re-initialize some language-model layers, for random-init checks of the training methods.

    python tools/reinit_layers.py --layers 20-23 --out reinit_last_chunk.pt [--seed 0]

The base model was trained by behaviour cloning and is near optimal already, so
a method can look good just by not breaking it. Re-initializing the chunk
nearest the output (layers 21-24, indices 20-23) keeps the pretrained
perception and removes the cloned policy: the training method must build a
policy on top of working vision.

Each re-initialized layer gets fresh weights from the model's own
initializer (`_init_weights`: normal(0, initializer_range) for Linear weights,
the norms' defaults). The result is a trainer-style checkpoint holding just
those layers, loadable with `--init` (actor and trainer) or by the probes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--layers", default="20-23", help="inclusive index range of language-model layers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the re-initialized Linear weights by this (e.g. 1e-6: a near-zero start, connections "
                         "not grown yet; the blocks then pass the residual stream through and their gradients are "
                         "second-order small, the deadlock energy's forced firing was built to break)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import Qwen3_5ForConditionalGeneration

    lo, hi = (int(x) for x in a.layers.split("-"))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32)
    torch.manual_seed(a.seed)
    layers = model.model.language_model.layers
    names = {}
    for i in range(lo, hi + 1):
        for m in layers[i].modules():
            model._init_weights(m)
            if a.scale != 1.0 and isinstance(m, torch.nn.Linear):
                with torch.no_grad():
                    m.weight.mul_(a.scale)
        prefix = f"model.language_model.layers.{i}."
        for n, p in layers[i].named_parameters():
            names[prefix + n] = p
    params = dict(model.named_parameters())
    missing = [n for n in names if n not in params]
    assert not missing, missing[:3]
    torch.save({n: p.detach().float() for n, p in names.items()}, a.out)  # float32: a 1e-6 scale underflows fp16
    print(json.dumps({"layers": [lo, hi], "scale": a.scale, "tensors": len(names), "out": str(a.out)}))


if __name__ == "__main__":
    main()
