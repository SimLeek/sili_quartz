"""Does LoRA training concentrate the weight change in a few directions? (evaluation, weights only, CPU)

    python tools/spectrum_probe.py --out s.jsonl CK1 CK2 ...

For every Linear in the language model's layers, against the base weights W0:
- rel_change = ||W - W0|| / ||W0||;
- eff_rank of dW = exp(entropy of its normalized singular values), and the
  share of ||dW||^2 in its top 8 singular values;
- intruders: of W's top-10 left singular vectors, how many have best |cosine|
  < --eps with every left singular vector of W0 (Shuttleworth et al. 2024,
  "LoRA vs Full Fine-tuning: An Illusion of Equivalence": new high-ranking
  directions unlike the pretrained ones, linked to forgetting).
Prints per checkpoint the means over matrices, and the matrices with the most
intruders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sili_quartz.continual import load_checkpoint  # noqa: E402

PAT = re.compile(r"language_model\.layers\.\d+\..*\.weight$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--model", default=os.path.expanduser("~/models/GameScaling-0.8B-work"))
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from transformers import Qwen3_5ForConditionalGeneration

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    base = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.float32)
    w0 = {n: p.detach() for n, p in base.named_parameters() if PAT.search(n) and p.dim() == 2}
    u0 = {}
    for name in a.checkpoints:
        sd = torch.load(name, map_location="cpu", weights_only=True)
        rows = []
        for n, W0 in w0.items():
            if n not in sd:
                continue
            W = sd[n].float()
            dW = W - W0
            rel = float(dW.norm() / W0.norm())
            s = torch.linalg.svdvals(dW)
            p = s / s.sum() if float(s.sum()) > 0 else s
            eff = math.exp(float(-(p[p > 0] * p[p > 0].log()).sum())) if float(s.sum()) > 0 else 0.0
            top8 = float((s[:8] ** 2).sum() / (s ** 2).sum()) if float(s.sum()) > 0 else 0.0
            if n not in u0:
                u0[n] = torch.linalg.svd(W0, full_matrices=False).U
            U = torch.linalg.svd(W, full_matrices=False).U[:, :a.top]
            best = (U.T @ u0[n]).abs().max(1).values
            rows.append({"name": n, "rel_change": rel, "eff_rank": eff, "top8_share": top8,
                         "intruders": int((best < a.eps).sum()), "min_best_cos": float(best.min())})
        if not rows:
            continue
        out = {"checkpoint": name, "matrices": len(rows),
               "mean_rel_change": round(sum(r["rel_change"] for r in rows) / len(rows), 6),
               "mean_eff_rank": round(sum(r["eff_rank"] for r in rows) / len(rows), 1),
               "mean_top8_share": round(sum(r["top8_share"] for r in rows) / len(rows), 3),
               "total_intruders": sum(r["intruders"] for r in rows),
               "matrices_with_intruders": sum(r["intruders"] > 0 for r in rows),
               "worst": sorted(rows, key=lambda r: (-r["intruders"], r["min_best_cos"]))[:5]}
        print(json.dumps({k: v for k, v in out.items() if k != "worst"}), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
