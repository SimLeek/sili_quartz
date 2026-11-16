"""Train Titans-MAC on key-value recall and report accuracy against the honest baseline.

python examples/train_recall.py              # long-gap: only memory can solve it
python examples/train_recall.py --gap 0      # in-context control
python examples/train_recall.py --no-memory  # attention-only control (must stay near baseline)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from sili_quartz.models import FAMILIES
from sili_quartz.models.common import recall_loss
from sili_quartz.tasks import (
    RecallSpec,
    accuracy,
    assert_gap_hides_pairs,
    constant_predictor_baseline,
    distinct_predictions,
    make_batch,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gap", type=int, default=16)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--pairs", type=int, default=4)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--gates", default="const", choices=["const", "kernel_matched"])
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--family", default="titans", choices=list(FAMILIES))
    p.add_argument("--mem-tokens", type=int, default=4, help="RMT/combined state tokens")
    p.add_argument(
        "--gap-curriculum", default="", help="comma list of gaps to train through in order, e.g. 0,8,16"
    )
    p.add_argument("--no-key-from-prev", action="store_true", help="key writes on the same token (weaker)")
    p.add_argument("--tbptt", type=int, default=64, help="chunks between memory-state detaches")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    spec = RecallSpec(n_pairs=a.pairs, n_queries=a.pairs, gap=a.gap)
    if a.gap > 0:
        assert_gap_hides_pairs(spec, a.chunk)
    kw = dict(vocab=spec.vocab, d=a.d, n_heads=4, chunk=a.chunk)
    if a.family in ("titans", "combined"):
        kw.update(
            gates=a.gates,
            eta=a.eta,
            alpha=a.alpha,
            write_key_from_prev=not a.no_key_from_prev,
            tbptt_chunks=a.tbptt,
        )
    elif a.family == "rmt":
        kw.update(tbptt_chunks=a.tbptt, mem_tokens=a.mem_tokens)
    model = FAMILIES[a.family](**kw).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ev_toks, ev_tgts = make_batch(spec, 512, np.random.default_rng(12345))
    ev_toks, ev_tgts = ev_toks.to(a.device), ev_tgts.to(a.device)
    base = constant_predictor_baseline(ev_tgts)
    print(
        f"device={a.device} gap={a.gap} chunk={a.chunk} pairs={a.pairs} family={a.family}",
        f"key_from_prev={not a.no_key_from_prev} tbptt={a.tbptt}",
        f"params={sum(q.numel() for q in model.parameters())} baseline={base:.4f}",
    )

    t0 = time.time()
    stages = [int(g) for g in a.gap_curriculum.split(",") if g != ""] or [a.gap]
    per_stage = max(1, a.steps // len(stages))
    for step in range(1, a.steps + 1):
        stage_gap = stages[min((step - 1) // per_stage, len(stages) - 1)]
        toks, tgts = make_batch(RecallSpec(n_pairs=a.pairs, n_queries=a.pairs, gap=stage_gap), a.batch, rng)
        logits, _ = model(toks.to(a.device))
        loss = recall_loss(logits, tgts.to(a.device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, a.steps // 10) == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                ev_logits, _ = model(ev_toks)
            model.train()
            acc = accuracy(ev_logits, ev_tgts)
            k = distinct_predictions(ev_logits, ev_tgts)
            flag = "  <-- COLLAPSED" if k <= 2 else ""
            print(
                f"step {step:5d}  loss {loss.item():.4f}  acc {acc:.4f}  (+{acc - base:+.4f} vs base)  "
                f"distinct {k}  {time.time() - t0:6.1f}s{flag}",
                flush=True,
            )


if __name__ == "__main__":
    main()
