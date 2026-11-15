"""Train the memory model once, then evaluate on longer gaps and more live
bindings WITHOUT retraining (ledger rows 18/19). Prints an accuracy grid.

    python examples/capacity_sweep.py
"""

from __future__ import annotations

import numpy as np
import torch

from sili_quartz.tasks import RecallSpec, accuracy, constant_predictor_baseline, make_batch
from sili_quartz.titans import TitansMAC
from sili_quartz.titans.mac import recall_loss

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK, TRAIN_PAIRS, TRAIN_GAP, STEPS = 8, 4, 16, 1500
N_KEYS = N_VALUES = 64  # fixed above the largest binding count, so vocab never changes


def train(seed: int) -> TitansMAC:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    spec = RecallSpec(
        n_keys=N_KEYS, n_values=N_VALUES, n_pairs=TRAIN_PAIRS, n_queries=TRAIN_PAIRS, gap=TRAIN_GAP
    )
    m = TitansMAC(
        vocab=spec.vocab, d=64, n_heads=4, chunk=CHUNK, write_key_from_prev=True, tbptt_chunks=64
    ).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(STEPS):
        t, y = make_batch(spec, 64, rng)
        logits, _ = m(t.to(DEV))
        loss = recall_loss(logits, y.to(DEV))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m.eval()


@torch.no_grad()
def evaluate(m: TitansMAC, pairs: int, gap: int) -> tuple[float, float]:
    spec = RecallSpec(n_keys=N_KEYS, n_values=N_VALUES, n_pairs=pairs, n_queries=min(pairs, 8), gap=gap)
    t, y = make_batch(spec, 256, np.random.default_rng(999))
    logits, _ = m(t.to(DEV))
    return accuracy(logits, y.to(DEV)), constant_predictor_baseline(y)


def main() -> None:
    for seed in (0, 1):
        m = train(seed)
        print(f"\nseed {seed}: trained at pairs={TRAIN_PAIRS} gap={TRAIN_GAP}; d_k=64. rows=pairs, cols=gap")
        gaps = (16, 64, 256, 1024)
        print(f"{'pairs':>6} " + " ".join(f"{g:>8}" for g in gaps))
        for pairs in (2, 4, 8, 16, 32, 64):
            cells = []
            for g in gaps:
                acc, base = evaluate(m, pairs, g)
                cells.append(f"{acc:.3f}/{base:.2f}")
            print(f"{pairs:>6} " + " ".join(f"{c:>8}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
