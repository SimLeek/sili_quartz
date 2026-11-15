"""Key-value recall with a controllable gap, as a next-token task.

Stream per episode:
    k_1 v_1 k_2 v_2 ... k_n v_n  [BLANK]*gap  k_j1 ? k_j2 ? ...
Targets are -100 everywhere except at each queried key, where the target is
its value. gap >= chunk puts every pair outside the attention window of its
query, so only persistent memory can answer; gap = 0 keeps it in-context.

Keys, values and BLANK are disjoint id ranges, so the answer can never be
read off the query token itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class RecallSpec:
    n_keys: int = 16
    n_values: int = 16
    n_pairs: int = 4
    n_queries: int = 4
    gap: int = 0

    @property
    def blank(self) -> int:
        return self.n_keys + self.n_values

    @property
    def vocab(self) -> int:
        return self.blank + 1

    @property
    def length(self) -> int:
        return 2 * self.n_pairs + self.gap + 2 * self.n_queries


def make_batch(spec: RecallSpec, batch: int, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
    """Returns (tokens [B, T] long, targets [B, T] long with -100 for unscored)."""
    B, T = batch, spec.length
    toks = np.full((B, T), spec.blank, dtype=np.int64)
    tgts = np.full((B, T), -100, dtype=np.int64)
    for b in range(B):
        keys = rng.choice(spec.n_keys, size=spec.n_pairs, replace=False)
        vals = rng.integers(0, spec.n_values, size=spec.n_pairs) + spec.n_keys
        toks[b, 0 : 2 * spec.n_pairs : 2] = keys
        toks[b, 1 : 2 * spec.n_pairs : 2] = vals
        q = rng.choice(spec.n_pairs, size=spec.n_queries, replace=spec.n_queries > spec.n_pairs)
        base = 2 * spec.n_pairs + spec.gap
        for i, j in enumerate(q):
            toks[b, base + 2 * i] = keys[j]
            tgts[b, base + 2 * i] = vals[j]
    return torch.from_numpy(toks), torch.from_numpy(tgts)


def constant_predictor_baseline(targets: Tensor) -> float:
    """Accuracy of always predicting the most frequent target (the honest chance level)."""
    t = targets[targets != -100]
    return float(torch.bincount(t).max().item() / t.numel())


def accuracy(logits: Tensor, targets: Tensor) -> float:
    m = targets != -100
    return float((logits.argmax(-1)[m] == targets[m]).float().mean().item())


def distinct_predictions(logits: Tensor, targets: Tensor) -> int:
    """How many different answers the model gives on scored positions; <= 2 means collapse."""
    return int(logits.argmax(-1)[targets != -100].unique().numel())


def assert_gap_hides_pairs(spec: RecallSpec, chunk: int) -> None:
    """Every query must sit >= chunk tokens after the last pair, else attention can see it."""
    first_query = 2 * spec.n_pairs + spec.gap
    last_pair = 2 * spec.n_pairs - 1
    if first_query - last_pair <= chunk:
        raise AssertionError(f"gap {spec.gap} leaves pairs inside the {chunk}-token window")
