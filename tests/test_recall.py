import numpy as np
import pytest
import torch

from sili_quartz.tasks import (
    RecallSpec,
    assert_gap_hides_pairs,
    constant_predictor_baseline,
    make_batch,
)


def test_batch_shapes_and_ranges():
    spec = RecallSpec(gap=20)
    toks, tgts = make_batch(spec, 8, np.random.default_rng(0))
    assert toks.shape == tgts.shape == (8, spec.length)
    assert toks.max() < spec.vocab and toks.min() >= 0
    scored = tgts[tgts != -100]
    assert scored.numel() == 8 * spec.n_queries
    assert (scored >= spec.n_keys).all() and (scored < spec.blank).all()


def test_queries_are_keys_that_were_shown():
    spec = RecallSpec(gap=5)
    toks, tgts = make_batch(spec, 4, np.random.default_rng(1))
    for b in range(4):
        shown = dict(
            zip(
                toks[b, 0 : 2 * spec.n_pairs : 2].tolist(),
                toks[b, 1 : 2 * spec.n_pairs : 2].tolist(),
                strict=False,
            )
        )
        for pos in torch.nonzero(tgts[b] != -100).flatten().tolist():
            assert shown[toks[b, pos].item()] == tgts[b, pos].item()


def test_gap_region_is_blank():
    spec = RecallSpec(gap=7)
    toks, _ = make_batch(spec, 3, np.random.default_rng(2))
    assert (toks[:, 2 * spec.n_pairs : 2 * spec.n_pairs + 7] == spec.blank).all()


def test_baseline_is_max_class_frequency():
    t = torch.tensor([[5, -100, 5, 6], [-100, 7, 5, 5]])
    assert constant_predictor_baseline(t) == pytest.approx(4 / 6)


def test_gap_check():
    assert_gap_hides_pairs(RecallSpec(n_pairs=4, gap=16), chunk=16)
    with pytest.raises(AssertionError):
        assert_gap_hides_pairs(RecallSpec(n_pairs=4, gap=3), chunk=16)
