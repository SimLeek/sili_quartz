# sili_quartz

Working models and runnable examples. Research, prototypes and the journal
live in `sili_peridot`; this repo only holds things that work and the tests
that keep them working.

## Status

**Titans memory-as-context in PyTorch, learning beyond its attention window.**

| run (`examples/train_recall.py`) | result |
|---|---|
| in-context recall, attention only (`--gap 0 --no-memory`) | 1.0000 @ 400 steps |
| beyond-window recall, attention only (`--gap 16 --no-memory`) | 0.056 — at the 0.071 baseline, i.e. no leak |
| beyond-window recall, with memory (`--gap 16`) | 4/4 seeds learn: 1.0000 @ 150 steps (seeds 1, 2); 0.90 @ 800 (seed 0); 0.87 @ 1500, still falling (seed 3) |

Beyond-window means every (key, value) pair sits ≥ 16 tokens before its
query with an 8-token attention chunk, so only the memory can answer.

**Capacity, measured** (`examples/capacity_sweep.py`, train once at 4 pairs,
evaluate without retraining): recovered bindings saturate at **≈16 ≈ d_k/4**
for d_k=64 (acc ≈ 16/n beyond that: 32 pairs → 0.47, 64 → 0.25). Decay with
gap is mild — under 1 point per gap doubling out to 1024 tokens — and is a
training-quality effect, not intrinsic. That is ATLAS's O(d_k) cliff seen
directly, and the reason polynomial key features are next.

Two settings were necessary and are the defaults: writes keyed on the
*previous* token's representation (so a pair is stored as key→value), and
backprop through the memory across the whole episode (`tbptt_chunks`
spanning the gap). Without either, accuracy stalls around 0.2–0.3.

## Layout

```
sili_quartz/titans/memory.py   TitansMemory  eqs 10-15, batched, closed-form delta rule, const or Law-5 gates
sili_quartz/titans/mac.py      TitansMAC     chunked memory-as-context transformer
sili_quartz/tasks/recall.py    key-value recall with a gap; honest baseline; leak check
examples/train_recall.py       the run above
docs/PROPERTIES.md             what a working Titans must do, sourced, mapped to tests
tests/                         fast unit tests; `-m integration` trains for real
```

## Run

```bash
pip install -e ".[dev]"
pytest                      # ~3 s
pytest -m integration       # ~2 min on a GPU
python examples/train_recall.py
```

## Roadmap

1. **This**: Titans working and learning in PyTorch, guarded by tests.
2. Replace kernels one at a time with sili CUDA kernels; each must pass the
   same unit tests as its OpenMP version. Known exception: Adam has no exact
   sili equivalent (sili's optimizer is single-state importance).
3. Grow `docs/PROPERTIES.md` from "todo" to "guarded" as the port proceeds.
