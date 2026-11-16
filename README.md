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

**Capability matrix, 3 GPUs in parallel** (`tools/run_batches.py`, 2026-11-15):
all 8 integration tests pass — every switch combination (`base`, `poly`,
`omega`, `both`) still learns beyond-window recall; `poly2` recovers more
bindings than the plain matrix at equal d (the new capability); `omega`
costs none. Magnitudes at d=64, 32 live bindings: base recovers 18–19, **omega 28.3**, **poly2 32/32 (probe saturated)**. Laptop 1660 Ti 138 s, RTX 2060 203 s, Tesla P40 740 s (the
P40 is launch-latency-bound on tiny steps; give it the big-batch jobs).

**Family matrix** (same task, d=64, 32 live bindings): transformer fails as required;
**rmt recovers 1.5** (plateaus at 0.33 beyond-window: carries the set of values, never binds
key→value — 4 runs, m=4/16, ± curriculum); **titans 19.4**; **combined 23.7**. Details and
sources in `docs/PROPERTIES.md` rows 26–29.

## Layout

```
sili_quartz/models/
  transformer.py   A  chunked causal attention, no memory        (control)
  rmt.py           B  passed state tokens (RMT)
  titans.py        C  memory as weights (Titans MAC + switches)
  combined.py      B+C (ARMT-style)
  memory.py        TitansMemory: eqs 10-15, poly2 keys, Omega window, Law-5 gates
  common.py        shared Block + one segment mask
  *_sili.py        (planned) the same models on sili kernels, same tests
sili_quartz/tasks/recall.py    key-value recall with a gap; honest baseline; leak check
examples/                      train_recall.py --family …, capacity_sweep.py
tools/run_batches.py           integration batches across the 3 GPUs under tools/gpu_guard.sh
docs/PROPERTIES.md             what a working model must do, sourced, mapped to tests
docs/ARCHITECTURES.md          why these four families and not more
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
