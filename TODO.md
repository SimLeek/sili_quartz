# TODO (next session: 2026-11-16 15:00)

## 1. Memory-family tests: B, C, B+C, and why A fails them (integration, overnight)
- C exists (`TitansMAC`, memory=True). Add **B**: RMT-style register of m
  memory tokens read/written per chunk, state carried across chunks, gradient
  via `tbptt_chunks`. Add **B+C** (both on). **A**: Transformer-XL cache of the
  previous chunk's K/V — include as the control that must FAIL beyond its reach.
- Shared task: recall with gap; sweep gap (16…1024) and live bindings (2…32).
  Pass = beats baseline beyond the window; the *shape* (where each family falls
  off) is the result. Ledger rows 18/19. `examples/capacity_sweep.py` runs the
  C half now (train once, evaluate longer streams / more pairs, no retrain).
- Prediction to falsify: C falls off at O(d_k) bindings; B at m×d; A at cache length.

## 2. Omega window + polynomial keys in `TitansMemory` (first two ATLAS switches)
- `window=c` (c=chunk): one weighted gradient over the chunk's (k,v) pairs
  instead of the per-token loop — memorizes context, and removes the scan.
- `features="poly_p"`. Re-derive ledger row 1 as "window stored exactly when
  keys are independent" BEFORE switching; that test doubles as the capacity probe.

## 3. Training-regime constraints (user, 2026-11-15)
- **Sleep window (≤ 8 h/day, once a day) = annealing + health passes that
  improve the optimizer state.** Sparse backprop runs during BOTH wake and
  sleep. fp4-fix is a separate, later thing (only once we are at fp4). So the
  memory tier does not have to hold a whole day alone — weights learn all day;
  what sleep adds is the optimizer-state repair. Design the schedule around that.
- Sparse amortized learning with health/annealing may run far below max
  learning speed: closer to inference in some phases, to training in others.
  Memory-family results must be re-checked under THAT rate, not Adam's.
- Stochastic selection has won every training-related comparison so far; it
  will change these properties too — include it as an arm, not a footnote.

## 4. Still open from before
- PROPERTIES.md rows 14/15 ablation tests; row 24 (gates not random at init).
- Then: peridot-style (sili Tensor) port, then CUDA kernels one at a time,
  each passing the same tests. Adam is the known non-exact exception.
