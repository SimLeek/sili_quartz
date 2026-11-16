# TODO

## 0. Working agreements
- `main` is branch-protected: work on branches, open PRs, the user merges.
- Planned layout — every model family and every implementation of it lives
  side by side in ONE directory, so a reviewer sees the torch and sili versions
  of the same thing next to each other:

      sili_quartz/models/
        transformer.py        base (chunked attention)            torch
        rmt.py                B: passed state tokens              torch
        titans.py             C: memory as weights (+ switches)   torch
        combined.py           B + C (ARMT-style)                  torch
        *_sili.py             same models on sili kernels, same tests
      sili_quartz/titans/     (current location; migrates into models/ when rmt lands)


## 1. Memory-family tests: B, C, B+C, and why A fails them — DONE (PR #2). B cannot bind; C and B+C are the path
- C exists (`TitansMAC`, memory=True). Add **B**: RMT-style register of m
  memory tokens read/written per chunk, state carried across chunks, gradient
  via `tbptt_chunks`. Add **B+C** (both on). **A**: Transformer-XL cache of the
  previous chunk's K/V — include as the control that must FAIL beyond its reach.
- Shared task: recall with gap; sweep gap (16…1024) and live bindings (2…32).
  Pass = beats baseline beyond the window; the *shape* (where each family falls
  off) is the result. Ledger rows 18/19. `examples/capacity_sweep.py` runs the
  C half now (train once, evaluate longer streams / more pairs, no retrain).
- Prediction to falsify: C falls off at O(d_k) bindings; B at m×d; A at cache length.

## 1b. Long-run B+C vs C — DONE (row 30); next: a probe that does not saturate
- Rerun with 128-key/128-value vocab, train 16 pairs, stress 64 pairs, EQUAL STEPS
  (e.g. 40k) on the 2060 and 1660 Ti; log both step and minutes. That decides
  whether combined's lead grows with training or only converges earlier.
- P40 long runs: power cap 200 W (user, 2026-11-16: 51 °C at 150 W is far below
  limit); watch the guard csv. The 1660 Ti is the card nearest its abort (75 °C).

## 2. ATLAS switches — DONE for poly2 + omega; next: find poly2's ceiling
- Both landed (377db7e) with the capability matrix green on 3 GPUs. poly2
  saturated the 32-binding probe; omega alone gave 1.5×. Raise the probe:
  `capacity_sweep.py --features poly2` with N_KEYS=128 and pairs up to 128,
  on the P40 (large-batch job). Then `both`, then Muon / alt losses.
- Old notes:
- `window=c` (c=chunk): one weighted gradient over the chunk's (k,v) pairs
  instead of the per-token loop — memorizes context, and removes the scan.
- `features="poly_p"`. Re-derive ledger row 1 as "window stored exactly when
  keys are independent" BEFORE switching; that test doubles as the capacity probe.

## 2b. Two planned changes are TRADEOFFS, not wins — test them as such
Each gets a two-axis measurement (what it should improve, what it should
cost) rather than a pass/fail "better". Record both numbers in the ledger.

- **e-BPTT via blame buffer(s).** A buffer receives ~2% of forward activity
  continually and is backpropagated to once (or several such buffers at
  different rates). Expected: better long-term recall (credit reaches writes
  beyond the BPTT window) at the cost of SLOWER learning for anything inside
  the BPTT window. Test: long-gap accuracy vs steps AND in-window accuracy vs
  steps, side by side with full in-episode BPTT; the cost is the right-shift of
  the in-window curve.
- **Adam → sili importance (single-state optimizer).** Expected: worse
  accuracy/steps until the mean/variance control and eigenvalue stochastic
  annealing land (those may recover it). The benefit that is certain is RAM:
  ~2/3 of the full model's footprint for the same model, because sili keeps no
  backprop state. Test: same curves as above plus peak memory per parameter;
  a regression in steps-to-threshold is EXPECTED and must be quantified, not
  hidden. Adam remains the diagnostic reference.

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

## 5. Compute
- 3 GPUs online (laptop 1660 Ti; arch-sandbox P40 + 2060), `tools/run_batches.py`.
- 2 more GPUs are available via a LAN box + another computer, more after that.
  Not needed yet. If we do scale out, **track watts per job** alongside
  wall-clock in the batch logs, from the start.
