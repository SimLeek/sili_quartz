# TODO

Open work first. Finished items keep a one-line entry here and their full
record under [Done](#done) at the bottom; results live in `docs/results/`.

## 0. Working agreements
- `main` is branch-protected: work on branches, open PRs, the user merges.
- Every model family and every implementation of it lives side by side in
  `sili_quartz/models/`, so the torch and sili versions sit next to each other:

      transformer.py        A: base (chunked attention)         torch
      rmt.py                B: passed state tokens              torch
      titans.py             C: memory as weights (+ switches)   torch
      combined.py           B + C (ARMT-style)                  torch
      *_sili.py             same models on sili kernels, same tests

## 1. Memory-family tests (B, C, B+C, A as control) — Done → [details](#done-1)

## 1a. PLAN (user, 2026-11-16), in order
1. **Seeds**: run the same program several times (`--seed`), several processes per
   GPU; steps are logged so curves are matched afterwards — no step matching, no
   separate harness. Seeds 1–3 of titans (P40) and combined (2060) running,
   `runs/seeds/`.
2. **poly2 at scale**: diagnose why poly2 + omega barely trains at the 128-key /
   16-pair spec (d_feat = 2144; 13 GB vs 1.2 GB for identity keys); separate
   poly2 from omega. Base / omega-only / poly2-only runs in `runs/poly/`.
3. **Ablation tests** for the two required settings (ledger rows 14/15) and the
   gates-not-random-at-init test (row 24).
No learning-rate schedules as fixes: that is an oracle solution.

## 1b. Long-run B+C vs C — Done: combined leads → [details](#done-1b)

## 2. ATLAS switches poly2 + omega — Done at d_k=64 → [details](#done-2); open: poly2 at scale (1a.2)

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
  **1/2** of the Adam footprint for the same model — weight + one importance
  value, and sili keeps no backprop state (a conventional engine with a
  one-value optimizer would be 2/3). Test: same curves as above plus peak
  memory per parameter; a regression in steps-to-threshold is EXPECTED and must
  be quantified, not hidden. Adam remains the diagnostic reference.

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
- PROPERTIES.md rows 14/15 ablation tests; row 24 (gates not random at init) — see 1a.3.
- Step speed: runs are kernel-launch-bound (~2.5 steps/s). Profile one step,
  then torch.compile / CUDA Graphs / vectorized write, with a parity test.
  (Proposed, awaiting go-ahead.)
- Then: peridot-style (sili Tensor) port, then CUDA kernels one at a time,
  each passing the same tests. Adam is the known non-exact exception.
- Later: Omega distractor test (row 20), chunk sweep, MIRAS losses, MLP memory,
  Muon, XL cache (A) as a live control.

## 5. Compute
- 3 GPUs online (laptop 1660 Ti; arch-sandbox P40 + 2060), `tools/run_batches.py`.
  Direct link: `sudo tools/lan_up.sh <host-number> [iface]` (laptop 2, sandbox 1).
- 2 more GPUs are available via a LAN box + another computer, more after that.
  If we scale out, **track watts per job** alongside wall-clock in the batch
  logs, from the start.
- P40 long runs: power cap 200 W (51 °C at 150 W is far below its limit); watch
  the guard csv. The 1660 Ti is the card nearest its abort (75 °C).

---

# Done

<a id="done-1"></a>
## 1. Memory-family tests: B, C, B+C, and why A fails them (PR #2)
Result: **B cannot bind; C and B+C are the path.** Ledger rows 18/19, 29;
`docs/ARCHITECTURES.md` for the taxonomy.
- Built **B** (`models/rmt.py`, RMT register of m memory tokens read/written
  per chunk, gradient via `tbptt_chunks`), **C** (`models/titans.py`,
  `TitansMAC`), **B+C** (`models/combined.py`), **A** (`models/transformer.py`)
  as the control that must fail beyond its reach. One `segment_mask` serves all.
- Shared task: recall with gap; sweep gap and live bindings.
  `examples/capacity_sweep.py` trains once and evaluates longer streams / more pairs.
- Prediction tested: C falls off at O(d_k) bindings (confirmed, C ≈ d_k/4 ≈ 16);
  B at m×d (falsified — B never binds: plateaus at 0.33, a bag of values for any
  m or curriculum).
- Family capacity at 32 bindings: rmt 1.5, titans 19.4, combined 23.7
  (`tests/test_mac.py::test_family_capacity_probe`).

<a id="done-1b"></a>
## 1b. Long-run B+C vs C (PR #3) — combined leads (ledger row 30)
- 230-min runs, 32-pair probe: titans 30.2 vs combined 30.7 of 32 (windowed).
  [docs/results/2026-11-15-long-family-runs.md](docs/results/2026-11-15-long-family-runs.md)
- Equal 40k steps, 128-key vocab, train 16 pairs, stress 64 pairs: 30–40k
  window titans 55.4 ± 3.1 (min 43.4) vs combined 56.4 ± 1.5 (min 53.7).
  [docs/results/2026-11-16-equal-steps-64-pair-probe.md](docs/results/2026-11-16-equal-steps-64-pair-probe.md)
- Reading: combined is slightly higher in both experiments and far more stable
  (half the variance, much better worst case). Seeds (1a.1) size the gap.
- Lesson: compare windowed statistics (mean/sd/min over 10k steps), never single evals.

<a id="done-2"></a>
## 2. ATLAS switches: poly2 keys + Omega window (377db7e)
- `features="poly2"`: keys `[k; upper(kkᵀ)]`. `window=c`: one joint step over
  the chunk's c pairs, per-key normalized, scaled by 1/λ_max of the normalized
  Gram (differentiable). Ledger row 1 re-derived as "window stores exactly when
  keys are independent" before switching; that test doubles as the capacity probe.
- At d_k=64, 32-binding probe: poly2 saturated 32/32; omega alone 28.3 vs base ~19.
- At scale (128-key, 16 pairs) poly2 + omega **fails to train** (ledger row 31,
  train acc 0.74 at 40k) — open as 1a.2.
