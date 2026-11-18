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
1. **Seeds** — Done → [details](#done-1a1). Seed 1 never trains for either
   family (JOURNAL row 32).
2. **poly2** — Dropped (user, 2026-11-17) → [details](#done-1a2)
3. **Ablation tests** (rows 14/15/24) — Done → [details](#done-1a3)
4. **Seed-1 failure** (JOURNAL row 32): combined fails on *data stream 1* alone; titans
   only with init 1 + data 1; poly2 r16 collapses on seed 2. Hypothesis (user): early training + Adam without stochastic
   annealing; industry would discard such runs, sili may recover them. A hard
   data stream is an ASSET: keep it as a fixture and re-run it under the sili
   optimizer later to see if the optimizer fix heals it. Test: every seed
   escapes the start within N steps (expected to fail under Adam — mark xfail
   with the reason, don't paper over it).
5. **40k seeds for the omega arms** — Done → [details](#done-1a5)
6. **After this PR merges: port all three to sili CUDA kernels** — titans,
   combined, and the omega window (user, 2026-11-17: the data does not rule any
   of them out). Same tests per kernel; Adam is the known non-exact exception.
   The start-up fixture (1a.4) and the 2b tradeoff tests then run on sili.
No learning-rate schedules as fixes: that is an oracle solution.
No torch step-speed work: speed is not the limit now; speedups happen in sili.
No further B/C-family literature to fold in (user, 2026-11-16).

## 1b. Long-run B+C vs C — Done: combined leads → [details](#done-1b)

## 2. ATLAS switches — omega kept, poly2 dropped → [details](#done-2)

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
  - **Torch first pass (branch research/e-bptt-fake-backprop, 2026-11-17):**
    `models/ebptt.py`. Overriding a saved input during backward works
    (`saved_tensors_hooks`; tested on starmap / blob-walk z=t / noise inputs).
    On row 15 with the state detached every chunk: credit from a plain sum of
    remembered inputs 0.268, buffer 0.212. With the V-side Jacobian carried
    (I − c k kᵀ): trace 0.969, buffer-16 0.891 (pass). W_V-only credit 0.061.
    Sparse credit backward over a noise-chosen 1/4 of memory dims matches the
    dense credit (9/11 vs 6/8 seeds ≥ 0.95); |G|-chosen is biased and stalls
    (Bend proof: sili__new bend_proofs/noise_selected_credit). Per-chunk
    backward (`stream_backward`) makes memory flat in T: 22 MiB at T = 28…1036
    vs 45…1582 MiB for full BPTT. Row-15 test at constant memory: 0.980.
    Open: the in-window cost axis (running), larger d for the wall-time saving.
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
- Then: peridot-style (sili Tensor) port, then CUDA kernels one at a time,
  each passing the same tests. Adam is the known non-exact exception.
- Later: Omega distractor test (row 20), chunk sweep, MIRAS losses, MLP memory,
  Muon, XL cache (A) as a live control.

## 5. Compute
- Router LAN (DHCP, 192.168.1.0/24). Hosts are found by MAC, not IP:
  `tools/lan_hosts.py [name]` reads `~/.config/sili_quartz/lan_hosts` (kept out of
  this public repo), checks the ARP table, sweeps the wired /24 if needed.
  `run_batches.py` resolves `sandbox` and `ai0` this way: 5 GPU batches.
  `tools/lan_up.sh` is only for a direct cable with no router.
- GPUs: laptop 1660 Ti 6 GB; sandbox P40 24 GB + 2060; ai0 GTX 1080 8 GB +
  1660 SUPER 6 GB (driver 580). ai0 is the user's ttrpg machine: files in
  `~/claude_code` only, ask before sudo, GPU jobs only if asked.
- If we scale out, **track watts per job** alongside wall-clock in the batch
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
## 1b. Long-run B+C vs C (PR #3) — combined leads (JOURNAL row 30)
- 230-min runs, 32-pair probe: titans 30.2 vs combined 30.7 of 32 (windowed).
  [docs/results/2026-11-15-long-family-runs.md](docs/results/2026-11-15-long-family-runs.md)
- Equal 40k steps, 128-key vocab, train 16 pairs, stress 64 pairs: 30–40k
  window titans 55.4 ± 3.1 (min 43.4) vs combined 56.4 ± 1.5 (min 53.7).
  [docs/results/2026-11-16-equal-steps-64-pair-probe.md](docs/results/2026-11-16-equal-steps-64-pair-probe.md)
- Reading: combined is slightly higher in both experiments and far more stable
  (half the variance, much better worst case). Seeds: see [1a.1](#done-1a1).
- Lesson: compare windowed statistics (mean/sd/min over 10k steps), never single evals.

<a id="done-2"></a>
## 2. ATLAS switches: poly2 keys + Omega window (377db7e)
- `features="poly2"`: keys `[k; upper(kkᵀ)]`. `window=c`: one joint step over
  the chunk's c pairs, per-key normalized, scaled by 1/λ_max of the normalized
  Gram (differentiable). Ledger row 1 re-derived as "window stores exactly when
  keys are independent" before switching; that test doubles as the capacity probe.
- At d_k=64, 32-binding probe: poly2 saturated 32/32; omega alone 28.3 vs base ~19.
- At scale (128-key, 16 pairs) poly2 + omega **fails to train** (JOURNAL row 31,
  train acc 0.74 at 40k) — open as 1a.2.

<a id="done-1a1"></a>
## 1a.1 Seeds 0–3, titans vs combined (PR #3)
[docs/results/2026-11-16-seeds-and-poly2-at-scale.md](docs/results/2026-11-16-seeds-and-poly2-at-scale.md)
- Same program, `--seed` 0–3, 40k steps, 64-pair stress probe; seeds 1–3 three
  processes per GPU (titans P40 ~279 min, combined 2060 ~220 min).
- Seeds that trained (0, 2, 3), 30–40k windowed: titans 54.7, combined 56.3 of
  64; worst eval 43.4 vs 53.7. Combined ≥ titans for every trained seed.
- Seed 1 never trains for either family (combined collapses to a constant
  predictor; titans creeps to 0.07 train acc). titans s3 plateaus until ~10k.
  Spread is *whether/when training escapes the start* → JOURNAL row 32, plan 1a.4.

<a id="done-1a3"></a>
## 1a.3 Ablation tests for the required settings (PR #3)
`tests/test_mac.py::test_ablation_breaks_long_gap_recall` switches one setting off
in the verified long-gap config (gap 16, 3 pairs, 800 steps, seed 0, base 0.074;
full config 1.000) and requires accuracy below base + 0.5, the same line the
positive test must clear:
- previous-token keys off (row 14): 0.383 — partial, still far below 1.000.
- detach every chunk, so no gradient across the gap (row 15): 0.061, chance.
- alpha 0.5, a random-init-like forget gate (row 24): 0.061, chance.
`test_default_gates_are_not_random` (row 24, unit): defaults alpha = eta = 0,
not learnable parameters, identical across seeds.

<a id="done-1a2"></a>
## 1a.2 poly2 at scale — dropped (user, 2026-11-17)
[docs/results/2026-11-17-seven-seeds-split-omega.md](docs/results/2026-11-17-seven-seeds-split-omega.md)
- poly2 + omega stalls at full size (d_feat 2144, 13 GB) and at `poly_rank=16`
  (200 features, 1.06 GB).
- Alone: full poly2 trains 2/2 (2–4k 58.1, 48.2); rank 16 trains 1/2 (59.5, the
  best run, but seed 2 collapses to a constant on a stream every other arm trains).
- Omega alone trains 3/3 (2–4k 51.8 on each), so omega stays.
- Removed: `features`, `poly_rank`, `poly2()`, the poly2 unit/integration tests and
  the `--features/--poly-rank` flags.

<a id="done-1a5"></a>
## 1a.5 40k seeds for the omega arms (2026-11-17)

Titans + omega and combined + omega, seeds 0–6, 40k steps, same spec as the
7-seed runs. Results: [docs/results/2026-11-17-omega-forty-k-seeds.md](docs/results/2026-11-17-omega-forty-k-seeds.md).
- titans + omega: 7/7 train, 30–40k 56.9 ± 2.0 of 64, six seeds at 57.6–58.0.
- combined + omega: 6/7 (seed 5 stuck at acc 0.09), 56.4 ± 2.1, best worst
  case (window min 49.7).
- Plain: titans 55.2 ± 1.8, combined 55.9 ± 1.5, each 6/7.
- Start-up is not reproducible across GPUs: combined + omega s2 stalled on ai0
  (10k run) and trained at 4.5k on the 2060. The 1a.4 fixture needs pinned
  hardware or a failure count over many seeds.
- 1080 peak 77 °C (3 under the abort), 1660 SUPER 70, 2060 63, P40 51.
