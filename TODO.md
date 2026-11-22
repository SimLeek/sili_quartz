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
6. **Port all three to sili CUDA kernels (now part of 5)** — titans,
   combined, and the omega window (user, 2026-11-17: the data does not rule any
   of them out). Same tests per kernel; Adam is the known non-exact exception.
   The start-up fixture (1a.4) and the 2b tradeoff tests then run on sili.
No learning-rate schedules as fixes: that is an oracle solution.
No torch step-speed work: speed is not the limit now; speedups happen in sili.
No further B/C-family literature to fold in (user, 2026-11-16).

## 1b. Long-run B+C vs C — runs done, not separated (ledger row 30, open) → [details](#done-1b)

## 2. ATLAS switches — omega kept, poly2 dropped → [details](#done-2)

## 2b. torch + Adam → sili + importance: will our fp4 setup train?
Both planned changes are TRADEOFFS, not wins: each gets a two-axis
measurement (what it should improve, what it should cost), both numbers in the
ledger, the cost quantified and never hidden.

- **Adam → sili importance (single-state optimizer).** Expected: worse
  accuracy/steps until the mean/variance control and eigenvalue stochastic
  annealing land. Certain benefit: RAM, **1/2** of the Adam footprint (weight +
  one importance value; sili keeps no backprop state; a conventional engine with
  a one-value optimizer would be 2/3). Adam remains the diagnostic reference.
- **e-BPTT** — torch first pass done (PR #4) → [details](#done-2b-ebptt).
  Open items are under 7.

1. **Bend proof first.** Prove that weights + optimizer state in fp4 with
   hadamard / additive LoRAs, plus annealing, can train and learn the recurrent
   MQAR, possibly requiring stochastic learning (stochastic rounding /
   selection). Not necessarily "as well" at equal size: information theory /
   compression may stop low bits from matching fp32 per parameter. If so, state
   the bound and the size that overcomes it: a larger fp4 model can trade
   precision-type complexity for more if-then (decision-type) complexity.
   Deliverable: the comparative model size in fp4 for equal capacity, and any
   other conversion facts that fall out (bits/parameter, rank of the additive
   correction).
2. **If the proof can't be done:** port a minimal fp4 hadamard / additive
   kernel to CUDA and train a transformer with it on the P40, then measure.
3. **If something is off in training:** look for NVIDIA's solution (their fp4
   training recipes) and biology's. Constraint: no more than **4 bits** of
   optimizer state per parameter. (The user had Gemini estimate how many bits
   per parameter each neurochemical is equivalent to.)

## 2c. Sili energy port and model collapse
Model collapse: a language model trained on another LM's samples, chained
generation after generation, loses rare words first (the tails vanish) and
eventually repeats forever (Shumailov et al., Nature 2024, "AI models collapse
when trained on recursively generated data"). Raising sampling temperature does
not fix it: it keeps the tails but loses sentence structure. The claim to test:
even a small dose of neuron energy RL in the model that *generates* the data
(per-neuron temperature, not token temperature) stops the collapse.
Not stochastic annealing: that rescues the optimizer state in self-training.

1. **Torch energy.** Port sili's `EnergyDynamics` (sili__new `sili/energy.py`;
   the older torch version is behind it). Mostly a library rename sili → torch.
   The one real difference: sili's `abs` backward has slope 1 at exactly 0
   (`sili/cpu.py` `abs_backward`, so neurons can wake from zero), torch's has 0.
   Use `torch.where(x == 0, x, x.abs())`. Unit tests: forward/backward match
   sili on fixed inputs, including h = 0.
2. **Model.** "Some model", kept outside the codebase: Qwen3.5-0.8B or 2B
   (hybrid recurrent: 3 linear-attention layers per full-attention layer;
   recurrent states are sensitive to small float errors, so harder to quantize
   and the more useful case). The user's download is GameScaling-0.8B
   (`~/claude_code/GameScaling-0.8B`, a Qwen3.5-0.8B finetune on games:
   keyboard/mouse + vision, for robonet later). Work on a COPY; keep the
   original as is (slow to re-download). P40: fp32 (no fast bf16/fp16); for
   fp16, keep fp16 in global memory and convert to fp32 in local memory per op.
   Later: A_lora ⊙ hadamard(B_low_bit) + C_lora for the recurrent parts.
3. **Two models talking to each other.**
   - 2060: runs a copy for inference with neuron energy accumulating,
     occasionally forcing firing / acting as per-neuron temperature.
   - P40: trains with energy (LM loss partly mixed with the energy RL aux
     loss) on the 2060's text, and sends its own text responses back.
   - Turns: a simple end token ends a message and passes the turn; max message
     length a fixed limit in 1k–4k tokens.
4. **Collapse metrics, per round:** rare-token mass (bottom tail of the
   original model's unigram distribution), distinct-n, longest repeated span /
   repetition rate, perplexity on fixed real held-out text, a structure check
   (original model's perplexity on the samples).
5. **Arms:** energy off at T = 1 (expected: collapse by round n); temperature
   sweep (expected: tails kept, structure lost); energy dose sweep from tiny
   (1e-6, expected useless) up, to find the line. Pass = no collapse through
   n + k rounds (k a fixed budget) at some dose, structure kept. Seeds = same
   program run N times. Results to `docs/results/`, ledger row.

## 3. Stochastic annealing + health pass: "sleep"
- **Sleep window (≤ 8 h/day, once a day) = annealing + health passes that
  improve the optimizer state.** Sparse backprop runs during BOTH wake and
  sleep. fp4-fix is a separate, later thing (only once we are at fp4). So the
  memory tier does not have to hold a whole day alone: weights learn all day;
  sleep adds the optimizer-state repair. Design the schedule around that.
- Sparse amortized learning with health/annealing may run far below max
  learning speed: closer to inference in some phases, to training in others.
  Memory-family results must be re-checked under THAT rate, not Adam's.
- Stochastic selection has won every training-related comparison so far; it
  will change these properties too: include it as an arm, not a footnote.

## 4. HDD hippocampus
1. Port sili cpu so that forward can add contribution to one subset of input
   rows while backward adds gradient + optimizer updates to another subset.
2. fp4 hadamard working in VRAM on an MQAR example.
3. A million × million recurrent transformer block, backprop through a random
   subset of rows where the subset is time-related (nearby time entries are
   chosen together, as in e-BPTT's smoothed noise). Backed by the sili database
   in HDD memory.

## 5. Port kernels
Port the required kernels and code one by one, each passing the same tests
(Adam is the known non-exact exception). Includes 1a.6 (titans, combined,
omega window on sili CUDA) and the peridot-style sili Tensor port. So that we can:
1. Convert Qwen3.8-27B to fp4 with fp4 + hadamard + additive as close to the
   original weights as possible (and the optimizer state if we can get it,
   otherwise a fake, relatively high optimizer state). Download models onto
   the sandbox (Bee), which has the free space.
2. Get it running by transferring kernels one by one.
3. Train it with intrinsic RL: energy, stochastic annealing.

## 6. Energy for RL task solving
Get energy working for RL task solving too. Reading: "Energy-Based Reasoners:
Your Recurrent Reasoner Should Be an Energy-Based Model".

## 7. Random top-k with error feedback
Implement and test random top-k with error feedback on small MQAR recurrent
transformer models. See sili__new `bend_proofs/noise_selected_credit/README.md`
(error feedback section). Open question: does a cheap, correct error feedback
exist for this setting, or not?
- e-BPTT items still open: clean in-window test (episode inside one chunk);
  more steps / seeds at gap 64; larger d for the wall-time saving; K-side
  Jacobian.

## Backlog: integration tests + advancements from other models
- Omega distractor test (row 20), chunk sweep, MIRAS losses, MLP memory, Muon,
  XL cache (A) as a live control.

## Reference: compute
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
## 1b. Long-run B+C vs C (PR #3) — combined narrowly ahead; still open (ledger row 30)
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

<a id="done-2b-ebptt"></a>
## 2b. e-BPTT torch first pass (PR #4, 2026-11-17)
[docs/results/2026-11-17-ebptt-credit-sweep.md](docs/results/2026-11-17-ebptt-credit-sweep.md)
`models/ebptt.py`. Overriding a saved input during backward works
(`saved_tensors_hooks`; tested on starmap (no temporal association), 2D slices of smoothed 3D gaussian noise, temporal-only smoothed noise, white noise).
On row 15 with the state detached every chunk: credit from a plain sum of
remembered inputs 0.268, buffer 0.212. With the V-side Jacobian carried
(I − c k kᵀ): trace 0.969, buffer-16 0.891 (pass). W_V-only credit 0.061.
Sparse credit backward over a noise-chosen 1/4 of memory dims matches the
dense credit (9/11 vs 6/8 seeds ≥ 0.95). |G|-chosen stalls on 2 of 3 seeds
and its choice is biased (measured: top 8 dims take 29 % of picks), but
that the bias CAUSES the stall is not shown. The Bend proof
(sili__new bend_proofs/noise_selected_credit) only shows the noise
choice is unbiased; error feedback is the known fix for biased top-k. Per-chunk
backward (`stream_backward`) makes memory flat in T: 22 MiB at T = 28…1036
vs 45…1582 MiB for full BPTT. Row-15 test at constant memory: 0.980.
Cost axis: steps to 0.95 vs full BPTT ×5–10 at gap 16; at gap 64 BPTT 3/3
by 250, e-BPTT 1/3 by 750 and 2/3 stuck ~0.4 (cost grows with gap).
Open: clean in-window test (episode inside one chunk); more steps / seeds
at gap 64; larger d for the wall-time saving.
