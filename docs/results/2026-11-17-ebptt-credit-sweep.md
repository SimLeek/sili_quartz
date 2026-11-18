# e-BPTT credit sweep on ledger row 15: 2026-11-17

Task: row 15's test (3 pairs, 3 queries, gap 16, chunk 8, d 64, batch 32,
Adam 3e-3), but 3000 steps, eval every 250, seeds 0 and 1. Memory state
detached every chunk (`tbptt_chunks=1`) unless stated. Code:
`sili_quartz/models/ebptt.py`, options on `TitansMemory(credit=...)`. Script
and raw CSVs: `runs/ebptt_sweep/`.

Noise prior (user's idea): remembered inputs multiplied by
`1[|x| > τ·rms(x)] · Φ(z_t)`, z_t a per-dimension gaussian drifting with a
16-token correlation time, so credit stays on similar inputs for a while. It is
a soft mask (multiplied, not thresholded), so it does not cut compute yet.

| arm | Jacobian carry | noise prior | seed | first ≥ 0.95 | acc at 3000 | best |
|---|---|---|---|---|---|---|
| full BPTT (tbptt 8) | — | — | 0 / 1 | 250 / 250 | 1.000 / 1.000 | 1.000 / 1.000 |
| carry_full | exact | — | 0 / 1 | 1000 / 1000 | 0.990 / 0.986 | 0.992 / 0.990 |
| carry_none (pure spoof) | none | — | 0 / 1 | — / — | 0.066 / 0.057 | 0.298 / 0.280 |
| none_noise_t0 | none | inputs, τ 0 | 0 / 1 | — / — | 0.066 / 0.065 | 0.307 / 0.241 |
| none_noise_t05 | none | inputs, τ 0.5 | 0 / 1 | — / — | 0.066 / 0.105 | 0.272 / 0.293 |
| full_noise_t05 | exact | inputs, τ 0.5 | 0 / 1 | — / — | 0.421 / 0.896 | 0.733 / 0.896 |
| masked_carry | on noise-masked key dims | — | 0 / 1 | 2000 / — | 0.954 / 0.066 | 0.979 / 0.327 |
| masked_noise_t05 | on noise-masked key dims | inputs, τ 0.5 | 0 / 1 | 750 / — | 0.926 / 0.273 | 0.984 / 0.314 |
| tbptt2_none | none | — (2 chunks real BPTT) | 0 / 1 | — / — | 0.064 / 0.065 | 0.098 / 0.232 |

Chance (constant predictor) is 0.074.

- Every arm without a carry peaks at 0.24–0.31 around step 250 and then falls
  back to chance by 3000, on both seeds. Two chunks of real BPTT on top do not
  change that.
- Exact carry: both seeds pass 0.95 by step 1000 and stay there.
- Noise-masked carry: one of two seeds reaches 0.95 (step 2000, or 750 with the
  input mask too), the other ends at chance or 0.27. n = 2 cannot separate these
  from the exact carry or from each other; they need more seeds.
- Noise mask on the inputs, on top of the exact carry: neither seed reaches 0.95
  (0.42 falling, 0.90 rising).
- GPU peaks: laptop 1660 Ti 69 °C, 1080 54, 1660 SUPER 51, 2060 51, P40 40
  (200 W cap). Wall times are not comparable across arms (different cards, 1–5
  jobs per card).

## Cost per arm

Analytic multiply-adds per token per batch row, d_in = d_mem = d = 64,
chunk L = 8, window 1. The full Jacobian dM_t/dM_{t-1} is never formed: it
is (I − c k kᵀ) acting on M from the right, i.e. I − step·Hessian of the
write's own loss ½‖Mk − v‖², so applying it to the trace is one
vector-matrix product and one outer product.

| piece | MACs per token | note |
|---|---|---|
| the memory write itself (Mk, outer product, update) | 3d² ≈ 12k | reference |
| trace sums E_V, E_K (all trace arms) | 2d² ≈ 8k | |
| exact carry (kᵀE_V, then outer) | +2d² ≈ 8k | the "Jacobian" cost |
| noise-masked carry (soft mask, as run) | +2d² + O(d) ≈ 8k | no saving until the mask is hard; at density p ≈ 2p·d² |
| noise mask on inputs | O(d) | negligible |
| credit backward at each detach (G·E_V, Gᵀ·E_K) | 2d³ / L ≈ 66k | **largest piece**, once per chunk |
| buffer-16: rebuild E from the buffer at each detach | 2·16·d² / L ≈ 16k | plus sort/gather overhead |
| explicit Jacobian, if it were formed (d² × d²) | d⁴ ≈ 17M | what generic RTRL would need |

The credit backward could drop to 2n·d²/L with a buffer of n (compute
(G·ck)·xᵀ instead of forming E), 4× less for n = 16.

Measured on the laptop 1660 Ti, batch 32, 10 steps after warm-up (peak
memory above the resident model):

| arm | gap 16 (T = 28): ms/step | MiB | gap 256 (T = 268): ms/step | MiB |
|---|---|---|---|---|
| full BPTT across the episode | 91.8 | 44.8 | 915.4 | 411.7 |
| detach every chunk, no credit | 61.0 | 22.8 | 467.6 | 146.1 |
| pure spoof (trace, no carry) | 71.7 | 26.6 | 503.0 | 183.6 |
| exact carry | 77.7 | 26.6 | 536.8 | 183.6 |
| buffer-16 | 91.4 | 26.1 | 677.7 | 183.5 |

At gap 256 the exact carry takes 59 % of BPTT's time and 45 % of its memory.
Memory still grows with T in every arm, because the loss is summed over the
episode and backpropagated once, which keeps every chunk's own graph until
the end. The trace removes only the cross-chunk part. Backpropagating each
chunk's loss as soon as the chunk ends would make memory O(1) in T; the credit
node already allows that, because nothing but E_V/E_K crosses a chunk boundary.

## Sparse credit backward (exact carry, top-p memory dims), 3 seeds

`credit_density=p` keeps the top p of M's key dims for G·E_V and of its value
dims for Gᵀ·E_K, scored by |G| (`grad`), |G|·Φ(z) (`grad_noise`) or Φ(z)
alone (`noise`; z drifts with a 16-token correlation time). Batched top-k, then
a smaller dense matmul: credit backward 2p·d³ per chunk instead of 2d³. Raw CSVs:
`runs/ebptt_sweep/sbw/`.

| arm | p | credit MACs / token | first ≥ 0.95 (s0/s1/s2) | acc at 3000 | runs ≥ 0.95 at 3000 |
|---|---|---|---|---|---|
| exact carry, dense | 1 | ≈ 66k | 1000 / 1000 / 2500 | 0.990 / 0.986 / 0.974 | 3/3 |
| grad | 0.25 | ≈ 16k | — / — / 2250 | 0.906 / 0.079 / 0.967 | 1/3 |
| grad_noise | 0.25 | ≈ 16k | — / 2500 / — | 0.064 / 0.974 / 0.893 | 1/3 |
| **noise** | 0.25 | ≈ 16k | 1500 / 1000 / 2000 | 0.995 / 0.970 / 0.973 | **3/3** |
| grad_noise | 0.125 | ≈ 8k | — / 2500 / 2250 | 0.086 / 0.848 / 0.966 | 1/3 |
| grad_noise | 0.5 | ≈ 33k | 1250 / 1250 / — | 0.975 / 0.986 / 0.862 | 2/3 |

- Noise-only selection at 25 % is the only sparse arm whose three seeds all end
  ≥ 0.97 with no fall-back; its curves rise like the dense carry's.
- Every arm that scores by |G| has at least one seed that stalls or collapses
  (sbw_grad_p25 s1 never leaves chance; sbw_gn_p25 s0 reaches 0.54, then falls
  to chance after step 1500; sbw_gn_p125 s0 peaks at 0.35, ends at 0.09).
- n = 3 per arm. GPU peaks: laptop 66 °C, 1080 55, 2060 52, 1660 SUPER 51.

## More seeds: noise-scored sparse credit vs the dense carry

Same task and 3000 steps. Seeds 0–2 from the table above, the rest new. Raw
CSVs: `runs/ebptt_sweep/seeds/`.

| arm | credit MACs / token | seeds | acc at 3000, per seed | ≥ 0.95 at 3000 | ever ≥ 0.95 | median first ≥ 0.95 |
|---|---|---|---|---|---|---|
| dense carry | ≈ 66k | 0–7 | .990 .986 .974 .973 .938 .999 **.059** .979 | 6/8 | 7/8 | 1250 |
| noise, p = 0.25 | ≈ 16k | 0–10 | .995 .970 .973 .992 **.065** .996 .975 .995 .938 .979 .970 | 9/11 | 9/11 | 1750 |
| noise, p = 0.125 | ≈ 8k | 0–7 | .957 .641 .984 .996 .376 .806 .367 .996 | 4/8 | 4/8 | 1125 |

- Noise at p = 0.25 and the dense carry are not separated: 9/11 vs 6/8 runs end
  ≥ 0.95. Each arm has one run that never leaves chance from the first eval
  (dense s6, noise s4), the start-up failure seen in every arm of the long runs
  (JOURNAL row 32). The other short-of-0.95 run in each arm is still rising at
  3000 (noise s8: 0.42 → 0.94; dense s4 0.938).
- At p = 0.125, 4 of 8 reach ≥ 0.95. Of the rest, two sit at 0.37 for all 3000
  steps and two are still rising (0.64, 0.81).
- GPU peaks: 1080 57 °C, 1660 SUPER 53, 2060 52, P40 43 (200 W cap).

## Per-chunk backward: memory flat in context length

`ebptt.stream_backward` runs the model one chunk at a time (`TitansMAC.stream`)
and backpropagates each chunk's loss as soon as the chunk ends: the same
gradient as the episode-summed loss (unit test, titans and combined), but only
one chunk's graph is alive. The previous chunk's last token, which keys the
next chunk's first write, is now detached at each tbptt cut too (row 15 trace
0.969 → 0.947 from that; buffer and ablations unchanged).

Laptop 1660 Ti, batch 32, 5 steps after warm-up:

| T | full BPTT | noise p25, one backward | noise p25, per-chunk | dense carry, per-chunk |
|---|---|---|---|---|
| 28 | 92 ms, 45 MiB | 79 ms, 26 MiB | 72 ms, 22 MiB | 62 ms, 22 MiB |
| 268 | 935 ms, 410 MiB | 636 ms, 187 MiB | 534 ms, 22 MiB | 481 ms, 22 MiB |
| 1036 | 3789 ms, 1582 MiB | 2614 ms, 699 MiB | 1996 ms, 22 MiB | 1726 ms, 22 MiB |

- Peak memory is 22 MiB at every T with per-chunk backward.
- Time is linear in T for every arm (each token is processed); per-chunk is
  about half of full BPTT.
- At d = 64 the noise top-k is slower in wall time than dense: top-k and
  gather cost more than the 4× smaller matmul saves on a latency-bound GPU.
  The multiply-add saving is real; wall time should follow at larger d or in
  sili's kernels, not measured here.
- Row-15 test at constant memory (noise p25, per-chunk, 1500 steps): 0.980.

## The tradeoff (TODO 2b): steps to 0.95 vs full BPTT, by gap

3 seeds, 3000 steps, batch 32. e-BPTT arm = noise p25 sparse credit with
per-chunk backward (constant memory). Raw CSVs: `runs/ebptt_sweep/axes/`.

| gap | full BPTT (untruncated): first ≥ 0.95 · acc at 3000 | e-BPTT: first ≥ 0.95 · acc at 3000 |
|---|---|---|
| 0 | — — — · 0.783 0.792 0.788 | 1250 — — · 0.990 0.796 0.777 |
| 16 | 250 250 250 · 1.000 ×3 | 1250 — 2750 · 0.987 0.940 0.948 |
| 64 | 250 250 250 · 1.000 ×3 | — — 750 · 0.441 0.383 0.990 |

- **Gap 16:** e-BPTT needs 5–10× the steps of full BPTT; one seed is still
  rising at 0.94.
- **Gap 64:** full BPTT solves all three seeds by step 250. e-BPTT solves one
  by 750; the other two sit at 0.38–0.61 for all 3000 steps. The step cost
  grows with the gap. (A first gap-64 BPTT arm with `tbptt_chunks=8`, 64
  tokens, shorter than the ~70-token pair-to-query span, stayed at chance:
  truncated below the gap, not a BPTT result.)
- **Gap 0 does not measure the e-BPTT cost.** At chunk 8 the first query
  lands in the pairs' own chunk, which only attention can answer. Full BPTT,
  measured per query: that query 0.37 (≈ 1/3, guessing among the three values),
  the two memory-answered queries 1.000 → the 0.79 plateau. One e-BPTT seed
  learned it (0.99). A clean in-window test needs the whole episode inside one
  chunk.
- Memory for the same runs: full BPTT grows with T (45 → 1582 MiB from T = 28
  to 1036, table above); e-BPTT stays at 22 MiB.
