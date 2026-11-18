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
