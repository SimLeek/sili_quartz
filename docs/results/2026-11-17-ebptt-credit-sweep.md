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
