# Seeds 0–3 and poly2 alone at scale, 2026-11-16

Same spec and program as the equal-steps runs
([2026-11-16-equal-steps-64-pair-probe.md](2026-11-16-equal-steps-64-pair-probe.md)):
keys/values 128, train 16 pairs, gap 64, chunk 8, batch 64, 40k steps, stress
probe every 500 steps at 64 pairs / gap 64. Only `--seed` changes. Seed 0 is the
earlier run (`runs/long2/`); seeds 1–3 ran three processes per GPU (titans on
the P40, ~279 min; combined on the 2060, ~220 min, peak 60 °C / 98 W), CSVs in
`runs/seeds/`. Each seed CSV has one stale row + header from an aborted launch at
the top; rows after the last header are the run.

## Seeds: stress bindings recovered (of 64), windowed

| run | 10–20k | 20–30k | 30–40k mean ± sd | min 30–40k | train acc ≥ 0.99 at |
|---|---|---|---|---|---|
| titans s0 | 53.1 | 54.2 | 55.4 ± 3.1 | 43.4 | 500 |
| titans s1 | 1.2 | 1.2 | 1.4 ± 0.1 | 1.2 | never (0.07 at 40k) |
| titans s2 | 53.6 | 55.4 | 56.5 ± 1.0 | 54.6 | 1 500 |
| titans s3 | 33.7 | 48.0 | 52.3 ± 1.3 | 49.0 | 12 500 |
| combined s0 | 53.7 | 55.4 | 56.4 ± 1.5 | 53.7 | 500 |
| combined s1 | 0.5 | 0.5 | 0.5 ± 0.1 | 0.4 | never (0.008 at 40k) |
| combined s2 | 54.4 | 56.4 | 57.2 ± 1.0 | 55.1 | 1 000 |
| combined s3 | 52.8 | 54.3 | 55.4 ± 0.7 | 53.8 | 1 000 |

Over the seeds that trained (s0, s2, s3): titans 54.7, combined 56.3 mean of
30–40k means; worst eval 43.4 vs 53.7. Combined is above titans for seeds 0 and 2
and equal-or-above for 3 (55.4 vs 52.3), and its worst case is higher in all three.

**Seed 1 fails for both families.** Same seed → same data stream and the same
init for the shared modules, and both get stuck:
- combined s1 collapses to a constant predictor: loss 4.85 ≈ ln 128, 1 distinct
  prediction from step ~12k to 40k.
- titans s1 is not collapsed but barely moves: loss 4.86 → 4.35, distinct
  predictions 5 → 42, train acc 0.07 at 40k.
- titans s3 sits on a plateau (acc ≈ 0.09, loss ~3.0) until ~10k steps, then
  breaks through to 0.99 by 12.5k. combined s3 had no plateau.

So the spread across seeds is dominated by *whether and when training escapes
the start*, not by the final level: 2 of 8 runs never escaped in 40k steps, 1
escaped late. That is a trainability property of this setup to test for, not
noise to average away.

## poly2 alone (no omega) at scale

titans `--features poly2`, window 1, seed 0, 4000 steps on the P40 (200 W cap,
peak 59 °C, 124 min, ~13 GB). Train acc / stress recovered of 64:

| step | 500 | 1000 | 1500 | 2000 | 3000 | 4000 |
|---|---|---|---|---|---|---|
| poly2 only | 0.06 / 1.3 | 0.70 / 34.3 | 1.00 / 55.1 | 1.00 / 58.1 | 1.00 / 54.4 | 1.00 / 58.4 |
| titans s0 (identity keys) | 1.00 / 45.1 | 1.00 / 46.0 | 1.00 / 50.1 | 1.00 / 50.9 | 1.00 / 51.8 | 1.00 / 48.8 |
| titans s2 | 0.05 / 1.1 | 0.88 / 23.8 | 1.00 / 48.2 | 1.00 / 48.9 | 1.00 / 49.0 | 1.00 / 48.7 |
| poly2 + omega (long2) | — | 0.05 / 1.2 | 0.06 / 1.4 | — | — | 0.07 / 1.1 |

poly2 alone trains at this spec and recovers more bindings at 1.5–4k steps
(55–60) than any identity-key seed at the same step (≤ 52). Its 4k value is
already above titans' 30–40k windowed means. One seed and 4k steps only.

**poly2 + omega's failure at scale is not poly2's.** The remaining suspect is
omega (window = chunk) at 128 keys, or the combination. The omega-only and
base comparison runs on the laptop were stopped by the guard at 80 °C after
2 minutes (two jobs at once), so omega alone at this spec is **not yet measured**.

## Not yet measured
- omega-only at this spec (one job at a time on the laptop, or on the P40).
- Whether seed 1's failure is the data stream or the init (swap one at a time:
  `--seed` currently sets both).
- poly2 beyond 4k steps and on more seeds; its 13 GB footprint.
