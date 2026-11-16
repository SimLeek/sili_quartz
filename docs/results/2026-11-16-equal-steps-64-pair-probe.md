# Equal-steps runs with a non-saturating probe, 2026-11-16

Spec: keys/values 128, train 16 pairs, gap 64, chunk 8, batch 64, **40k steps
each**, seed 0; stress probe every 500 steps at 64 pairs / gap 64. Raw CSVs in
`runs/long2/`. Under `tools/gpu_guard.sh`: titans on the 1660 Ti (peak 75 °C,
4.5 h), combined on the 2060 (peak 45 °C, 2.5 h), titans + poly2 + omega on the
P40 at a 200 W cap (peak 63 °C, 3 h).

Stress bindings recovered (of 64), statistics over each 10k-step window
(20 evals per window). **Single seed** — differences of ~1 binding are within noise.

| arm | 10–20k mean ± sd | 20–30k | 30–40k | min in 30–40k |
|---|---|---|---|---|
| titans | 53.1 ± 2.7 | 54.2 ± 2.2 | 55.4 ± 3.1 | 43.4 |
| combined | 53.7 ± 1.9 | 55.4 ± 1.8 | 56.4 ± 1.5 | 53.7 |
| titans + poly2 + omega | 0.1 ± 0.1 | 1.3 ± 0.7 | 8.3 ± 3.5 | 2.9 |

Both titans and combined solve the training spec (train acc 1.000) and are
still improving at 40k. Peaks are equal (58.3 vs 58.7).

**Reading.**
- **Combined leads titans.** Two independent experiments (32-pair and 64-pair
  probes) both put combined's windowed mean slightly above titans' (30.7 vs 30.2;
  56.4 vs 55.4), and combined's worst eval is far higher (53.7 vs 43.4) with about
  half the variance — the RMT tokens make the memory both slightly better and much
  more stable. Seeds 1–3 of each are running (`runs/seeds/`) to size the gap.
- Single eval points mislead in both directions (the first draft quoted 56.8 vs
  43.4 from one titans dip); compare windowed statistics.
- **poly2 + omega fails to train at this spec**: train acc 0.08 at 20k, first
  ≥ 0.5 at 36k, 0.74 at 40k. At d=64, poly2's feature dim is 64 + 2080 = 2144,
  so M is 64 × 2144 and every key is 33× wider than identity. The capability
  matrix (16-key vocab, 3 pairs) did not catch this; it is a regression at scale,
  not a capability win.

**Note on `2026-11-15-long-family-runs.md`.** Its point comparisons overstated
the gap at the 32-pair spec; the windowed means (30.2 vs 30.7) still put
combined ahead, by less.
