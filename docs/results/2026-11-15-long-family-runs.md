# Long family runs, 2026-11-15 → 16 (230 min each)

Spec: keys/values 64, train 8 pairs, gap 64, chunk 8, batch 64, seed 0, same
hyperparameters; stress probe every 500 steps at 32 pairs / gap 64. One family
per GPU under `tools/gpu_guard.sh`. Raw CSVs: `runs/long/*.csv` (laptop:
titans; arch-sandbox: combined, rmt). Columns:
`step,minutes,loss,acc,distinct,stress_acc,stress_recovered`.

**Equal wall-clock gave unequal steps** — titans 45k (1660 Ti), combined 81k
(2060), rmt 450k (P40) — so the comparison below is step-matched.

| step | titans stress acc | rec/32 | combined stress acc | rec/32 |
|---|---|---|---|---|
| 1000 | 0.673 | 21.6 | 0.209 | 6.7 |
| 2000 | 0.771 | 24.7 | 0.816 | 26.1 |
| 3000 | 0.823 | 26.3 | 0.863 | 27.6 |
| 5000 | 0.879 | 28.1 | 0.873 | 27.9 |
| 7500 | 0.851 | 27.2 | 0.946 | 30.3 |
| 10000 | 0.834 | 26.7 | 0.862 | 27.6 |
| 15000 | 0.919 | 29.4 | 0.939 | 30.1 |
| 20000 | 0.899 | 28.8 | 0.956 | 30.6 |
| 30000 | 0.927 | 29.7 | 0.946 | 30.3 |
| 40000 | 0.896 | 28.7 | 0.978 | 31.3 |
| 44000 | 0.956 | 30.6 | 0.963 | 30.8 |

| | titans | combined |
|---|---|---|
| steps to train acc ≥ 0.99 | 1500 | 2000 |
| steps to stress ≥ 0.5 / ≥ 0.9 | 1000 / 7000 | 2000 / 5500 |
| final stress acc | 0.922 @ 45k | 0.974 @ 81k |

RMT alone (m=8): max train acc 0.022, `distinct` = 1 for all 450k steps —
total collapse at this spec (short runs had plateaued at 0.33).

**Note (2026-11-16):** windowed over 30–45k steps: titans 30.2 ± 0.5 vs combined 30.7 ± 0.4 of 32 — combined ahead, by less than the single-eval comparisons below suggest.

**Reading.** Both families solve the training spec by ~2k steps. Combined
(B+C) leads titans (C) on the stress probe from ~2k steps on, reaches 0.9
earlier, and is more stable (titans oscillates 0.83–0.96). Its cost is the
slower first ~1k steps — the tradeoff predicted. The probe saturates near
32/32, so this does NOT yet show the lead *growing* with training; the
follow-up (`runs/long2/`, 128-key vocab, 64-pair probe, equal 40k steps)
is designed to.

**Thermal.** Peak under the guard: 1660 Ti 75 °C (nearest its 80 °C abort),
P40 51 °C at a 150 W cap, 2060 45 °C. P40 cap raised to 200 W for the
follow-up.
