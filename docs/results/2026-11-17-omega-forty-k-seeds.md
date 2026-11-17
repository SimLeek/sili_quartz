# Omega at 40k steps, seeds 0–6: 2026-11-17

Spec as in [2026-11-17-seven-seeds-split-omega.md](2026-11-17-seven-seeds-split-omega.md)
plus `--window 8`: keys/values 128, train 16 pairs, gap 64, chunk 8, batch 64;
stress probe 64 pairs / gap 64, eval every 500. Raw CSVs: `runs/omega40/`.
Seeds 0–4 ran five at once per card on arch-sandbox (titans + omega on the P40,
combined + omega on the RTX 2060); seeds 5–6 ran one at a time on ai0 (titans +
omega on the GTX 1080, combined + omega on the GTX 1660 SUPER).

## Per seed: stress bindings recovered (of 64)

| seed | arm | 0.99 at | 2–4k | 5–10k | 30–40k mean ± sd | 30–40k min |
|---|---|---|---|---|---|---|
| 0 | titans + omega | 500 | 52.2 | 54.4 | 57.6 ± 1.0 | 54.3 |
| 0 | combined + omega | 500 | 51.7 | 53.7 | 57.6 ± 1.2 | 55.7 |
| 1 | titans + omega | 1 000 | 51.9 | 53.0 | 57.6 ± 1.4 | 55.1 |
| 1 | combined + omega | 1 500 | 51.6 | 53.1 | 56.7 ± 1.2 | 53.5 |
| 2 | titans + omega | 1 000 | 52.9 | 53.8 | 57.6 ± 1.4 | 54.7 |
| 2 | combined + omega | **4 500** | 6.3 | 44.8 | 52.2 ± 1.0 | 49.7 |
| 3 | titans + omega | 500 | 52.1 | 53.4 | 57.6 ± 1.3 | 54.0 |
| 3 | combined + omega | 500 | 51.4 | 53.2 | 57.4 ± 1.1 | 55.8 |
| 4 | titans + omega | 500 | 51.4 | 53.7 | 57.8 ± 1.4 | 55.4 |
| 4 | combined + omega | 500 | 51.5 | 53.2 | 56.9 ± 1.2 | 54.7 |
| 5 | titans + omega | 1 000 | 52.4 | 53.0 | 58.0 ± 1.1 | 55.5 |
| 5 | combined + omega | **never** (acc 0.09 at 40k) | 1.4 | 0.4 | 1.0 | 0.6 |
| 6 | titans + omega | 1 000 | 49.7 | 50.1 | 52.3 ± 3.3 | 44.3 |
| 6 | combined + omega | 1 500 | 51.1 | 53.7 | 57.7 ± 1.2 | 55.3 |

## All four arms, 7 seeds, 40k steps

| arm | seeds that train | 30–40k mean ± sd (trained seeds) | mean of window minima | worst eval | failed seed |
|---|---|---|---|---|---|
| titans | 6/7 | 55.2 ± 1.8 | 50.0 | 43.4 | 1 |
| combined | 6/7 | 55.9 ± 1.5 | 52.2 | 43.4 | 1 |
| titans + omega | **7/7** | **56.9 ± 2.0** | 53.3 | 44.3 | — |
| combined + omega | 6/7 | 56.4 ± 2.1 | **54.1** | **49.7** | 5 |

- **titans + omega is the only arm to train on all seven seeds.** Six of them
  land within 0.4 of each other (57.6–58.0). Seed 6 is the outlier (52.3, min
  44.3), and seed 6 is also the worst window for plain combined (43.4) and a
  weak one for plain titans (47.4).
- **combined + omega has the best worst case among seeds that train** (lowest
  window minimum 49.7; every other arm dips to 43–44), but it fails seed 5
  outright (the acc ≈ 0.09 plateau) and starts late on seed 2 (4.5k steps).
- **Paired, combined + omega vs titans + omega** (6 seeds both train): combined
  is higher on 1 of 6 (s6, +5.4), level on s0, lower on s1/s3/s4 (−0.2 … −0.9)
  and s2 (−5.4, the late start).
- **Omega vs no omega, titans:** higher on 5 of 6 seeds that both train (+0.9 …
  +5.3), lower on s6 (−3.4), and it rescues seed 1.
- **Omega vs no omega, combined:** higher on s0/s3/s6 (+1.2 … +4.5), level on s4,
  lower on s2 (−5.0); it rescues seed 1 and loses seed 5.

## Start-up is not reproducible across GPUs

Combined + omega seed 2 stalled for all 10k steps in the earlier run on ai0. In
this run, on the 2060, it trained at 4.5k. Same program, same seeds, different
card. Which seed fails to start depends on floating-point details, not only the
data stream, so the start-up fixture (plan 1a.4) has to pin the hardware, or
count failures over many seeds, rather than name one bad seed.

## Temperatures (guard logs, abort at 80 °C)

| GPU | jobs | wall time | peak | mean | peak power |
|---|---|---|---|---|---|
| P40 (200 W cap) | 5 at once | 2.8 h | 51 °C | 49 °C | 99 W |
| RTX 2060 | 5 at once | 2.2 h | 63 °C | 60 °C | 111 W |
| GTX 1080 (ai0) | 2 in sequence | 2.5 h | **77 °C** | 73 °C | 116 W |
| GTX 1660 SUPER (ai0) | 2 in sequence | 2.7 h | 70 °C | 66 °C | 73 W |

The 1080 peaked 3 °C under the abort, about 1 °C hotter than its previous run
alongside the user's game. The 1660 SUPER ran 1–4 °C hotter than before. The
tower is the place for more airflow.
