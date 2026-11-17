# Seven seeds, seed split, omega alone, small poly2: 2026-11-16 → 17

Spec as in [2026-11-16-seeds-and-poly2-at-scale.md](2026-11-16-seeds-and-poly2-at-scale.md):
keys/values 128, train 16 pairs, gap 64, chunk 8, batch 64; stress probe 64
pairs / gap 64. Raw CSVs: `runs/seeds_ai0/` (seeds 4–6, ai0), `runs/split/`,
`runs/omega/`, `runs/poly/`, `runs/polyr/`.

## Seeds 0–6, 40k steps: stress bindings recovered (of 64), 30–40k window

| seed | titans mean ± sd | titans min | combined mean ± sd | combined min | combined − titans |
|---|---|---|---|---|---|
| 0 | 55.4 ± 3.1 | 43.4 | 56.4 ± 1.5 | 53.7 | +1.0 |
| 1 | 1.4 (never trains) | 1.2 | 0.5 (collapses) | 0.4 | — |
| 2 | 56.5 ± 1.0 | 54.6 | 57.2 ± 1.0 | 55.1 | +0.7 |
| 3 | 52.3 ± 1.3 | 49.0 | 55.4 ± 0.7 | 53.8 | +3.1 |
| 4 | 53.9 ± 1.5 | 50.3 | 57.1 ± 1.3 | 54.3 | +3.2 |
| 5 | 57.1 ± 1.5 | 55.0 | 56.3 ± 1.3 | 52.8 | −0.8 |
| 6 | 55.7 ± 2.5 | 47.4 | 53.2 ± 3.5 | 43.4 | −2.5 |

Seeds 4–6 ran one per card on ai0 (titans on the GTX 1080, combined on the GTX
1660 SUPER), ~205–250 min each.

Over the six seeds that trained (seed 1 excluded for both):
- titans 55.15 ± 1.79, combined 55.93 ± 1.48 (mean of window means ± sd across seeds).
- mean of window minima: titans 50.0, combined 52.2. Worst single eval: both 43.4.
- Paired by seed: combined higher in 4 of 6 (mean difference +0.8, range −2.5 … +3.2).
- Escaping the start: both fail seed 1 only (6/7 each). titans s3 took 12.5k steps;
  every combined run that trained reached 0.99 train accuracy by 1.5k.

Seeds 4–6 shrink the lead seen with seeds 0, 2, 3 (+1.6 then; +0.8 now). Seed 6
is the first run where combined has the worse window and the worse minimum.

## Seed split: init seed vs data seed, 20k steps (2060, four at once)

| run | `--seed` (init) | `--data-seed` | trains? | 0.99 at | 10–20k rec |
|---|---|---|---|---|---|
| combined | 1 | 0 | yes | 1 000 | 50.3 |
| combined | 0 | 1 | **no** (acc 0.008) | — | 0.5 |
| titans | 1 | 0 | yes | 2 500 | 52.8 |
| titans | 0 | 1 | yes | 3 000 | 53.5 |

- **combined fails on data stream 1 alone**, whatever the init. It is a hard data
  stream for combined + Adam → kept as a fixture (plan 1a.4).
- **titans fails only with init 1 and data 1 together**; either alone trains. It
  starts slower in both split runs (2.5–3k steps vs 0.5–1.5k for most seeds).

## Omega alone (`--window 8`), 10k steps, 3 seeds (P40)

| seed | 0.99 at | rec at 4k | 5–10k rec |
|---|---|---|---|
| 0 | 500 | 50.9 | 54.0 |
| 1 | 750 | 52.6 | 53.8 |
| 2 | 1 000 | 54.2 | 53.5 |

Identity-key titans seeds that trained, same spec: 4k 42.8–50.9, 5–10k 45.6–52.7.
Omega trains on all three seeds (including seed 1, whose data stream is the same)
and is ahead of every identity-key seed at 5–10k.

## Combined + omega (`--family combined --window 8`), 10k steps, 3 seeds (ai0)

| seed | 0.99 at | 2–4k rec | 5–10k rec | plain combined on this seed |
|---|---|---|---|---|
| 0 | 500 | 51.8 | 53.9 | trains (500) |
| 1 | 1 250 | 50.0 | 52.4 | **collapses** |
| 2 | **never** (acc 0.09 at 10k) | 1.4 | 1.0 | trains (1 000) |

- Omega rescues combined on data stream 1, and combined + omega stalls on seed 2,
  which plain combined and titans + omega both train. Seed 2's stall is the
  acc ≈ 0.09 plateau (125–128 distinct predictions), not a constant-predictor
  collapse; titans s3 sat on the same plateau until ~10k.
- Where it trains, combined + omega matches titans + omega (5–10k 52.4–53.9 vs
  53.5–54.0). 1080 peak 74 °C (shared with the user's game), 1660 SUPER 68 °C.
- Every arm so far has a seed that fails to start, and *which* seed moves with
  the configuration: a start-up property of the setup under Adam, not one bad
  data stream.


| arm | seed | 0.99 at | rec at 4k | GPU memory |
|---|---|---|---|---|
| full poly2 (d_feat 2144) | 0 | 1 500 | 58.4 | ~13 GB |
| full poly2 | 1 | 1 000 | 50.2 | ~13 GB |
| poly2 + omega | 0 | never in 40k (0.74 at 40k) | 1.1 | ~13 GB |
| `poly_rank=16` + omega (d_feat 200) | 0 | **never in 4k** (acc 0.009) | 0.45 | 1.06 GB |
| `poly_rank=16` alone | 0 | 1 500 | 60.5 (2–4k mean 59.5, min 56.2) | 1.06 GB |
| `poly_rank=16` alone | 2 | **never in 4k** (acc 0.009, 1 distinct prediction) | 0.5 | 1.06 GB |

2–4k windowed stress recovered: r16 s0 59.5; full poly2 58.1 / 48.2; omega
51.7 / 51.8 / 51.8 (3 of 3 train); identity-key titans at 4k 42.8–50.9.

- `poly_rank=16` fixes the memory (1.06 GB vs 13 GB) and gives the best run so
  far (seed 0), but collapses on seed 2, a data stream that trains every other
  arm: poly2 r16 adds a start-up failure of its own.
- poly2 + omega stalls at both sizes.
- Omega alone is the only switch that trained on every seed, and its seeds agree
  to 0.1 binding. r16 alone was re-run on ai0 (1080 peak 76 °C shared with the
  user's game; 1660 SUPER peak 69 °C) after the laptop aborted at 80 °C.

## Temperatures (guard logs, abort at 80 °C)

| GPU | hours | peak | mean | peak power | fan |
|---|---|---|---|---|---|
| GTX 1080 (ai0) | 10.9 | 70 °C | 64 °C | 106 W | 55 % → up to 115 % reported |
| GTX 1660 SUPER (ai0) | 11.2 | 66 °C | 61 °C | 92 W | 33–34 % |
| P40 (200 W cap) | 1.6 | 70 °C | 60 °C | 147 W | not readable |
| RTX 2060 | 2.1 | 61 °C | 58 °C | 108 W | 0–65 % |
| 1660 Ti laptop | — | **80 °C, abort** | — | — | not readable |

The 1080 is ai0's main GPU and was also running the user's game during these
runs, so its readings are game + one training job, not the job alone (95 % of
samples ≥ 64 °C, fan ramped; nvidia-smi reported up to 115 %, above nominal).
The 1660 SUPER ran only the training job and its fan never left idle speed. Both
stayed ≥ 10 °C under the abort. The laptop hit the 80 °C abort on one poly2 job:
check it for dust.
