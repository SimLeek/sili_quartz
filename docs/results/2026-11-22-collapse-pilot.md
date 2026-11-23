# Model-collapse pilot: energy on all layers, 10 generations

Raw results, no verdict. Method and the "prevented" formula were fixed before
the run: [2026-11-22-model-collapse-method.md](2026-11-22-model-collapse-method.md).

## Setup

Run as of commit `b30c2b9`, `examples/model_collapse.py`, on the sandbox.

- **Model:** GameScaling-0.8B (Qwen3.5-0.8B), work copy.
- **Hardware:** teacher on the RTX 2060 in fp16; student on the P40 in fp32.
- **Each generation:**
  - 128 fresh 64-token wikitext-2 train blocks.
  - The teacher continues each block by 64 tokens.
  - The student trains one epoch on the continuations (loss on the continuation only).
  - Training settings: AdamW lr 1e-5, batch 8, backpropped as two halves of 4.
  - The teacher then takes the student's weights.
- **Energy:**
  - On every one of the 24 layers' MLP neurons, in both teacher and student.
  - decay = drive/4. This setting lets energy settle against a loop instead of building up; see [energy-vs-loops](2026-11-22-energy-vs-loops.md).
  - State accumulates across the whole arm.
- **Test perplexity:** on 128 fixed wikitext-2 test blocks.
- **Diversity columns** (distinct-3, looping = fraction of samples with distinct-3 < 0.5, rare-token mass, vocab used):
  - Measured on that generation's **training data**, i.e. the teacher's continuations.
  - Measured with energy on.
  - The real arms train on real text, so their columns are the real data's.
- **One seed per arm.** The P40 peaked at 16.1 GiB without energy and 16.6 GiB with it, and stayed at 73 °C or below. Wall time was 4.1 h.

`prevented = 1 − rise(energy) / rise(same sampling, energy off)`, with
`rise = ln ppl(g) − ln ppl(0)`. The real arms use real_control as the reference.

## Summary at generations 5 and 10

| arm | ppl g1 | g5 | g10 | prevented g5 | prevented g10 | distinct-3 g10 | looping g10 | rare mass g10 | vocab g10 | student changed g10 | s/gen |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| real_control | 19.4 | 19.0 | 18.7 | — | — | 0.979 | 0.000 | 0.092 | 2200 | — | 28 |
| real_energy_0.1 | 22.5 | 21.3 | 20.4 | 1.37 | 0.87 | 0.979 | 0.000 | 0.092 | 2200 | 0.0191 | 111 |
| greedy | 51.8 | 566.2 | 1529.1 | — | — | 0.313 | 0.836 | 0.018 | 481 | — | 56 |
| greedy_energy_0.001 | 35.8 | 203.5 | 506.9 | 0.31 | 0.26 | 0.193 | 0.969 | 0.021 | 384 | 0.0122 | 176 |
| greedy_energy_0.01 | 28.9 | 97.2 | 226.0 | 0.53 | 0.44 | 0.192 | 0.992 | 0.015 | 290 | 0.0174 | 176 |
| greedy_energy_0.1 | 24.6 | 65.9 | 126.1 | 0.65 | 0.58 | 0.197 | 0.992 | 0.009 | 211 | 0.0202 | 177 |
| t0.9 | 23.1 | 51.1 | 123.9 | — | — | 0.218 | 0.922 | 0.010 | 402 | — | 53 |
| t0.9_energy_0.001 | 23.0 | 37.0 | 63.6 | 0.36 | 0.37 | 0.504 | 0.492 | 0.032 | 706 | 0.0104 | 174 |
| t0.9_energy_0.01 | 23.6 | 36.2 | 52.3 | 0.38 | 0.48 | 0.243 | 0.953 | 0.029 | 404 | 0.0158 | 176 |
| t0.9_energy_0.1 | 23.2 | 33.8 | 50.9 | 0.46 | 0.50 | 0.330 | 0.758 | 0.010 | 238 | 0.0194 | 173 |
| t1.0 | 22.0 | 30.5 | 42.8 | — | — | 0.859 | 0.039 | 0.124 | 1977 | — | 53 |

How to read these columns:

- **real_energy_0.1:** its rise is negative, so "prevented" > 1 or near 1 only means its perplexity fell less than the real control's did. It started 1.8 above the control at g1 and is 1.7 above at g10.
- **student changed:** the fraction of the student's MLP outputs that energy altered, measured on that generation's training pass.
- **t1.0:** has no energy arm in this pilot.

## Test perplexity per generation

| arm | g0 | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 | g10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| real_control | 20.7 | 19.4 | 19.2 | 18.9 | 18.8 | 19.0 | 18.7 | 19.1 | 18.7 | 18.4 | 18.7 |
| real_energy_0.1 | 20.7 | 22.5 | 22.7 | 21.9 | 21.1 | 21.3 | 21.0 | 21.5 | 20.8 | 20.0 | 20.4 |
| greedy | 20.7 | 51.8 | 175.7 | 426.9 | 631.0 | 566.2 | 713.3 | 910.2 | 1066.6 | 1254.2 | 1529.1 |
| greedy_energy_0.001 | 20.7 | 35.8 | 72.4 | 118.9 | 208.9 | 203.5 | 246.0 | 285.7 | 393.9 | 454.2 | 506.9 |
| greedy_energy_0.01 | 20.7 | 28.9 | 40.1 | 61.8 | 80.0 | 97.2 | 115.5 | 135.1 | 143.3 | 175.2 | 226.0 |
| greedy_energy_0.1 | 20.7 | 24.6 | 33.7 | 45.0 | 55.6 | 65.9 | 73.8 | 85.0 | 107.5 | 119.1 | 126.1 |
| t0.9 | 20.7 | 23.1 | 27.2 | 33.0 | 40.0 | 51.1 | 63.2 | 74.4 | 90.3 | 109.0 | 123.9 |
| t0.9_energy_0.001 | 20.7 | 23.0 | 25.9 | 28.8 | 32.0 | 37.0 | 42.4 | 47.3 | 53.3 | 59.5 | 63.6 |
| t0.9_energy_0.01 | 20.7 | 23.6 | 26.5 | 29.4 | 32.8 | 36.2 | 41.5 | 44.7 | 48.1 | 49.5 | 52.3 |
| t0.9_energy_0.1 | 20.7 | 23.2 | 26.2 | 28.6 | 31.4 | 33.8 | 36.7 | 39.2 | 42.1 | 45.7 | 50.9 |
| t1.0 | 20.7 | 22.0 | 24.1 | 26.0 | 28.4 | 30.5 | 32.7 | 34.2 | 37.5 | 40.2 | 42.8 |

## Looping fraction of the training data per generation

| arm | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 | g10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| greedy | 0.70 | 0.82 | 0.84 | 0.78 | 0.83 | 0.77 | 0.86 | 0.88 | 0.83 | 0.84 |
| greedy_energy_0.001 | 0.71 | 0.83 | 0.97 | 0.97 | 0.92 | 0.91 | 0.95 | 0.95 | 0.97 | 0.97 |
| greedy_energy_0.01 | 0.73 | 0.88 | 0.84 | 0.92 | 0.93 | 0.98 | 0.97 | 0.97 | 0.97 | 0.99 |
| greedy_energy_0.1 | 0.72 | 0.72 | 0.88 | 0.92 | 0.95 | 0.95 | 0.98 | 0.99 | 0.99 | 0.99 |
| t0.9 | 0.08 | 0.20 | 0.38 | 0.50 | 0.58 | 0.71 | 0.81 | 0.85 | 0.91 | 0.92 |
| t0.9_energy_0.001 | 0.02 | 0.05 | 0.21 | 0.33 | 0.42 | 0.48 | 0.48 | 0.47 | 0.66 | 0.49 |
| t0.9_energy_0.01 | 0.38 | 0.53 | 0.55 | 0.55 | 0.73 | 0.70 | 0.85 | 0.87 | 0.85 | 0.95 |
| t0.9_energy_0.1 | 0.46 | 0.34 | 0.43 | 0.44 | 0.55 | 0.67 | 0.67 | 0.72 | 0.75 | 0.76 |
| t1.0 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.05 | 0.03 | 0.02 | 0.05 | 0.04 |

The real arms loop 0.00 in every generation.

## Measured facts that bear on reading the tables

- **Test perplexity and looping move in opposite directions across the energy arms.**
  - In the greedy family, more energy gives lower test perplexity and *more* looping in the training data (0.84 → 0.97–0.99 at g10).
  - [energy-vs-loops](2026-11-22-energy-vs-loops.md) measured that the original model scores looped text about 10× better. So test perplexity alone cannot tell "less forgetting" from "learned less from its own data".
- **The perplexity columns include energy's cost to learning.** real_energy_0.1 trains on clean real text and still sits 1.7–3.3 above the real control at every generation.
- **This pilot uses decay = drive/4 and MLP sites only.**
  - Both were measured after the launch, and the run was left as configured.
  - decay = drive/4 lets energy settle against a loop (93.5% of neurons sit in the dead band).
  - MLP + attention energy at decay = drive/40 is the configuration that broke 3/3 test loops into readable text.
  - The site sweep (`sites_4fe5180`, running) covers those configurations and adds free-running student metrics.

Raw data: `~/collapse/out/pilot_b30c2b9/` on the sandbox (`metrics.csv`, `samples.jsonl`, `run.log`).
