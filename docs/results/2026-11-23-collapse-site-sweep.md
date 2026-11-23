# Model-collapse site sweep, plus seeds 1–3 of the core arms

Raw results, no verdict. The method, the "prevented" formula and the choice of
core arms and seeds were all fixed before any results:
[2026-11-22-model-collapse-method.md](2026-11-22-model-collapse-method.md).
Setup as in the [pilot](2026-11-22-collapse-pilot.md), with these changes
(commit `4fe5180`, `--suite sites`):

- **Energy sites:** per-arm `sites` (MLP neurons, attention output, residual)
  at decay = drive/40. In the MLP + attention arms each site has its own cost,
  0.03 and 0.01.
- **noise_0.35:** white noise of σ = 0.35 × the site mean on all three sites,
  in teacher and student. It is the damage-matched control.
- **penalty_1.5:** the teacher writes greedily with a CTRL/HF repetition
  penalty of 1.5.
- **free_\*** columns: the student writes freely. Greedy, energy off,
  16 test prompts × 64 tokens. Test perplexity is teacher-forced and cannot
  see repetition.
- **data_\*** columns: the teacher's continuations that the student trains on,
  written with the arm's energy on.
- **Seeds:**
  - seed 0 runs all 18 arms;
  - seeds 1–3 rerun the 8 core arms;
  - a seed changes the training order, the teacher's sampling, and the energy
    and noise generators;
  - the prompts and test text are the same for every seed.
- **Run:** wall time 7.9 h for seed 0 and 3.15 h per extra seed. P40 peak
  75 °C.

**"Prevented" for the real arms is not usable.** real_control's perplexity
*falls* (rise ≈ −0.10), so the ratio's denominator is near zero and of the
other sign. For the real arms, read their test perplexity against
real_control's 18.7 instead.

### Seed 0, all arms, generation 10

| arm | test ppl g5 | g10 | prevented g10 | free distinct-3 | free looping | free vocab | data distinct-3 | data looping | student changed | s/gen |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| real_control | 19.0 | 18.7 | — | 0.280 | 0.75 | 108 | 0.979 | 0.00 | — | 33 |
| real_mlp_0.05 | 23.6 | 22.3 | 1.77 | 0.491 | 0.44 | 144 | 0.979 | 0.00 | 0.1060 | 116 |
| real_attn_0.02 | 33.0 | 36.9 | 6.85 | 0.462 | 0.56 | 112 | 0.979 | 0.00 | 0.2695 | 115 |
| real_noise_0.35 | 21.7 | 21.0 | 1.16 | 0.371 | 0.62 | 123 | 0.979 | 0.00 | 1.0000 | 33 |
| greedy | 566.2 | 1529.1 | — | 0.229 | 1.00 | 72 | 0.313 | 0.84 | — | 59 |
| greedy_mlp_0.05 | 40.0 | 61.8 | 0.75 | 0.125 | 0.94 | 61 | 0.080 | 1.00 | 0.1006 | 181 |
| greedy_attn_0.02 | 399.8 | 891.2 | 0.13 | 0.030 | 1.00 | 9 | 0.025 | 1.00 | 0.1936 | 178 |
| greedy_resid_0.01 | 454.9 | 1399.4 | 0.02 | 0.017 | 1.00 | 3 | 0.027 | 1.00 | 0.0502 | 176 |
| greedy_all_0.003 | 143.4 | 285.1 | 0.39 | 0.025 | 1.00 | 7 | 0.019 | 1.00 | 0.0686 | 431 |
| greedy_mlp0.03_attn0.01 | 70.9 | 72.5 | 0.71 | 0.084 | 1.00 | 24 | 0.016 | 1.00 | 0.1244 | 304 |
| greedy_noise_0.35 | 622.5 | 2856.2 | -0.15 | 0.070 | 1.00 | 20 | 0.062 | 1.00 | 1.0000 | 64 |
| greedy_penalty_1.5 | 2057.7 | 13470.7 | -0.51 | 1.000 | 0.00 | 354 | 1.000 | 0.00 | — | 60 |
| real_mlp0.03_attn0.01 | 47.5 | 52.9 | 10.47 | 0.412 | 0.69 | 77 | 0.979 | 0.00 | 0.1395 | 205 |
| t0.9 | 51.1 | 123.9 | — | 0.116 | 1.00 | 46 | 0.218 | 0.92 | — | 60 |
| t1.0 | 30.5 | 42.8 | — | 0.168 | 0.94 | 58 | 0.859 | 0.04 | — | 60 |
| t0.9_mlp_0.05 | 30.4 | 38.2 | 0.66 | 0.133 | 1.00 | 58 | 0.093 | 0.99 | 0.0976 | 181 |
| t0.9_attn_0.02 | 169.7 | 410.4 | -0.67 | 0.054 | 1.00 | 8 | 0.087 | 1.00 | 0.1792 | 179 |
| t0.9_mlp0.03_attn0.01 | 111.4 | 369.1 | -0.61 | 0.019 | 1.00 | 4 | 0.444 | 0.56 | 0.1696 | 300 |

### Core arms, seeds 0–3, generation 10: mean [min, max]

| arm | test ppl | prevented | free distinct-3 | free looping | data looping |
|---|--:|--:|--:|--:|--:|
| real_control | 18.7 [18.7, 18.8] | — | 0.329 [0.265, 0.394] | 0.70 [0.56, 0.88] | 0.00 [0.00, 0.00] |
| greedy | 2025.0 [1333.9, 3483.3] | — | 0.330 [0.229, 0.426] | 0.72 [0.50, 1.00] | 0.71 [0.53, 0.86] |
| greedy_mlp0.03_attn0.01 | 78.5 [71.7, 85.6] | 0.70 [0.68, 0.72] | 0.086 [0.043, 0.168] | 0.95 [0.81, 1.00] | 1.00 [1.00, 1.00] |
| greedy_noise_0.35 | 2293.7 [1259.3, 3200.1] | -0.03 [-0.15, 0.02] | 0.090 [0.070, 0.115] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| greedy_penalty_1.5 | 17702.6 [13470.7, 23640.5] | -0.50 [-0.69, -0.31] | 1.000 [1.000, 1.000] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| real_mlp0.03_attn0.01 | 52.5 [50.7, 55.0] | 11.15 [9.64, 12.86] | 0.420 [0.398, 0.443] | 0.70 [0.62, 0.75] | 0.00 [0.00, 0.00] |
| t0.9 | 121.3 [116.9, 124.2] | — | 0.309 [0.116, 0.406] | 0.75 [0.56, 1.00] | 0.55 [0.36, 0.92] |
| t0.9_mlp0.03_attn0.01 | 236.8 [159.5, 369.1] | -0.35 [-0.61, -0.14] | 0.029 [0.019, 0.040] | 1.00 [1.00, 1.00] | 0.89 [0.56, 1.00] |

### Core arms, per seed, generation 10: test ppl / free distinct-3

| arm | seed 0 | seed 1 | seed 2 | seed 3 |
|---|--:|--:|--:|--:|
| real_control | 18.7 / 0.280 | 18.7 / 0.376 | 18.8 / 0.394 | 18.8 / 0.265 |
| greedy | 1529.1 / 0.229 | 1333.9 / 0.426 | 1753.8 / 0.309 | 3483.3 / 0.355 |
| greedy_mlp0.03_attn0.01 | 72.5 / 0.084 | 71.7 / 0.049 | 84.3 / 0.043 | 85.6 / 0.168 |
| greedy_noise_0.35 | 2856.2 / 0.070 | 1259.3 / 0.115 | 1859.0 / 0.075 | 3200.1 / 0.102 |
| greedy_penalty_1.5 | 13470.7 / 1.000 | 23640.5 / 1.000 | 16770.0 / 1.000 | 16929.1 / 1.000 |
| real_mlp0.03_attn0.01 | 52.9 / 0.412 | 50.7 / 0.425 | 51.6 / 0.443 | 55.0 / 0.398 |
| t0.9 | 123.9 / 0.116 | 116.9 / 0.365 | 120.3 / 0.350 | 124.2 / 0.406 |
| t0.9_mlp0.03_attn0.01 | 369.1 / 0.019 | 222.9 / 0.026 | 195.6 / 0.040 | 159.5 / 0.032 |

### Core arms, test perplexity per generation, mean of 4 seeds

| arm | g0 | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 | g10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| real_control | 20.6 | 19.5 | 19.3 | 18.8 | 18.8 | 19.0 | 18.7 | 19.0 | 18.6 | 18.5 | 18.7 |
| greedy | 20.6 | 51.6 | 166.2 | 365.0 | 505.4 | 550.8 | 879.5 | 1209.2 | 1449.9 | 1567.5 | 2025.0 |
| greedy_mlp0.03_attn0.01 | 20.6 | 27.7 | 46.5 | 61.7 | 70.4 | 75.1 | 75.9 | 76.5 | 77.2 | 77.9 | 78.5 |
| greedy_noise_0.35 | 20.6 | 43.6 | 97.1 | 214.4 | 392.6 | 586.4 | 738.3 | 1007.9 | 1299.1 | 1675.3 | 2293.7 |
| greedy_penalty_1.5 | 20.6 | 42.8 | 140.7 | 470.3 | 1058.3 | 2210.3 | 3476.3 | 5306.6 | 8237.9 | 12104.2 | 17702.6 |
| real_mlp0.03_attn0.01 | 20.6 | 27.3 | 37.2 | 44.1 | 48.6 | 50.6 | 51.8 | 53.2 | 52.9 | 53.1 | 52.5 |
| t0.9 | 20.6 | 22.6 | 26.5 | 32.1 | 39.7 | 50.4 | 62.2 | 72.2 | 89.0 | 103.8 | 121.3 |
| t0.9_mlp0.03_attn0.01 | 20.6 | 26.4 | 40.9 | 61.4 | 84.1 | 107.0 | 131.5 | 158.5 | 185.2 | 210.0 | 236.8 |

### Core arms, free distinct-3 per generation, mean of 4 seeds

| arm | g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8 | g9 | g10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| real_control | 0.379 | 0.413 | 0.445 | 0.397 | 0.579 | 0.471 | 0.704 | 0.296 | 0.420 | 0.329 |
| greedy | 0.430 | 0.419 | 0.430 | 0.422 | 0.364 | 0.334 | 0.316 | 0.302 | 0.328 | 0.330 |
| greedy_mlp0.03_attn0.01 | 0.219 | 0.136 | 0.097 | 0.071 | 0.071 | 0.073 | 0.071 | 0.088 | 0.087 | 0.086 |
| greedy_noise_0.35 | 0.244 | 0.205 | 0.166 | 0.144 | 0.132 | 0.121 | 0.103 | 0.111 | 0.103 | 0.090 |
| greedy_penalty_1.5 | 0.912 | 0.953 | 0.995 | 0.998 | 0.992 | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 |
| real_mlp0.03_attn0.01 | 0.488 | 0.387 | 0.393 | 0.435 | 0.395 | 0.385 | 0.403 | 0.357 | 0.396 | 0.420 |
| t0.9 | 0.452 | 0.338 | 0.359 | 0.399 | 0.436 | 0.369 | 0.361 | 0.361 | 0.331 | 0.309 |
| t0.9_mlp0.03_attn0.01 | 0.220 | 0.124 | 0.089 | 0.062 | 0.057 | 0.050 | 0.040 | 0.034 | 0.030 | 0.029 |

### Seeds where the energy arm beats its own baseline, generation 10

| pair | lower test ppl | higher free distinct-3 | lower free looping |
|---|--:|--:|--:|
| real_mlp0.03_attn0.01 vs real_control | 0/4 | 4/4 | 2/4 |
| greedy_mlp0.03_attn0.01 vs greedy | 4/4 | 0/4 | 0/4 |
| greedy_noise_0.35 vs greedy | 2/4 | 0/4 | 0/4 |
| greedy_penalty_1.5 vs greedy | 0/4 | 4/4 | 4/4 |
| t0.9_mlp0.03_attn0.01 vs t0.9 | 0/4 | 0/4 | 0/4 |
| greedy_mlp0.03_attn0.01 vs greedy_noise_0.35 | 4/4 | 2/4 | 1/4 |


## Samples at generation 10 (seed 0, the student's training data)

- **real_control:** wikitext prose (" = Theme and influences = = = \n Clarkson has cited the color red …").
- **greedy:** `= = \n = = \n …` and `= = = = …`.
- **greedy_mlp0.03_attn0.01:** `= = = = …` and `the the the the …`.
- **greedy_noise_0.35:** `= = = = …` and `. \n = = = = …`.
- **greedy_penalty_1.5:** fluent, drifting English with no repeats ("In addition, there are no records indicating whether these works have been successfully published …"). It avoids wikitext's common tokens, which is consistent with its test perplexity of 13k–24k.

## Measured facts that bear on reading the tables

- **greedy + MLP/attention energy:**
  - *Test perplexity stops rising.* It is 75.1 at g5 and 78.5 at g10 (mean of 4 seeds; range 71.7–85.6 at g10). Greedy keeps climbing to 2025 [1334, 3483], and the noise control to 2294 [1259, 3200]. Lower than greedy in 4/4 seeds, and lower than noise in 4/4.
  - *Its free-running output and its training data loop more than greedy's.* Free distinct-3 is 0.086 vs 0.330, and 0/4 seeds are higher than greedy. Data looping is 1.00 in every seed.
  - *Noise lowers free distinct-3 as much (0.090), with no perplexity plateau.*
- **t0.9 + MLP/attention energy is worse than t0.9 on all three counts in 4/4 seeds:** test perplexity 237 vs 121, free distinct-3 0.029 vs 0.309.
- **Repetition penalty 1.5:** free distinct-3 is 1.000 and looping is 0 in 4/4 seeds, and test perplexity is the worst of all arms (17.7k).
- **Energy's cost on clean real text:**
  - real_mlp0.03_attn0.01 ends at 52.5 [50.7, 55.0] vs 18.7.
  - In seed 0: real_mlp_0.05 at 22.3, real_attn_0.02 at 36.9, noise at 21.0.
  - The real MLP + attention arm has the highest free distinct-3 of the real arms (0.420 vs 0.329, 4/4 seeds).
- **Free-running greedy loops even without collapse.** real_control's own free looping is 0.70 [0.56, 0.88]. 16 prompts × 64 greedy tokens gives a noisy, high-floor measure.
- **CPU loop test vs here.** In the CPU loop test, MLP + attention energy broke 3/3 loops in a fixed model. Here, with energy in both teacher and student over 10 generations, the training data it produced was single-token loops in every seed.

Raw data on the sandbox: `~/collapse/out/sites_4fe5180{,_seed1,_seed2,_seed3}/`
(`metrics.csv`, `samples.jsonl`), plus `*.log` and `*.p40.csv`.
