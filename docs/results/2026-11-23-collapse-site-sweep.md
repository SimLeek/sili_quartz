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
