# 2. Methodology (draft, 2026-11-23)

Built from [part 1](1-introduction.md), mainly 1.11. Everything here is fixed
before any controller result exists. A change after results must say so and
why.

## 2.1 Rules for every experiment

- **No oracles.**
  - Track P may use its training targets.
  - Track A may use only its own screen, its keyboard/mouse stream and its
    internal state.
  - Game-side data (Minecraft statistics, a position-logging mod) is for
    **evaluation only** and is never fed to the agent.
- **Measured two ways.** Every arm reports its gain *and* its cost: wall time,
  GPU memory, and any extra forward passes.
- **Seeds.** Each seed is the same program run again, at least 4 per arm for
  any comparison that is claimed. A lead needs an agreeing sign across seeds
  and a better worst seed.
- **Raw results first.** Each run's tables go to `docs/results/` with no
  verdict. Part 3 collects them, and part 4 draws conclusions.
- **Runs.** Sandbox only, under gpu_guard (80 °C abort), launched detached,
  with a watcher.

## 2.2 Shared parts (build once)

| Part | Choice | Status |
|---|---|---|
| Energy | `sili_quartz/energy.py`, `energy_hooks.py` (MLP and attention sites, decay = drive/40) | built |
| Effect meter | KL(p_energy ‖ p_clean) on next-step outputs, from a clean pass on a sampled subset of steps | to build |
| Surprise meter | running −log p of the model's own output | to build |
| Learning-progress meter | drop in loss over a short and a long window | to build |
| MI estimator | gaussian_bc (primary); InfoNCE (nonlinear arm, d ≤ 64); gaussian on the residuals of a nonlinear forward model (next arm) | benched ([results](../results/2026-11-23-mi-estimator-bench.md)) |
| Health stats | per-layer participation ratio, kept fraction vs PR, branching ratio, dormant fraction, energy shutoff fraction, optimizer second-moment mean and std, weight std | partly built (sili trackers) |
| Health floor | PR ≥ 0.8 × early PR; kept ≥ PR; branching ratio in [0.97, 0.99]; second moment and weight std within the measured shape (ratio to own early values) | from MQAR research, 1.11 |
| Dual update | λ ← max(0, λ + η·(floor − health)) | to build |
| Dose granularity | per region where the model allows it; number of controlled units set by the GPU memory left | to build |

## 2.3 Track P: prediction (accuracy is the extrinsic signal)

**Model:** GameScaling-0.8B, the text path as in 2c.

**Experiment P-1, the loop test** (`loop_vs_temperature`-style; one run per
arm; fast).
- **Arms:** P0a no energy; P0b Mirostat; P0c fixed dose; P1; P1a; P1b; and
  ×R/D on P1.
- **Measures:**
  - loops broken (distinct-3 on 3 looped prompts);
  - on real text, distinct-3 and the original model's perplexity of what was
    written, against the true continuation;
  - the effect KL;
  - time per token.

**Experiment P-2, the collapse loop** (`examples/model_collapse.py`, 10
generations, 4 seeds, greedy teacher).
- **Arms:** P0c (rerun as the matched baseline); P1; P2; P2a; P2+A.
- **Measures:**
  - test perplexity per generation;
  - free-running distinct-3 and looping per generation, *and* during each
    generation's training run, to test the claim that the comfortable phase
    should end within one training run;
  - the training loss floor;
  - the dose trajectory;
  - health stats.
- **Pre-registered reading:**
  - *Better than the fixed dose* means test perplexity at g10 no worse than
    P0c (mean 78.5, worst 85.6), **and** free distinct-3 at g10 above P0c's
    0.086 in at least 3 of 4 seeds.
  - *The phase ends within a run* means free distinct-3 rising during a
    generation's training run while test perplexity holds.

## 2.4 Track A: actor (no extrinsic signal at all)

**Environment:** vanilla Minecraft on the sandbox, played through robonet on
localhost. The AI sees only the screen, through a viewport it zooms and pans
itself, and acts only through keys and the mouse.
- **Containment:**
  - Block OS key combinations (Alt+Tab, Alt+F4, Super) at robonet.
  - Block Esc, or use the no-quit Fabric mod.
  - A watchdog restarts the game into the same world if it exits.
- **Evaluation only:** Minecraft's per-player statistics file (distance,
  blocks mined, items), plus chunks visited if a mod is needed. None of it
  reaches the agent.
- **Setup (user, needs a short internet window):** sign in, install
  Minecraft, the Fabric loader and the mod, and pick the Minecraft version
  the mod supports.

**Architecture:**
- **First:** GameScaling's own (Qwen3.5-0.8B hybrid, vision plus
  keyboard/mouse), used as the trained predictor. The actor takes its
  information from it.
- **Still open:** whether the actor is a separate small head or network, or
  a role region inside the same transformer. The standard approach goes
  first; the Dreamer/Plan2Explore reading decides the exact form.

**Experiment A-0, setup checks** (no training).
- GameScaling's vision path loads and runs on the P40.
- robonet frames reach it; its actions reach the game.
- The untouched model's behavior is recorded as the reference for "does not
  degrade".

**Experiment A-1, reward-free exploration** (long sessions; 4 seeds per arm).
- **Arms:**
  - A1 (predictive information under the health floor; λ is the dose);
  - the ablations A1−act, A1+self and A1−λ;
  - A2 (empowerment);
  - A3 (region-to-region MI);
  - A1+bored.
- **Baselines:** untouched GameScaling; the fixed energy dose; no energy; a
  random policy.
- **Measures (evaluation only):** distance travelled, unique blocks and items
  seen and obtained, chunks visited, time alive, and how long it takes to
  get stuck (no new stats for N minutes).
- **Measures (internal):** the I_pred estimate, health stats, the dose and λ
  trajectories, and wall time per step.
- **The four GameScaling questions:**
  1. Does training degrade the untouched model's play?
  2. Does it improve it?
  3. Can a new init approach it?
  4. Does the actor use the predictor?

**Experiment A-2, later:** sili's learned prefrontal regulator (L) setting
per-region targets, trained on the same objective.

## 2.5 Order

1. Build the shared meters (2.2).
2. P-1, then P-2 (prediction side, fast feedback, existing harnesses).
3. In parallel, the user does the Minecraft setup window; then A-0.
4. A-1.
5. A-2.

## 2.6 Still open

- The exact actor architecture (after the Dreamer/Plan2Explore reading).
- The I_pred feature space: which layer's state, and what dimension (≤ 64
  for InfoNCE).
- Session length and the "stuck" threshold N for A-1.
- The Minecraft version, set by the mod's support.
