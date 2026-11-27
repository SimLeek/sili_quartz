# Screen stage, Tin3_e: input/interior energy at dose 1 made actions random; the trainer's energy barely acts (2026-11-27)

Commit 17c6d50, `~/runs/t_Tin3_e/`, 11:00–13:00 UTC, full 2 hours. Settings:
Tin3_c (input tether on layers 1–12, no MI, no policy term), plus energy at a
fixed dose of 1 on the interior sites (MLP and attention outputs of every
layer) and the input neurons (token embeddings, vision merger), none on the
action choice (`--energy-io-side in`), plus `--renorm`.

## What happened

15,836 decisions, 766 trainer steps. Per 10 minutes:

| minute | decisions | distinct | keys per decision | LB share | median mouse | entropy | log-prob | input-tether error | copy baseline | energy changes per step |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,322 | 1,285 | 8.5 | 0.16 | 190 | 0.74 | −2.34 | 1.70 | 0.59 | 0.59 M |
| 10 | 1,252 | 1,252 | 11.6 | 0.18 | 523 | 1.39 | −5.63 | 0.60 | 0.26 | 0.73 M |
| 30 | 1,325 | 1,325 | 11.4 | 0.15 | 594 | 1.39 | −5.41 | 0.41 | 0.22 | 0.77 M |
| 60 | 1,326 | 1,326 | 11.2 | 0.14 | 795 | 1.42 | −5.36 | 0.50 | 0.60 | 0.73 M |
| 90 | 1,336 | 1,336 | 11.0 | 0.12 | 814 | 1.47 | −5.14 | 0.87 | 0.91 | 0.81 M |
| 110 | 1,339 | 1,339 | 11.1 | 0.13 | 780 | 1.48 | −5.15 | 0.80 | 0.96 | 0.82 M |

Every decision was distinct from minute 10 on, with mouse moves in the
hundreds. Game: 1 death (drowned), 6 blocks mined, 29 m walked, 115 min of
play time. The row-norm band rescaled no rows.

## Tin3_curio (stopped at minute 4)

Tin3_e plus the energy-curiosity reward. Its smoke run and its first steps
logged a drive of exactly 0.0 on every frame, so its reward was 0, and it
would have been Tin3_e again. It was stopped.

## The mismatch this exposes

- **Trainer:** energy starts at rest at every step (`begin_step(at_rest=True)`)
  and runs over one replayed window of about 2,000 tokens. Over that span
  almost no neuron reaches the fire or shut-off thresholds. Intervention
  (fire or shut) was ~0 on the input-side image tokens, and the 0.3–0.8 M
  "changes" per step are mostly dead-zoned near-zero activations.
- **Actor:** its energy state is never reset during a session, so it reaches
  equilibrium. **Correction (checked with the reference kernel the same day):**
  an earlier version of this bullet said any neuron off its site's mean by
  more than ~5 % gets forced or shut. Only the "above" half is right, and it
  is stronger than that. Late-token share intervened, at equilibrium:

  | typical activity / site mean | 0 | 0.5 | 0.9 | 0.96–1.04 | 1.06 | 1.5 |
  |---|---|---|---|---|---|---|
  | dose 1 | 1.4 % | 0.7 % | 0.1 % | 0 | 100 % | 100 % |
  | dose 0.1 | 0.1 % | 0.1 % | 0 | 0 | 100 % | 100 % |

  - **Above the mean:** every neuron whose typical activity is more than
    about 5 % above its site's mean is shut off for good, at any dose. Energy
    settles at decay_div · (1 − |x| / mean) = 40 · (1 − |x| / mean), so the
    band is ±2/40 = 5 %, and a held shut-off never releases while the neuron
    stays above it.
  - **Below the mean:** neurons fire rarely (0.1–1.4 % of tokens), because a
    fire resets energy to 0 and it climbs back slowly.
  - **The dose only sets how fast:** about 1,300 tokens at dose 1, about
    13,000 at dose 0.1. A session is millions of tokens.
  - **Result:** the consistently most active neurons, often the most
    informative, end up silenced. That is the likely cause of the random
    actions in Tin3_e and FULLe_c. The collapse runs never showed it: their
    sequences were short or their state reset, so energy never reached
    equilibrium.
  - **Metric fix:** a held shut-off outputs a constant of 0, so the first
    curiosity drive metric (`const != 0`) missed it. Fixed: an intervention
    is now a fire, or a shut-off of an active neuron.
- **Consequences, for every energy arm so far:**
  - The PPO ratio compares a forward pass with almost no energy (trainer)
    against log-probs sampled under heavy energy (actor).
  - The energy-curiosity drive reads 0.
- **Built** (0811164): the actor saves its energy state every 30 decisions;
  the trainer starts each replayed window from the nearest saved state.
- **For the user to decide:** a per-neuron set point. Compare each neuron
  with its own running typical activity instead of its site's mean. Then a
  neuron is shut off only above its own norm (exhaustion) and fired only below
  it (boredom), and the curiosity drive means "far from its own norm".

## Running meanwhile

- **Eint_c:** one hour of interior-only energy at dose 1, no training signal.
  Does interior energy alone randomize the actor?
- **A2c, A1c:** energy-free controls.
