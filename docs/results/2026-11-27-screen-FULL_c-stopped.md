# Screen stage, FULL_c: columns and norm band with no drive; output went random (stopped, 2026-11-27)

Commit e5bdf8c, `~/runs/t_FULL_c/`, 07:39–08:02 UTC, stopped at about minute
25. Arm settings:

- `--tether chunks --tether-out`: input and output depth columns, chunks of 4.
- `--renorm`: row-norm band.
- `--health pr+in`.
- `--energy-io` with the dual dose. The dose stayed at 0 throughout, because
  the PR guard never fired, so energy was effectively off.
- `--w-rep 0 --w-pg 0`: no MI and no policy term.

This is the "columns plus norm band" reference for FULLe_c, which is the same
arm with energy on at a fixed dose of 1.

## What happened

| minutes | decisions | distinct | keys per decision | median mouse | action entropy | log-prob | input-tether error | copy baseline | output-tether error | PR |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–5 | 811 | 710 | 2.7 | 24 | 0.44 | −0.47 | 2.14 | 0.85 | 1.35 | 55.2 |
| 5–10 | 835 | 734 | 2.2 | 16 | 0.45 | −0.44 | 1.16 | 0.34 | 1.05 | 56.2 |
| 10–15 | 730 | 730 | 6.0 | 13 | 0.90 | −0.78 | 0.99 | 0.28 | 1.03 | 52.3 |
| 15–20 | 620 | 620 | 14.0 | 8 | 1.81 | −1.81 | 0.86 | 0.32 | 1.11 | 51.9 |
| 20–25 | 309 | 309 | 4.3 | 7 | 1.90 | −2.69 | 0.77 | 0.16 | 1.32 | 51.1 |

The input-share measure stayed at 0.60–0.61, and the row-norm band rescaled
no rows. Actions from minute 15 look like
`A W W W ; A W W V ; W E W W ; D W D W ; D Q W ; L D W W`: WASD with stray
letters, every decision different. Game: 144 m travelled, 47 jumps, 8 blocks
mined, 6 picked up, 0 deaths. Play time stopped at 9.4 minutes.

## Reading

- **Input tether:** the error kept falling (2.1 → 0.8). It stayed worse than
  predicting that the next frame equals the current one.
- **Output tether:** the error never fell. Its targets are the model's own
  samples, drawn at temperature 1. Pulling the last 4 layers toward them is
  regression onto sampling noise, so the output distribution flattens and the
  next samples are noisier. With no drive and no outside data, this is a
  self-training loop like the model-collapse runs.
- **User's reading:** the tether solves half the problem, grounding, but gives
  no reason to do anything. Mixing randomness into several layers then
  settles the output on a random average behaviour.

## Next

- **FULLe_c:** the same arm with energy at a fixed dose of 1, from the base
  model. Tests whether energy holds the plateau here, as in the collapse runs.
- **Tin_c:** input tether only, no output tether. Separates the output
  tether's loop from "any tether without a drive".
