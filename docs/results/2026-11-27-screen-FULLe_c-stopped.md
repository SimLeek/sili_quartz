# Screen stage, FULLe_c: energy on the output neurons made actions random from the start (stopped, 2026-11-27)

FULL_c's arm with energy on at a fixed dose of 1, from the base model.
Commit 0cf2290, `~/runs/t_FULLe_c/`, 08:03–08:15 UTC, stopped at about minute
12. Settings: `--energy-io` (energy on the input neurons *and* the
action-choice neurons), `--tether chunks --tether-out --renorm`,
`--health pr+in`, no MI and no policy term.

| minutes | decisions | distinct | keys per decision | action entropy | log-prob | input-tether error | copy baseline | output-tether error | energy changes per step |
|---|---|---|---|---|---|---|---|---|---|
| 0–5 | 703 | 677 | 8.1 | 0.53 | −1.61 | 1.94 | 0.78 | 1.42 | 0.52 M |
| 5–10 | 563 | 563 | 9.6 | 1.13 | −5.28 | 1.79 | 0.47 | 1.49 | 1.24 M |

For comparison, FULL_c (the same arm, no energy) had a log-prob of −0.47 and
2.7 keys per decision in its first 5 minutes. With energy at dose 1 on the
action-choice neurons, the choice was noisy before training had changed
much, and it got worse as the energy changes per step rose. Typical late
action: `-1000 44 2 ; B ;  ;  ;  ;  ;`.

## Changes made because of this

- `--energy-io-side in` (10a8434): energy on the input neurons only, none on
  the action choice. Output energy waits until a policy term drives the
  output (user: policy terms first; output energy and output averaging after,
  at lower weight).
- Next pair:
  - **Tin_c:** input tether alone.
  - **Tin_e:** input tether with energy on the interior and input neurons,
    no output tether and no output energy.
