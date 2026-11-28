# Screen stage, Eint_c: interior energy at dose 1 alone makes the base model's actions random (2026-11-27)

Commit 17c6d50, `~/runs/t_Eint_c/`, 13:17–14:17 UTC, 1 hour. Settings:

- No training signal: `--w-rep 0 --w-pg 0`, no tether. The weights stay the
  base model's.
- Energy at a fixed dose of 1 on the interior sites only (MLP and attention
  outputs of every layer), no input or output neurons.

This is the diagnostic for the Tin3_e and FULLe_c failures.

| minutes | decisions | distinct | keys per decision | LB share | median mouse | log-prob (actor, energy on) |
|---|---|---|---|---|---|---|
| 0–5 | 752 | 699 | 6.0 | 0.15 | 307 | −0.99 |
| 5–10 | 662 | 662 | 11.2 | 0.09 | 1,025 | −1.54 |
| 10–60 | 660–687 per 5 min | all distinct | 10.6–11.4 | 0.08–0.12 | 1,008–1,080 | −1.53 to −1.58 |

Game: 8 deaths in 33.5 min of play time, 16 blocks mined, 389 m walked.
(This actor predates the per-decision intervention count, so that column is
empty.)

## Reading

With the weights untouched, energy alone turned the actions random within 5
minutes of the actor reaching dose 1, and they stayed random. This matches
the equilibrium analysis in the
[Tin3_e doc](2026-11-27-screen-Tin3_e.md). At equilibrium every neuron more
than about 5 % above its site's mean activity is shut off for good, at any
dose. So the randomization comes from energy's site-wide set point, not from
the tether or any training. No energy arm runs until the set point is
decided; the proposal is a per-neuron set point.
