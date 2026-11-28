# Screen stage, Tin_c: tethering all six chunks made the policy a by-product of input prediction (stopped, 2026-11-27)

Commit 10a8434, `~/runs/t_Tin_c/`, 08:16–08:59 UTC, stopped at about minute
42. Settings:

- `--tether chunks`: all 6 chunks tethered, chunks of 4 layers, including the
  last chunk before the output head.
- No output tether.
- `--health pr`, dual dose (it stayed at 0).
- No MI, no policy term.

## What happened

| minutes | decisions | distinct | keys per decision | LB share | median mouse | log-prob (actor) | entropy | input-tether error | copy baseline |
|---|---|---|---|---|---|---|---|---|---|
| 0–5 | 810 | 739 | 3.2 | 0.16 | 36 | −0.52 | 0.43 | 2.08 | 0.86 |
| 10–15 | 840 | 708 | 2.2 | 0.07 | 22 | −0.48 | 0.42 | 0.93 | 0.24 |
| 20–25 | 883 | 760 | 1.3 | 0.11 | 31 | −0.50 | 0.45 | 0.94 | 0.20 |
| 30–35 | 904 | 513 | 2.0 | 0.02 | 2 | −0.31 | 0.33 | 0.65 | 0.14 |
| 35–40 | 290 | 192 | 2.4 | 0.01 | 4 | −0.36 | 0.26 | 0.46 | 0.12 |

Game: 398 m travelled, 82 jumps, 30 blocks mined, 2 deaths. From the
second death (about minute 30) it stayed on the "You Died!" screen. It made
tiny mouse moves with almost no clicks, and grew *more* certain of its actions:
log-prob −0.31 against −0.52 at the start.

## Drift from the base model (`tools/policy_drift.py`, checkpoint step 213, about minute 30)

The same 10 frames for each model, 6 sampled actions per frame:

| frames | LB share base → Tin_c | median mouse base → Tin_c | top keys base | top keys Tin_c |
|---|---|---|---|---|
| death screen, minutes 31–37 | 0.08 → 0.00 | 1 → 1 | Space, LB, A, Enter | Space, A, W, Esc |
| open world, minutes 5–15 | 0.23 → 0.08 | 45 → 34 | LB, Space, D, W | Space, W, D, LB |

Cosine between base and checkpoint hidden states at the decision token, per
layer:

| | embeddings | layers 1–8 | layers 9–12 | layers 13–16 | layers 17–20 | layers 21–24 |
|---|---|---|---|---|---|---|
| open world | 1.00 | 1.00 → 0.97 | 0.98 → 0.93 | 0.88 → 0.81 | 0.84 → 0.88 | 0.92 → 0.98 |
| death screen | 1.00 | 1.00 → 0.98 | 0.98 → 0.94 | 0.91 → 0.87 | 0.89 → 0.91 | 0.94 → 0.98 |

The drift peaks in the middle (layers 13–16), not right before the head.

## Empowerment check on the same checkpoint

Mean nats per set:

| | open world A2g | open world A2b | pause menu | death screen | link dialog |
|---|---|---|---|---|---|
| base | 161.3 | 164.7 | 162.7 | 164.5 | 171.0 |
| Tin_c, minute 30 | 158.7 | 161.5 | 164.0 | 159.2 | 167.7 |

Still no separation between static screens and the open world. On A2g's
frames, chunk 1's next-frame error is 1.90 against the copy baseline's 0.16,
even though Tin_c's own training error fell to about 0.34. The tether fit
Tin_c's own recent world rather than next frames in general.

## Reading

- **User's reading:** with every chunk tethered to predict the input, the
  features the action head reads are input-prediction features. The policy
  becomes a by-product of predicting the next input, confident because the
  prediction is good.
- **Why it reaches a dark room:** actions change the next input, so features
  trained to predict it fit best when actions do not change it.
- **Observed:** the trained model lost its clicks (none on the death screen),
  and its states moved most in the middle layers.

## Changes made because of this

- `--tether-chunks N` (86327aa) tethers only the first N chunks from the input
  side.
- Next arms tether layers 1–12 (`--tether-chunks 3`):
  - **Tin3_c:** no energy.
  - **Tin3_e:** with input and interior energy.
  - **Tin3_curio:** the energy-curiosity reward.
