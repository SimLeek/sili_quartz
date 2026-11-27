# Screen stage, A1c: A1 without guards collapses as A1 did (stopped, 2026-11-27)

Arm A1 (past = state and action), `--grammar`, `--health pr`, no entropy
guard, no energy (the PR dual dose stayed near 0), no tether. Commit 17c6d50,
`~/runs/t_A1c/`, stopped at about minute 25 by the user's call.

| minute | decisions | distinct | keys used | median mouse | entropy | log-prob | MI (nats) | most common action count | top keys |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,672 | 1,304 | 23 | 7 | 0.38 | −0.40 | 4.30 | 56 | LB, W, E |
| 10 | 1,712 | 1,076 | 18 | 2 | 0.28 | −0.33 | 5.17 | 145 | W, LB, D |
| 20 (5 min) | 735 | 229 | 15 | 2 | 0.16 | −0.24 | 5.38 | 280 | D, W, S |

Game at about minute 25: 228 m travelled, 83 jumps, 33 blocks mined, 1
death.

Prediction (objective-math doc): a static screen within the run. It had not
reached one by minute 25. It was narrowing the same way A1 did (A1 reached a
two-action loop by minute 50), with MI rising as the actions narrowed. Stopped
to make room for the Clip-Cov / KL-Cov arms and the offline repertoire probe.
