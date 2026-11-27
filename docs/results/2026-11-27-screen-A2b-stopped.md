# Screen stage, A2b: the two-sided entropy band oscillated; entropy guard removed (stopped, 2026-11-27)

Arm A2 (past = action, empowerment-like) with `--grammar` and the two-sided
action-entropy band (`--health pr+act`), commit 9d180d7, `~/runs/t_A2b/`,
06:51–07:28 UTC. Stopped by the user after about 23 minutes of training,
together with the queue. Not a full run.

## What happened

Band on the action entropy: 0.33–0.51 nats. Per 2 minutes (entropy, the two
multipliers and the dose are trainer means):

| minute | decisions | distinct | interval | median mouse | entropy | λ floor | λ ceiling | dose | log-prob | top keys |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 323 | 284 | 0.35 s | 37 | 0.39 | 0.005 | 0 | 0.005 | −0.43 | W, Space, LB, D |
| 6 | 275 | 221 | 0.41 s | 36 | 0.28 | 0.113 | 0 | 0.105 | −0.27 | LB, Space, RB, W |
| 10 | 275 | 268 | 0.45 s | 11 | 0.47 | 0.002 | 0.010 | 0.003 | −0.79 | RB, LB, Six, Eleven |
| 12 | 270 | 248 | 0.46 s | 4 | 0.49 | 0.001 | 0.042 | 0.001 | −1.00 | RB, LB, Eight, Eleven |
| 16 | 354 | 301 | 0.35 s | 12 | 0.27 | 0.068 | 0 | 0.056 | −0.58 | LB, RB, Six, Eight |
| 20 | 325 | 270 | 0.37 s | 11 | 0.50 | 0.009 | 0.049 | 0.004 | −1.10 | LB, RB, Eleven, Minus |

Entropy cycled between the two bounds about every 10 minutes: focused
stretches below the floor, then key-mashing near the ceiling with longer
completions (decision interval 0.35 → 0.47 s). Every decision parsed.

Mean surprisal of the sampled tokens by kind (nats per token; tokens per
decision in brackets), from the actor's logged log-probs:

| minute | all | numbers (mouse, scroll) | keys | `;` | spaces |
|---|---|---|---|---|---|
| 0 | 0.48 (21) | 1.37 (5.7) | 0.40 (3.6) | 0.02 | 0.19 |
| 10 | 0.69 (27) | 1.31 (5.0) | 0.74 (14.6) | 0.15 | 0.34 |
| 14 | 0.18 (32) | 0.64 (4.7) | 0.12 (18.3) | 0.06 | 0.05 |
| 18 | 0.68 (28) | 1.70 (6.8) | 0.45 (13.3) | 0.18 | 0.28 |
| 20 | 0.28 (24) | 1.23 (4.9) | 0.01 (10.3) | 0.02 | 0.20 |

The mouse digits carried 50–90 % of each decision's surprisal. The guard
averaged over all action tokens, so holding a key on every 33 ms step (up to
18 near-certain key tokens per decision) pulled the average under the floor,
and the floor then pushed randomness into everything, mostly the mouse.

Game at 8 minutes of play time: 50 m travelled, 15 blocks mined (oak leaves 7,
grass blocks 5, dirt 3), 1 death.

**Incident:** at about minute 15 the policy opened the Game Menu's bug-report
link dialog again (as in A1pa). The screen did not change for 185 s, so the
game watcher killed and relaunched the game at 07:10:26 UTC (about a minute
lost). The desktop and the launcher were on screen during the relaunch. No
browser opened (no-op handler, guard log empty since 02:37).

## Changes made because of this

The action-entropy guard is removed from every arm (user). It was a direct
loss on the model's output entropy: below the floor its gradient made the
output distribution flatter, above the ceiling sharper. The right entropy
depends on the moment (low while chopping wood, high for quick actions), so
any fixed band is wrong. Entropy is still logged as a measurement only. In its
place: depth-averaged columns on the input side (`--tether chunks`) and the
output side (`--tether-out`), with energy keeping neurons in a useful band.
