# Screen stage, A2g: got wood, then the one-sided entropy guard ran away (stopped, 2026-11-27)

Arm A2 (past = action, empowerment-like) with `--grammar` and `--health pr+act`,
commit 7780d8b, `~/runs/t_A2g/`, 06:45–07:07 UTC (about 22 minutes).
Stopped by the user along with the rest of the queue; not a full run.

## What happened

| minutes | decisions | distinct | most common | median mouse move | top keys |
|---|---|---|---|---|---|
| 0–5 | 795 | 701 | 36 | 27 | W, LB, A, Space, D |
| 5–10 | 856 | 711 | 27 | 21 | LB, Space, RB, W, D |
| 10–15 | 778 | 498 | 184 | 6 | LB (3,589), Space, RB, Three, W |
| 15–20 | 759 | 739 | 5 | 1,397 | eight, LB, seven, six, RB |
| 20–22 | 302 | 297 | 3 | 1,996 | Alt, Down, Right, Ctrl, Left |

All 3,490 decisions parsed (the grammar works). Game stats at minute 20:
mined 35, picked up 17, travelled 65 m, 36 jumps, 0 deaths. The tutorial
moved from "punch tree" to "craft planks" (it got wood) by minute 15. It also
changed settings: language `enp`, chat size and spacing, toggle sprint on.

Trainer (floor on the action entropy 0.354 nats):

| step | minute | entropy | lambda_act | dose | action log-prob |
|---|---|---|---|---|---|
| 102 | 11.1 | 0.43 | 0 | 0 | −0.46 |
| 117 | 13.1 | 0.05 | 0.23 | 0.18 | −0.02 |
| 129 | 14.7 | 0.50 | 0.37 | 0.40 | −0.96 |
| 147 | 17.5 | 0.73 | 0.05 | 0.10 | −1.14 |
| 159 | 19.1 | 1.53 | 0 | 0 | −3.56 |
| 177 | 21.6 | 1.37 | 0 | 0 | −3.61 |

PPO trained on 4–5 matched decisions every step, and gradient norms stayed
finite.

## Reading

While chopping wood the policy settled on holding LB, and the action entropy
fell to 0.05. The floor dual raised its multiplier, and with it the energy
dose, for about 4 minutes. Entropy then rose past the floor and kept rising
to 1.2–1.5 nats after the multiplier had returned to 0. The actions were
random valid strings: number-word keys, Alt/Ctrl/arrows, mouse moves in the
thousands. The guard only pushed up, so nothing brought the entropy back
down. The entropy push and the energy dose rose together, so this run cannot
separate them.

## Changes made because of this

- The action-entropy dual is now two-sided: a ceiling at the early entropy
  ÷ 0.8 with its own multiplier, so the guard works like a thermostat. The
  ceiling adds no energy dose.
- KL to a frozen copy of the untouched policy is not an arm: it is the
  behaviour-cloning target itself, an oracle (user).
