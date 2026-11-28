# Screen stage, A1pa and A1io: format breakage the loss could not see (2026-11-27)

Third and fourth trained sessions of the screen stage, same stack and
conditions as [A1](2026-11-26-screen-A1.md) and
[A1-act](2026-11-27-screen-A1-act.md). Both ran on commit aab87c1 with the
action-entropy dual (`--health pr+act`). Raw numbers first.

- **A1pa**: arm A1 (past = state and action), `~/runs/t_A1pa/`, 01:55–03:55 UTC.
- **A1io**: arm A1 with energy on the input and output neurons as well
  (`--energy-io`), `~/runs/t_A1io/`, 03:58–05:58 UTC.

**Both runs are contaminated by a trainer bug** (found during A1io, fixed
in f567653 and 7780d8b, see below). Read them as a record of what the bug
did, not as MI or energy results. Both arms are queued again with
`--grammar` (A1g, A1io_g).

## What happened

| | A1pa | A1io |
|---|---|---|
| Decisions | 10,999 | 11,700 |
| Parsed | 3,249 | 3,509 |
| Unparsed | 7,750 (7,743 distinct raw completions) | 8,191 (8,106 distinct) |
| Unparsed at the 64-token cap | 6,387 | 6,144 |
| First unparsed decision | minute 7.7 | minute 12.9 |
| Seconds per decision, median (min 0–10 / 10–30 / 30–120) | 0.33 / 0.43 / 0.80 | 0.34 / 0.38 / 0.79 |
| Trainer | 831 steps, 8.6 s per step, 7.8 GiB | 793 steps, 9.1 s per step, 7.8 GiB |

Unparsed decisions per 10 minutes, and the total in each window:

| minute | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1pa | 141 / 1,615 | 279 / 1,387 | 528 / 825 | 678 / 786 | 745 / 788 | 773 / 808 | 714 / 798 | 736 / 786 | 772 / 803 | 777 / 790 | 804 / 810 | 803 / 803 |
| A1io | 0 / 1,691 | 674 / 1,474 | 832 / 1,309 | 683 / 950 | 685 / 940 | 764 / 767 | 755 / 755 | 759 / 759 | 757 / 757 | 740 / 756 | 774 / 774 | 768 / 768 |

Examples of the raw unparsed completions:

```
>53 -269 -64 0 ; LB RB W ;   ;   ;   ;   ;  <|action_end|>        (4 numbers)
>1 -522 ;JK ;DZ ;K LB ;K ;LJ ;K ;JH ;J ;K ;J ;J ;J ;J ;J ;J ;J ...  (non-keys, too many groups)
>0;5470;656945;876262|six|five<|action>0 -215|85752539|8|868665<|action_end|>
>78E7J9J9J9M8A8F4J8Z8N4J8D8I8J8J8IJ8J8J8J8M8J8J8I8K8J8M8J8K8 8J8<|action_end|>
>89-99-898-898-899-899-899-899-899-899-899-899-899-899-899-884-8<|action_end|>
```

Robonet executes nothing for an unparsed completion. The long completions
are why the decision interval more than doubled.

Trainer, every 40 steps (selected). `ent` is the action-entropy the dual
watched; `lam_act` its multiplier, which sets the dose with the PR dual.

| run | step | minute | MI (nats) | logp | ent | lam_act | PR | dose |
|---|---|---|---|---|---|---|---|---|
| A1pa | 41 | 1 | 2.39 | −0.50 | 0.49 | 0 | 59.1 | 0.002 |
| A1pa | 81 | 7 | 4.33 | −0.05 | 0.14 | 0.117 | 56.8 | 0.089 |
| A1pa | 161 | 18 | 5.39 | −0.34 | 0.25 | 0.020 | 55.8 | 0.007 |
| A1pa | 321 | 40 | 5.68 | −0.45 | 0.69 | 0 | 50.9 | 0 |
| A1pa | 601 | 82 | 5.51 | −1.36 | 0.67 | 0 | 44.8 | 0 |
| A1pa | 831 | 120 | 5.56 | −0.71 | 0.69 | 0 | 44.9 | 0 |
| A1io | 41 | 1 | 2.61 | −0.35 | 0.28 | 0.008 | 45.5 | 0 |
| A1io | 121 | 12 | 4.82 | −0.26 | 0.34 | 0.130 | 64.6 | 0.130 |
| A1io | 201 | 23 | 5.49 | −1.95 | 0.54 | 0 | 54.9 | 0 |
| A1io | 281 | 34 | 5.33 | −0.36 | 0.32 | 0.019 | 50.5 | 0.016 |
| A1io | 324 | 40 | 5.48 | −1.37 | 0.57 | 0 | 49.8 | 0 |

PPO turns per step (decisions whose replay matched the sampled tokens) and
the largest dose, per 20 minutes of trained steps:

| minute | A1pa ppo_turns mean (steps with 0) | A1pa max dose | A1io ppo_turns mean (steps with 0) | A1io max dose |
|---|---|---|---|---|
| 0 | 4.38 (14 of 138) | 0.154 | 3.88 (27 of 140) | 0.141 |
| 20 | 1.01 (105 of 144) | 0.100 | 1.40 (84 of 144) | 0.058 |
| 40 | 0.06 (135 of 143) | 0 | 0.93 (99 of 135) | 0.017 |
| 60 | 0.05 (122 of 128) | 0 | 0.00 (114 of 114) | 0 |
| 80 | 0.01 (116 of 117) | 0 | 0.06 (113 of 116) | 0.010 |
| 100 | 0.02 (119 of 122) | 0.003 | 0.00 (105 of 105) | 0.014 |

From minute 40 on, nearly every trainer step updated the policy on nothing.

## The bug

The actor logs `text = NOOP_TEXT` for an unparsed completion (the real
string is in `raw`). The trainer rebuilds each past turn from `text`, so an
unparsed decision is replayed as the no-op, which never matches the tokens
that were actually sampled. PPO skips a turn whose replay does not match,
and the action-entropy dual measured the entropy over the substituted
no-op. So breaking the format had no gradient against it, and once a
policy broke the format it stayed broken: the steps with 0 PPO turns above.
The entropy the dual saw sat near 0.7 nats, well above its floor (~0.33),
the whole time, so the dual stayed off.

The fix, `--grammar`:

- **f567653**: decoding is masked to the action format (3 numbers, then up
  to 6 groups of known keys), so every completion parses. The trainer
  applies the same mask, so PPO's ratio and the entropy are over the
  distribution the actor sampled from.
- **7780d8b**: the first smoke run of f567653 hit `grad_norm: NaN` on its
  first update. The masked entropy computed `exp(lp) · lp` with `lp = −inf`
  on masked tokens, which is NaN in backward even under `torch.where`.
  Masked logits are now filled with −1e4 (same distribution in fp32), and a
  test checks the gradient is finite and the distribution matches the
  −inf mask.

## The screen and the game

| | untouched | random | A1 | A1-act | **A1pa** | **A1io** |
|---|---|---|---|---|---|---|
| Mined | 122 | 141 | 22 | 22 | 7 (jungle_leaves 6, oak_leaves 1) | 11 (jungle_leaves 3, sand 2, grass 2, fern 2, dirt 1, oak_leaves 1) |
| Picked up | 48 | 14 | 2 | 1 | 0 | 4 (sand 2, wheat_seeds 1, dirt 1) |
| Deaths | 2 | 5 | 0 | 0 | 1 | 2 (the second by a skeleton) |
| Travelled | 741 m | 1,145 m | 199 m | 60 m | 161 m, all by minute 15 | 202 m, all by minute 30 |
| Jumps | 330 | 370 | — | 22 | 60 | 54 |
| Play time | 105 min | 80 min | 115 min | 9.9 min | 11.4 min | 110 min (ticks run on the death screen) |

- **A1pa**: play ticks stopped at 11.4 minutes, before minute 15. From
  minute 25 the screen shows Minecraft's "open this link?" dialog for
  `https://aka.ms/snapshotbugs?ref=game` (the Game Menu's bug-report
  button), with "Saved screenshot" lines (F2) behind it. Nothing left the
  sandbox: the browser block below was added during this run, and its kill
  log recorded no Firefox process.
- **A1io**: died for the second time between minutes 15 and 30 and stayed
  on the "You Died!" screen to the end (frames at minutes 60 and 115 show
  it, "shot by Skeleton"). No respawn was ever clicked.

## Reasoning probes (the weights being trained, every 5 minutes)

| | thoughts | distinct3 | reuse4 | per 30-minute window (thoughts, distinct3) |
|---|---|---|---|---|
| A1pa | 33 | 0.912 | 0.027 | (11, 0.93), (10, 0.95), (6, 0.98), (6, 0.97) |
| A1io | 44 | 0.765 | 0.129 | (13, 0.87), (13, 0.84), (11, 0.92), (7, 0.90) |

A1pa's top phrases: "the player has" 10, "the player is" 9, "i need to" 6.
A1io's thoughts were about the screen it was on: "the death screen" 22,
"i need to" 21, "the respawn button" 18, "click the respawn" 11, "respawn
button with" 10. The thoughts named the right action, and the actions that
followed were broken strings that robonet could not execute. This is the
thought–action split (only the action tokens are trained), visible
directly.

## Incidents

- **Browser block (during A1pa).** Firefox was the sandbox's default
  http/https handler, so "Open in Browser" on that dialog would have sent a
  request to Microsoft. The handler now points at a no-op, with a kill loop
  as backstop, both behind `~/runs/sandbox_mode.sh session|desktop|status`
  (details in the [A1-act doc](2026-11-27-screen-A1-act.md)).
- **Duplicate session (A1io start).** When I swapped arms queues, A1pa had
  already ended and the old queue had already started A1io, so the new
  queue launched a second A1io. I killed the duplicate session script and
  its launcher before it reset the world. The original game, actor and
  trainer kept running. The game watcher (`minecraft.py watch`) was killed
  along with it: it was down for about 50 s from 04:06:02 UTC and was
  restarted. `t_A1io/reset.json` was left empty. A1io's game stats start
  from 0 (table above), so the world reset itself had happened.

## Not measured yet

- Thought–action agreement as a number (A1io's respawn thoughts are the
  clearest example so far).
- Whether MI on these arms collapses onto predictable streams once broken
  formats are impossible: the `--grammar` re-runs (A1g, A1io_g) answer it.
