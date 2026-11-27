# Screen stage, arm A1-act (past = state only): collapsed onto the pause menu (2026-11-27)

Second trained session of the screen stage, same stack and conditions as
[A1](2026-11-26-screen-A1.md). Run `~/runs/t_A1-act/` on the sandbox,
23:52–01:52 UTC, commit 17d92be, PR-only health (registered before the
`pr+act` change). Raw numbers.

Arm A1-act: like A1, but the past is the state alone
(I(s_t; s_t+1...) with no action in the past), PPO term rewarded by the
pointwise MI, dual on the PR of layers[12] setting the energy dose.

## What happened

| | |
|---|---|
| Decisions | 20,898; 19,902 parsed (the 996 unparsed are all one string, from minute 106) |
| Decision interval | median 0.32 s over the run; about 0.76 s in the last 15 minutes |
| Trainer | 764 steps, 7.75 s per step, 7.33 GiB, P40 at most 71 °C (2060 at most 68 °C) |
| Dose | **effectively 0**: PR floor 40.3, PR 39–60; dose at most 0.0077 units (step 705) |
| Game time | **9.9 min** of play ticks; the clock stopped at minute 30–35 and never restarted |

Distinct actions and the most common action per 10 minutes:

| minute | distinct | most common (count) |
|---|---|---|
| 0 | 1,421 | `0 0 0 ; (no keys)` (97) |
| 10 | 1,080 | `0 0 0 ; (no keys)` (73), `1 1 0 ; (no keys)` (72) |
| 20 | 277 | `1 16 0 ; (no keys)` (516) |
| 30–90 | **1** (2 at minute 90) | `1 16 0 ; (no keys)` (about 1,870 of 1,870) |
| 100 | 546 | `1 16 0` (342), then the unparsed `0 0 0 ; ; ; ; ; ;` (227) |
| 110 | **1** | the unparsed `0 0 0 ; ; ; ; ; ;` (769) |

`1 16 0` is a small mouse move with no keys or clicks. The unparsed string
differs from the no-key action only in single instead of double spaces
between the `;` separators; robonet's parser rejects it, so it does nothing.

**The screen** (1 fps recording): by minute 25 the game was in Spanish (the
Mods list, "Configuración"), and from minute 35 to the end every sampled
frame (35, 60, 105, 115 min) is the pause menu ("Menú", "Volver al juego")
over the night-time world. The game was paused for the last ~85 minutes,
which is why play ticks stopped at 11,902 (9.9 min). The policy held the
pause menu with a mouse move that touches no button.

Trainer, every 40 steps (selected):

| step | minute | MI (nats) | action log-prob | PR |
|---|---|---|---|---|
| 41 | 3 | 2.76 | −0.59 | 59.2 |
| 161 | 23 | 4.11 | −0.13 | 59.5 |
| 281 | 39 | 4.24 | −0.19 | 52.9 |
| 401 | 62 | 4.43 | −0.08 | 45.7 |
| 601 | 95 | 4.50 | −0.12 | 47.2 |
| 681 | 106 | 4.23 | −0.77 | 40.2 |
| 721 | 114 | 4.28 | −1.16 | 43.6 |
| 764 | 120 | 4.29 | −1.15 | 44.7 |

The switch to the unparsed string at minute 106 comes with PR dropping to
its lowest (39.0 at step 552, 40.2 at 681) and the action log-prob falling
from about −0.1 to −1.1. The dose stayed under 0.01 units throughout.

Game stats against the Esc-free A-0 runs and A1:

| | untouched | random | A1 | **A1-act** |
|---|---|---|---|---|
| Mined | 122 | 141 | 22 | 22 (oak_leaves 9, grass 6, dirt 3, fern 3, grass_block 1), all by minute 25 |
| Picked up | 48 | 14 | 2 | 1 (dirt) |
| Deaths | 2 | 5 | 0 | 0 |
| Travelled | 741 m | 1,145 m | 199 m | 60 m (walk 47, climb 13), all by minute 20 |
| Jumps | 330 | 370 | — | 22 |
| Game time | 105 min | 80 min | 115 min | 9.9 min (paused from minute ~35) |
| Key share per step: LB / W / Esc | 0.145 / 0.073 / 0.032 | 0.144 / 0.072 / 0.032 | — | 0.027 / 0.009 / 0.004 |

## Reasoning probes (the weights being trained, every 5 minutes)

34 thoughts. distinct3 0.852, reuse4 0.060 over the run; per 30-minute
window 4, 12, 11, 7 thoughts with distinct3 0.98, 0.88, 0.92, 0.98. No
sentence of four or more words repeats. Top phrases: "the player has" 11,
"so i should" 8, "i need to" 8, "button with lb" 6, "with the mouse" 6,
"destroy the tree" 5. As in A1, the thoughts stay fluent base-model text.

## Next session (A1pa, running at time of writing)

A1 with the action-entropy dual (`--health pr+act`), started 01:55 UTC. At
minute 38: entropy 0.73 nats against a floor of about 0.35, lambda_act back
at 0 after firing at minutes 7 and 18; distinct actions per 10 minutes 1,016,
740, 224, 106. 1,607 of its first 4,563 decisions (35 %) are the same
unparsed `0 0 0 ; ; ; ; ; ;`, from minute 7 on. The screen from minute 25 is
Minecraft's "open this link or copy it to your clipboard?" dialog for
`https://aka.ms/snapshotbugs?ref=game` (the bug-report button in the Game
Menu), with "Saved screenshot" lines in chat behind it (F2 presses).

**Sandbox change made because of this:** Firefox is installed on the sandbox
and was the default http/https handler, so "Open in Browser" would have sent
a request to Microsoft. The sandbox user's http/https handler now points at a
no-op desktop entry (`~/.local/share/applications/noop-url.desktop`,
`Exec=/bin/true`; `xdg-mime` and `gio` both resolve to it; the game's
environment sets no desktop or `BROWSER` variable, so `xdg-open` uses it).
A backstop loop (`~/runs/browser_guard.sh`) kills any Firefox process and
logs it to `~/runs/browser_guard.log`.

Both are now behind one switch, `~/runs/sandbox_mode.sh session|desktop|status`.
`session` turns them on; every training session calls it at start.
`desktop` turns them off and also stops what sessions leave running between
runs: the game watcher (relaunches the game, closes browsers and file
managers, keeps re-focusing the game), robonet's brain and endpoint, and the
game. It refuses while a session's actor is alive unless given `--force`. A
waiter (`~/runs/queue_end_desktop.sh`) runs `desktop` once the arms queue
prints QUEUE_DONE.

## Not measured yet

- Why the decision interval doubled to about 0.76 s in the unparsed phase.
- Whether the trainer's reward treats unparsed decisions differently from
  parsed no-key ones.
- Thought–action agreement. Only the action tokens are trained, so fluent
  thoughts show the shared weights are not broken in general. They do not
  show that the trained behaviour is healthy. A score of whether the action
  does what the thought says (keys and clicks named in the thought against
  the action that follows) would measure the split directly.
