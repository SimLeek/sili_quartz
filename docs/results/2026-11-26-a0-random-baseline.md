# A-0: random policy, one 2 h Minecraft session (2026-11-26)

Second half of step A.8 of [2-methodology.md](../energy-controller/2-methodology.md).
Companion to the [untouched-GameScaling session](2026-11-26-a0-untouched-baseline.md).
Raw numbers only; no thresholds are chosen here.

Run `a0_random_875e04b` on the sandbox (`~/runs/a0_random_875e04b/`), same
files as the untouched run.

## What "random" means here

`examples/gamescaling_actor.py --random-from ~/runs/a0_76fa621/actor.jsonl --seed 0`:
no model. Each decision is one of the untouched run's 23,720 actions, drawn
uniformly with replacement, after a wait drawn from that run's decision
times. Keys, mouse movement and cadence match the untouched model; the
choice ignores the screen and the past. A uniform draw over the raw action
space (74 keys per 33 ms step, any mouse offset) was not used: it would hold
many keys at once and say little about the model.

Everything else is identical to the untouched session: world reset to the
template seed, settings restored, robonet with the same blocked keys (F1, F3,
F11, Esc, ...), the focus guard, 1 fps screen recording, stats every 5 min,
7,200 s.

## Side by side

| Measure | untouched (`a0_76fa621`) | random (`a0_random_875e04b`) |
|---|---|---|
| Decisions (all parsed) | 23,720 | 23,733 |
| Interval between actions, median | 0.297 s | 0.297 s |
| Distinct action strings | 14,524 | 9,583 |
| Action stream, lzma bytes per decision | 3.64 | 3.71 |
| Action zlib ratio per minute, min / p10 / median / p90 / max | 9.3 / 10.1 / 11.8 / 15.7 / 19.4 | 9.8 / 10.2 / 10.5 / 10.9 / 11.1 |
| Game time (play ticks) | 81.7 min | 80.0 min |
| Mined | 129 (7 kinds) | 139 (8 kinds) |
| Picked up | 18 | 13 |
| Used (placed) | 0 | 3 (dirt 1, oak_sapling 2) |
| Deaths | 13 | 11 |
| Damage taken / dealt | 3,510 / 12 | 2,790 / 9 |
| Travelled (stats total) | 1,381 m | 922 m |
| Walk / under water / on water / climb / fall | 884 / 122 / 79 / 141 / 75 m | 619 / 36 / 111 / 87 / 38 m |
| Jumps | 372 | 353 |
| Furthest from spawn at a 5-min sample | 37.1 m | 21.5 m |
| Logs / planks / crafting tables | 0 / 0 / 0 | 0 / 0 / 0 |
| Screen spatial KB per frame, median (p10–p90) | 37.5 (25.2–45.1) | 37.6 (29.5–45.6) |
| Screen novelty bytes per frame, median (p10–p90) | 3,303 (234–7,206) | 3,620 (701–5,814) |
| Screen static share, median (p10–p90) | 0.48 (0.02–0.93) | 0.47 (0.23–0.87) |
| Minutes with novelty under 500 bytes per frame | 14 | 9 |

## Random session details

- **Mined:** oak_leaves 51, grass 44, fern 18, vine 16, jungle_leaves 6,
  tall_seagrass 2, grass_block 1, oak_sapling 1.
- **Picked up:** wheat_seeds 8, oak_sapling 3, stick 1, dirt 1.
- **Deaths:** 11, all killed by zombies. Respawn worked through random
  clicks; health 0 at the last sample (dead on the death screen).
- **Key share per step:** LB 0.164, W 0.106, A 0.036, Space 0.036, S 0.032,
  D 0.028, E 0.025, Esc (blocked) 0.016, RB 0.010. Within 0.003 of the
  untouched run for every key.

Stats every 5 minutes:

| min | mined | picked up | deaths | travelled (m) | from spawn (m) | inventory |
|---|---|---|---|---|---|---|
| 5 | 20 | 1 | 0 | 54 | 8.4 | wheat_seeds 1 |
| 10–20 | 25 | 1 | 0 | 73 | 6.2 | wheat_seeds 1 |
| 30 | 29 | 1 | 0 | 118 | 8.3 | wheat_seeds 1 |
| 40 | 51 | 2 | 0 | 222 | 13.9 | wheat_seeds 2 |
| 50 | 74 | 5 | 0 | 327 | 15.9 | wheat_seeds 2, oak_sapling 1, dirt 1 |
| 60 | 88 | 8 | 0 | 408 | 16.2 | stick, oak_sapling, dirt, wheat_seeds 3 |
| 70 | 98 | 10 | 0 | 492 | 21.5 | stick, oak_sapling, wheat_seeds 4 |
| 75 | 104 | 10 | 1 | 547 | 10.1 | — |
| 80–95 | 110 | 11 | 1 | 618–665 | 17.6–19.7 | wheat_seeds 1 |
| 100 | 116 | 12 | 2 | 684 | 7.4 | wheat_seeds 1 |
| 105 | 125 | 12 | 5 | 806 | 16.0 | — |
| 110–115 | 137 | 13 | 10 | 911 | 3.8 | — |
| 120 | 139 | 13 | 11 | 922 | 6.0 | — |

## Where the random policy stalled

- Minutes 10–20: stats flat and almost no game time. The 1 fps frame at
  minute 15 shows it in-game in tree leaves, so the pause screen came and
  went within the window (autosave lag blurs the timing).
- Minute 88: the inventory with JEI open (does not pause the game).
- Minute 112: the game's Options → Controls screen (pauses the game). Random
  clicks there can rebind keys. The next session's settings restore puts
  them back, but within a session a rebind would change what keys do.
- Low-novelty minutes: 22–24, 70, 103, 113–115, 118.

## Notes for the thresholds and the next arms

- The random stream is the noise end of the action-compression check: its
  per-minute zlib ratio stays within 9.8–11.1. The untouched model's normal
  play overlaps that band (its p10 is 10.1) and reaches 19.4 when it repeats
  itself (the Xaero map period, minutes 85–119).
- Both sessions lost about 40 minutes of the 120 to pause screens (game
  time 81.7 and 80.0 min): the untouched model in Xaero's world map, the
  random policy in its own menus. The open choice from the untouched
  write-up (unblock Esc, stop the map pausing, or measure against game
  time) applies to both.
- Game minutes per 5-minute window, from `play_ticks` in `stats.jsonl`
  (the stats file is written at autosave, so a window can lag by a few
  minutes):
  - untouched: 5, 5, 5, 2.5, 2.5, then 5 per window to minute 80, 6.6, then
    0 for the last 7 windows (the Xaero map);
  - random: 5, 1.9, 0, 0, 2.1, 1.0, 6.9, 3.1, then 5 per window to minute 80,
    0.1, 1.9, 3.1, 5, 5, 4.9, 0, 0.1. Its pauses are spread out; the
    untouched model's are one long block at the end.
