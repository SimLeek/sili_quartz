# A-0: untouched GameScaling, one 2 h Minecraft session (2026-11-26)

First half of step A.8 of [2-methodology.md](../energy-controller/2-methodology.md):
the reference statistics for the untouched model. The random-policy session
is the second half. Raw numbers only; no thresholds are chosen here.

Run `a0_76fa621` on the sandbox (`~/runs/a0_76fa621/`): `actor.jsonl`
(every decision), `actor.timing.jsonl`, `stats.jsonl` (every 5 min),
`final_stats.json` (the game's stats file at the end), `screen/` (1 fps
320×180), `screen_complexity.csv`, `action_compression.csv`, `temp.csv`.

## Setup

- GameScaling-0.8B-S3, untouched weights, fp16 on the RTX 2060;
  `examples/gamescaling_actor.py` at commit 76fa621 (StreamingPolicy: KV
  reuse with a sliding window, CUDA-graph decoder, 640×360 frames,
  pipelined executor), temperature 1.
- System prompt: the model-card template for Minecraft plus the respawn and
  menu-arrow lines (`MINECRAFT_RULES`). No goal.
- Minecraft Forge 1.20.1 (Prism instance "1.20.1"), world reset to the
  template seed before the start, game and mod settings restored from the
  template, fullscreen 1920×1080. Mods include no_way_out, JEI, Jade,
  Xaero's minimap and world map.
- robonet blocks F1, F3, F11, Esc, Alt+F4, Alt+Tab, Win and the Ctrl+Alt
  combos. Relative mouse; robonet draws the cursor into the AI frame.
- The focus guard closes file-manager and browser windows the mods open.
- 7,200 s; gpu_guard abort at 85 °C (peak 64 °C).

## Decisions and timing

| Measure | Value |
|---|---|
| Decisions | 23,720 in 7,198 s (3.3 per second) |
| Parsed | 23,720 (100 %) |
| Seconds per decision | median 0.296, p90 0.334, p99 0.403 |
| of which prefill / decode | median 0.099 / 0.194 |
| Reaction time (frame → action start) | median 0.297, p90 0.335, p99 0.403 |
| Interval between actions | median 0.297, mean 0.303 |
| Prompt tokens (steady state) | 2,419 |

## Actions

Share of the 142,320 33 ms steps in which each key is held:

| LB | W | A | Space | S | D | E | Esc* | RB | Ctrl | Shift | Q | F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.165 | 0.107 | 0.037 | 0.036 | 0.029 | 0.028 | 0.026 | 0.017 | 0.010 | 0.004 | 0.004 | 0.002 | 0.002 |

\* emitted, but blocked by robonet.

- Keys held per step: none 60 %, one 34 %, two 5.4 %, three or more 0.9 %.
- Mouse (thousandths of the screen per 200 ms chunk): median |dx| 3,
  |dy| 4; 35 % of chunks move left, 34 % move up. Scroll in 3.4 % of chunks.
- 14,524 distinct action strings. The most common is "no movement, no keys"
  (1,365), then "LB held all chunk" (478).
- Whole action stream: 1.45 MB of text, 86 KB with lzma (3.6 bytes per
  decision).

## Game statistics

Every 5 minutes, from the world's stats file and `level.dat`. These only
change at the game's autosave.

| min | mined | picked up | deaths | travelled (m) | from spawn (m) | health | inventory |
|---|---|---|---|---|---|---|---|
| 5 | 21 | 3 | 0 | 143 | 13.3 | 20 | wheat_seeds 1, dirt 2 |
| 10 | 26 | 5 | 0 | 237 | 37.1 | 20 | wheat_seeds 1, dirt 2 |
| 15 | 29 | 5 | 2 | 396 | 6.0 | 20 | — |
| 20 | 31 | 5 | 3 | 433 | 10.9 | 20 | — |
| 30 | 45 | 7 | 3 | 547 | 24.9 | 20 | dirt 2 |
| 40 | 64 | 9 | 6 | 720 | 2.9 | 20 | — |
| 45 | 81 | 11 | 10 | 829 | 14.7 | 15.3 | wheat_seeds 1, dirt 1 |
| 50 | 85 | 11 | 11 | 931 | 13.2 | 20 | — |
| 60 | 108 | 12 | 12 | 1,181 | 7.8 | 20 | dirt 1 |
| 70 | 119 | 15 | 12 | 1,222 | 8.5 | 20 | dirt 4 |
| 75 | 122 | 16 | 12 | 1,252 | 8.9 | 20 | dirt 4, oak_sapling 1 |
| 80 | 124 | 16 | 12 | 1,298 | 13.0 | 20 | dirt 4, oak_sapling 1 |
| 85–120 | 129 | 18 | 13 | 1,381 | 8.4 | 4.3 | dirt 1, oak_sapling 1 |

Final totals:
- **Mined (129):** oak_leaves 56, jungle_leaves 21, grass 21,
  grass_block 13, vine 13, fern 3, dirt 2.
- **Picked up (18):** dirt 13, wheat_seeds 2, oak_sapling 2, stick 1.
- **Movement:** walk 884 m, under water 122 m, on water 79 m, climb 141 m,
  fall 75 m, swim 41 m, sprint 14 m. 372 jumps.
- **Deaths:** 13. Killed by zombie 6, drowned (mob) 2, skeleton 1; the
  other 4 have no killer recorded (falls or drowning). 3,510 damage taken,
  12 dealt, 0 kills.
- **Other:** 0 crafted, 0 used, 2 dirt dropped.
- **Milestones:** logs mined 0, planks 0, crafting tables 0.
- Play time 97,994 ticks (81.7 min) of the 120 min session.

## Minutes 85–120: paused in Xaero's world map

From about minute 85 to the end, the screen recording shows Xaero's world
map, its waypoint list and its waypoint editor, with short returns to the
map's black void. These screens pause a single-player game. That is why
play time stopped at 81.7 min and the stats stay flat from minute 85. The
map is closed with M, Esc or a Done button. Esc is blocked by robonet, and
the model pressed M rarely. It did leave the map for short stretches
(minutes 98–109, where novelty rises again), but play time shows little
game time in them.

The earlier, stopped 105-minute session (`a0_ec7eb8c`, before F1 was
blocked) had flat stats from 15–25 min and 100–105 min, so short pauses
also happened there.

## Screen complexity

From `tools/screen_complexity.py` (160×90 grayscale, one-minute windows,
novelty against the previous 10 minutes).

| Window | spatial KB, median (p10–p90) | novelty bytes per frame, median (p10–p90) | static share, median (p10–p90) |
|---|---|---|---|
| min 0–84 (playing) | 38.5 (28.1–44.4) | 3,890 (1,818–7,630) | 0.40 (0.00–0.75) |
| min 85–119 (map) | 32.3 (15.4–46.9) | 807 (67–3,252) | 0.82 (0.47–1.00) |
| all 120 | 37.5 (25.2–45.1) | 3,303 (234–7,206) | 0.48 (0.02–0.93) |

The lowest-novelty minutes are all in the map period (94–97 and 110–116:
under 260 bytes per frame, static share 0.88–1.0). Before minute 85 the
lowest is 493 bytes per frame (minute 43).

## Action-stream compression (the stuck/noise check)

zlib (level 9) ratio of each minute's action text:

| Window | median | min | max |
|---|---|---|---|
| min 0–84 | 11.0 | 9.3 | 19.4 |
| min 85–119 | 14.9 | 11.7 | 18.2 |
| all 120, p10 / median / p90 | 10.1 / 11.8 / 15.7 | | |

Per-minute values are in `action_compression.csv`.

## Observations

- The model taps and holds LB a lot (16.5 % of steps). It breaks leaves,
  grass, vines and grass blocks, but no log in 2 h. Logs need about 3 s of
  continuous LB by hand; its holds are mostly under 1 s.
- It opens the inventory (E) and visits JEI and Xaero's screens often.
- Respawn works: 13 deaths, play continued after each.
- The cursor drawn by robonet sometimes shows in-game too, where the game
  hides its own cursor (the visibility check reads "visible" in-game). Not
  fixed in this run.

## Open for the random-policy session and the thresholds

- The Xaero world-map pause takes 35 of 120 minutes here. A random policy
  presses M (and clicks waypoint buttons) at a different rate, so the two
  sessions would differ in game time, not just behaviour. Options: unblock
  Esc (the pause menu cannot quit the game with no_way_out, and
  pauseOnLostFocus is off), stop the map pausing the game if Xaero allows it,
  or count only unpaused play time. This needs a decision before the random
  session.
