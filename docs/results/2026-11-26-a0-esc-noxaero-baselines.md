# A-0 again: Esc unblocked, no Xaero mods (2026-11-26)

Re-run of both A-0 baselines ([untouched](2026-11-26-a0-untouched-baseline.md),
[random](2026-11-26-a0-random-baseline.md)) after two changes, so the trained
arms have a baseline under the same conditions:

- **Esc unblocked** in robonet (`blocked_key_combos` no longer has Esc), so
  the model can close the menus it opens;
- **Xaero minimap and world map removed** (the untouched model spent its last
  35 minutes in Xaero's pausing world map), and the Xaero keys stripped from
  the template `options.txt`.

Everything else is unchanged: 7,200 s per session, world reset to the template
seed, settings restored, focus guard, 1 fps screen recording, stats every
5 minutes. A screen-hash watchdog (kills the game if the screen is unchanged
for 180 s) was added after an earlier attempt hung on the pause menu; it did
not fire in either run. Raw numbers only.

Runs on the sandbox: `~/runs/a0u_esc_noxaero/` (untouched GameScaling-0.8B,
17:46–19:46 UTC) then `~/runs/a0r_esc_noxaero/` (random, actions and timing
drawn from that untouched run, 19:48–21:48 UTC).

## Side by side

The first and third columns are the earlier, Esc-blocked runs.

| Measure | untouched, Esc blocked | **untouched, Esc free** | random, Esc blocked | **random, Esc free** |
|---|---|---|---|---|
| Decisions (all parsed) | 23,720 | 24,001 | 23,733 | 24,058 |
| Interval between actions, median | 0.297 s | 0.293 s | 0.297 s | 0.293 s |
| Distinct action strings | 14,524 | 14,641 | 9,583 | 9,803 |
| Action stream, lzma bytes per decision | 3.64 | 3.61 | 3.71 | 3.63 |
| Action zlib ratio per minute, min / p10 / median / p90 / max | 9.3 / 10.1 / 11.8 / 15.7 / 19.4 | 9.3 / 10.5 / 12.1 / 13.7 / 17.8 | 9.8 / 10.2 / 10.5 / 10.9 / 11.1 | 10.0 / 10.3 / 10.7 / 11.1 / 11.4 |
| Esc share per 33 ms step | 0.016 (blocked) | 0.032 | 0.016 (blocked) | 0.032 |
| Game time (play ticks) | 81.7 min | 105.0 min | 80.0 min | 80.0 min |
| World opened to LAN (see below) | no | minute 30 | no | minute 73 |
| Mined | 129 (7 kinds) | 122 (7 kinds) | 139 (8 kinds) | 141 (8 kinds) |
| Picked up | 18 | 48 | 13 | 14 |
| Used (placed) | 0 | 4 (dirt 3, jungle_sapling 1) | 3 | 0 |
| Deaths | 13 | 2 | 11 | 5 |
| Damage taken / dealt | 3,510 / 12 | 660 / 0 | 2,790 / 9 | 1,400 / 0 |
| Travelled (stats total) | 1,381 m | 741 m | 922 m | 1,145 m |
| Walk / under water / on water / climb / fall | 884 / 122 / 79 / 141 / 75 m | 538 / 21 / 31 / 93 / 48 m | 619 / 36 / 111 / 87 / 38 m | 577 / 119 / 80 / 252 / 10 m |
| Jumps | 372 | 330 | 353 | 370 |
| Furthest from spawn at a 5-min sample | 37.1 m | 37.5 m | 21.5 m | 29.5 m |
| Logs / planks / crafting tables | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Screen spatial KB per frame, median (p10–p90) | 37.5 (25.2–45.1) | 29.2 (20.7–40.5) | 37.6 (29.5–45.6) | 29.0 (23.4–36.6) |
| Screen novelty bytes per frame, median (p10–p90) | 3,303 (234–7,206) | 2,357 (97–6,058) | 3,620 (701–5,814) | 3,170 (1,893–4,857) |
| Screen static share, median (p10–p90) | 0.48 (0.02–0.93) | 0.53 (0.08–0.93) | 0.47 (0.23–0.87) | 0.40 (0.27–0.55) |
| Minutes with novelty under 500 bytes per frame | 14 | 26 | 9 | 0 |

Screen sizes are lower in both new runs; the Xaero minimap, a detailed
corner of every in-game frame, is gone.

## Untouched, Esc free: details

- **Mined:** dirt 30, oak_leaves 24, grass 22, jungle_leaves 15,
  grass_block 13, vine 11, fern 7. **Picked up:** dirt 36, stick 5,
  wheat_seeds 5, jungle_sapling 1, oak_sapling 1. It dug down into dirt (the
  1 fps frame at minute 50 is the inside of a dark pit) and ended holding 29
  dirt.
- **Deaths:** 2 (minutes 40–50), against 13 with Esc blocked.
- **Key share per step:** LB 0.145, W 0.073, Esc 0.032, Space 0.025, A 0.023,
  S 0.022, D 0.021, E 0.016, Ctrl 0.006, RB 0.006.
- **Minutes 50–100, almost no play:** mined 93 → 98 and travelled
  623 → 658 m while the game clock ran at full rate. Frames at minutes 50,
  60, 70, 80, 90 and 100: the pit, the Social Interactions player list, the
  Mods list, the Game Menu, the player list again, then back in game facing
  dirt. Mining and pickups resume after minute 100 (dirt 6 → 29).
- **Low-novelty minutes** (under 500 bytes per frame): 24, 50, 56, 59, 69,
  72, 74–84, 86–92, 110, 111. The block at 74–92 is the menu stretch above.

Stats every 5 minutes (the stats file is written at autosave, so a row can lag
a few minutes):

| min | mined | picked up | deaths | travelled (m) | from spawn (m) | game min in window |
|---|---|---|---|---|---|---|
| 10 | 0 | 0 | 0 | 40 | 19.1 | 1.8 |
| 15 | 9 | 2 | 0 | 66 | 17.5 | 2.8 |
| 20 | 12 | 3 | 0 | 110 | 19.0 | 2.2 |
| 25 | 16 | 3 | 0 | 161 | 10.4 | 2.9 |
| 30 | 27 | 4 | 0 | 263 | 18.2 | 3.6 |
| 35 | 32 | 7 | 0 | 344 | 21.3 | 3.3 |
| 40 | 46 | 10 | 0 | 396 | 37.5 | 3.4 |
| 45 | 48 | 10 | 1 | 463 | 12.6 | 5 |
| 50 | 60 | 10 | 2 | 612 | 8.8 | 5 |
| 55 | 87 | 14 | 2 | 623 | 12.5 | 5 |
| 60–70 | 93 | 20 | 2 | 625–634 | 13.6–14.0 | 5 |
| 75–85 | 96 | 23–25 | 2 | 642–654 | 13.5–13.9 | 5 |
| 90–95 | 98 | 27 | 2 | 658 | 14.0 | 5 |
| 100 | 103 | 31 | 2 | 658 | 14.0 | 5 |
| 105 | 105 | 33 | 2 | 668 | 13.4 | 5 |
| 110 | 111 | 39 | 2 | 709 | 13.7 | 5 |
| 115–120 | 118 | 45 | 2 | 729–731 | 13.4–14.9 | 5 |
| 125 (final) | 122 | 48 | 2 | 740 | 13.4 | 5 |

## Random, Esc free: details

- **Mined:** oak_leaves 61, grass 24, jungle_leaves 24, vine 18, fern 11,
  seagrass 1, tall_seagrass 1, cocoa 1. **Picked up:** wheat_seeds 4,
  cocoa_beans 3, stick 3, jungle_sapling 3, oak_sapling 1.
- **Deaths:** 5; killed by drowned 2, zombie 1 (the other two have no
  killer entity in the stats).
- **Key share per step:** within 0.002 of the untouched run for every key
  (LB 0.144, W 0.072, Esc 0.032).
- Steady progress through the session: mined 3 at minute 10, 50 at minute 50,
  141 at the end; no long flat stretch like the untouched run's minutes
  50–100.

## Pausing: Esc and LAN

With Esc free, both policies press it about once every 31 steps (0.032 per
step), so the pause menu opens and closes all the time. Game minutes per
5-minute window, before and after the world was opened to LAN:

- untouched: 1.8–3.6 per window until LAN, then exactly 5;
- random: 1.9–3.2 per window until LAN, then exactly 5.

**Both policies opened the world to LAN** from the Game Menu (the game's log:
"Started serving on 38947" at 18:16 UTC, minute 30 of the untouched run;
"Started serving on 42587" at 21:01, minute 73 of the random run). A LAN world
no longer pauses, which is why both game clocks run at full rate afterwards
and why the untouched run's game time (105 min) is higher than before. It also
means the game listened on a LAN port and sent LAN-discovery announcements for
the rest of each session; a restart closes it. Each new session restarts the
game, so it does not carry over, but any arm can do it again, and when it
happens changes that arm's game time.

## Settings the policies changed

From the stats loop's comparison against the template (5-minute resolution):

| min | untouched | random |
|---|---|---|
| 10 | — | lang enws (Shakespearean English) |
| 20 | — | lang es_ec, soundCategory_neutral 0.48, JEI client config added |
| 25 | — | modelPart_left_sleeve false |
| 40–45 | lang enp (pirate) | fovEffectScale 0.77, toggleSprint true, lang es_ar, oculus.properties edited |
| 80 | — | fov −1 (from 0), lang es_ve |
| 100 | — | lang es_uy, toggleSprint back to default |
| 115 | lang es_mx, chatHeightUnfocused 0 | lang esan, soundCategory_block 0.07, soundCategory_record 0.58 |
| 120–125 | autoSuggestions false, chat line spacing / width / focused height / text background changed, lang es_ve | (final: soundCategory_block 0.90, soundCategory_record 0) |

The language list is the most-clicked part of the options: the untouched model
went English → pirate → two Spanish variants, the random policy through six
languages. The pirate-speak frames in the reasoning probes
([base behaviors](2026-11-26-reasoning-probe-observations.md)) come from the
untouched run's minutes 45–115. The template restore at each session start
puts all of these back.

## Reasoning probes during these runs

`examples/reasoning_probe.py` ran beside the actor from 19:15 to 21:43 UTC on
the live 1 fps frame, untouched weights throughout: the untouched run's last
31 minutes, then the random run's first 115 minutes. 596 samples (4 per
probe), 593 with a valid action, **187 with a thought** (31 %). Scored with
`tools/probe_repetition.py`:

| | thoughts | distinct3 | reuse4 |
|---|---|---|---|
| whole run | 187 | 0.715 | 0.137 |
| 0–30 min (untouched's frames) | 34 | 0.851 | 0.061 |
| 30–60 min (random's frames from here) | 36 | 0.867 | 0.050 |
| 60–90 min | 30 | 0.861 | 0.056 |
| 90–120 min | 44 | 0.837 | 0.070 |
| 120–148 min | 43 | 0.844 | 0.059 |

- **No sentence of four or more words appears in two different thoughts.**
- Most repeated 3-word phrases (thoughts containing them): "so i should" 62,
  "the player has" 51, "to destroy the" 50, "i need to" 49, "destroy the
  tree" 49, "the player is" 49, "return to the" 43, "to the world" 39.
  The tutorial toast ("break down a tree") and closing menus to get back to
  the world are the recurring subjects.
- reuse4 rises with the number of earlier thoughts it is measured against
  (0.05–0.07 in a 30-minute window, 0.137 over the whole run), and distinct3
  falls with it. Trained arms have to be compared against this baseline at
  the same thought count, not just the same duration.

## Notes for the trained arms

- These two runs are the A-0 reference for the screen stage (Esc free, no
  Xaero). The trained arms run under the same conditions, with the same
  settings restore and watchdog.
- Game time is not comparable across arms while LAN opening is possible; the
  stats loop does not yet record it. The game's own log has the
  "Started serving" line, so it can be read per session afterwards.
