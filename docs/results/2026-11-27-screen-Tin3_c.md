# Screen stage, Tin3_c: input tether on layers 1–12 only, no drive (2026-11-27)

Commit 86327aa, `~/runs/t_Tin3_c/`, 08:59–10:59 UTC, full 2 hours. Settings:

- `--tether chunks --tether-chunks 3`: input tether on chunks 1–3, layers 1–12.
- Layers 13–24 get no gradient from any loss in this run.
- No output tether, no MI, no policy term.
- `--health pr` with the dual dose (it stayed under 0.01).

This is the reference for Tin3_e (the same plus energy) and Tin3_curio (plus
the energy-curiosity reward).

## What happened

20,319 decisions, 693 trainer steps. Per 10 minutes:

| minute | decisions | distinct | keys per decision | LB share | median mouse | entropy | log-prob | input-tether error | copy baseline | PR |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,530 | 1,312 | 4.5 | 0.16 | 30 | 0.44 | −0.45 | 1.89 | 0.57 | 58.8 |
| 10 | 1,707 | 1,389 | 2.2 | 0.21 | 27 | 0.40 | −0.41 | 1.16 | 0.24 | 58.5 |
| 20 | 1,735 | 1,278 | 1.7 | 0.12 | 13 | 0.40 | −0.40 | 0.71 | 0.20 | 55.0 |
| 30 | 1,777 | 1,212 | 1.5 | 0.10 | 11 | 0.35 | −0.33 | 0.27 | 0.13 | 54.4 |
| 40 | 1,750 | 1,296 | 1.2 | 0.17 | 16 | 0.42 | −0.40 | 1.24 | 0.26 | 52.6 |
| 50 | 1,788 | 1,285 | 1.4 | 0.11 | 14 | 0.39 | −0.36 | 0.63 | 0.19 | 52.0 |
| 60 | 1,739 | 1,149 | 2.7 | 0.15 | 9 | 0.38 | −0.36 | 1.11 | 0.32 | 49.1 |
| 70 | 1,793 | 425 | 3.4 | 0.16 | 0 | 0.19 | −0.14 | 0.70 | 0.20 | 51.9 |
| 80 | 1,793 | 501 | 3.0 | 0.12 | 0 | 0.20 | −0.14 | 1.09 | 0.25 | 50.2 |
| 90 | 1,534 | 1,121 | 8.9 | 0.44 | 41 | 0.29 | −0.31 | 1.28 | 0.35 | 52.0 |
| 100 | 1,547 | 1,178 | 9.5 | 0.36 | 36 | 0.41 | −0.46 | 1.19 | 0.34 | 51.7 |
| 110 | 1,626 | 1,105 | 7.5 | 0.37 | 4 | 0.37 | −0.50 | 0.86 | 0.23 | 52.2 |

Game, from the final stats file:

- **Survival:** 9 deaths, at least 2 killed by a zombie; 85 min of play
  time.
- **Mined, 77 blocks:** oak leaves 29, grass 20, jungle leaves 10, vine 5,
  fern 5, grass blocks 4, tall seagrass 2, dirt 2.
- **Movement:** about 1.4 km, 391 jumps.

Sampled frames at minutes 75 and 100 are "You Died!" screens. It respawned
repeatedly but kept dying, and spent long stretches on the death screen.
Minutes 70–90, with the mouse median at 0 and entropy at 0.19, line up with
death-screen time. The output never went random: action entropy stayed at
0.19–0.45 nats, against 1.9 in FULL_c. It changed chat settings
(`chatColors`, `chatDelay`, `chatLinksPrompt`) and `autoJump` around minute
25.

## Drift from the base model (checkpoint step 218, about minute 30)

| frames | LB share base → Tin3_c | median mouse base → Tin3_c | top keys base | top keys Tin3_c |
|---|---|---|---|---|
| minutes 5–15 | 0.27 → 0.10 | 11 → 19 | LB, W, A, Enter | LB, W, Enter, Space |
| minutes 22–28 | 0.05 → 0.00 | 8 → 9 | W, MB, LB, Esc | W, MB, A, E |

Cosine between base and checkpoint states at the decision token, minutes
5–15:

| layers 1–8 | layers 9–12 | layers 13–16 | layers 17–20 | layers 21–24 |
|---|---|---|---|---|
| 1.00 → 0.97 | 0.98 → 0.94 | 0.92 → 0.88 | 0.90 → 0.93 | 0.95 → 0.98 |

Almost the same profile as Tin_c, which tethered all six chunks. Layers 13–24
got no gradient here, so the dip at 13–16 is the change in layers 1–12
carried downstream and growing in the middle. The output side was trained on
the old representation and nothing retrains it, so it clicks less.

## Empowerment check (mean nats per set)

| checkpoint | open world A2g | open world A2b | pause menu | death screen | link dialog |
|---|---|---|---|---|---|
| base | 161.3 | 164.7 | 162.7 | 164.5 | 171.0 |
| step 218 (~min 30) | 155.5 | 157.4 | 162.5 | 156.0 | 164.7 |
| step 338 (~min 60) | 150.6 | 152.5 | 158.6 | 152.7 | 158.2 |
| step 518 (~min 90) | 163.1 | 164.1 | 167.0 | 164.1 | 168.3 |

No separation at any checkpoint. The pause menu and the dialog score at or
above the open world. Chunk 1's next-frame error on A2g's frames stayed at
1.87–1.91, against the copy baseline's 0.16–0.17, while its error on its own
replay fell to 0.27–1.3. The tether learns its own recent world, not next
frames in general, so the estimator is not usable yet.

## Reading

- **Tethering only the input side kept the output from going random**, unlike
  FULL_c (the output tether) and FULLe_c (output energy).
- **It still changed behaviour:** fewer clicks, many deaths. A tether with
  no drive shifts the representation the output side reads, and nothing
  trains the output side to follow. The output needs a policy of its own,
  which is what Tin3_curio adds.
