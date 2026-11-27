# Checkpoint probes: repertoire, fan-in, weight spectrum (2026-11-27)

Offline measurements on the saved trainer checkpoints (about minutes 30, 60
and 90 of each run). No game is involved and energy is off. These describe
what the training did to the *weights*, separately from what energy did at
play time.

## Repertoire (`tools/repertoire_probe.py`)

Setup: 120 actions sampled per model on the same 30 open-world states (A2b
and A2g, minutes 0–10), exactly as the actor samples (grammar mask,
temperature 1), with energy off. Each run's last checkpoint.

| model | distinct keys | key entropy | LB share | distinct actions / 120 | median mouse | log-prob per token |
|---|---|---|---|---|---|---|
| base | 13 | 2.46 | 0.38 | 99 | 9 | −0.41 |
| A1 | 9 | 2.55 | 0.16 | 60 | 2 | −0.16 |
| A1-act | 10 | 2.26 | 0.30 | 83 | 4 | −0.23 |
| A1pa | 18 | 2.51 | 0.56 | 99 | 17 | −0.48 |
| A1io | 16 | 2.44 | 0.44 | 114 | 36 | −0.55 |
| A2b (23 min) | 12 | 2.17 | 0.64 | 98 | 6 | −0.29 |
| A2c | 10 | 1.82 | 0.78 | 80 | 2 | −0.15 |
| Tin_c | 15 | 2.43 | 0.43 | 97 | 5 | −0.41 |
| Tin3_c | 13 | 2.44 | 0.38 | 97 | 4 | −0.37 |
| Tin3_e | 16 | 2.63 | 0.41 | 109 | 9 | −0.47 |

- **A2c's weights carry its collapse:** LB at 0.78, with high certainty.
- **A1's weights repeat themselves:** 60 distinct actions of 120.
- **Tin3_e's weights are close to the base model's:** its random play came
  from energy at play time, not from the weights.

## Static fan-in (`tools/fanin_probe.py`)

Each action key's direct fan-in from every layer's output neurons, through the
final norm and the tied head. In every checkpoint, strength is within 0.13 %
of the base model's and direction within cosine 0.9997. Overlap is identical
(0.158). The wiring to the outputs does not change.

## Active fan-in (`tools/fanin_active.py`)

Activation × direct path to the key that was actually emitted. Same 24 replay
windows (A2b, minutes 0–10; 1,103 key positions) for every model.

- **drive:** total contribution to the emitted key;
- **spread:** participation ratio over neurons;
- **overlap:** mean |cosine| with the other keys' contribution vectors.

| model @ step | drive | spread | overlap | LB drive | W drive | D spread |
|---|---|---|---|---|---|---|
| base | 18.57 | 4,313 | 0.188 | 21.7 | 22.1 | 3,326 |
| A1 @253 / @473 / @689 | 18.72 / 20.78 / 20.92 | 4,323 / 4,384 / 4,177 | 0.185 / 0.184 / 0.185 | 21.8 / 24.2 / 24.3 | 22.4 / 25.2 / 25.6 | 3,152 / 2,907 / 2,729 |
| A1-act @211 / @395 / @564 | 18.68 / 19.82 / 19.78 | 4,253 / 4,331 / 4,159 | 0.182 / 0.182 / 0.189 | 22.7 / 23.8 / 23.9 | 22.2 / 23.8 / 23.9 | 3,354 / 3,281 / 3,147 |
| A2c @239 / @461 / @674 | 19.54 / 21.76 / 21.52 | 4,057 / 4,160 / 4,232 | 0.193 / 0.192 / 0.193 | 25.0 / 30.1 / 29.9 | 22.4 / 24.7 / 23.6 | 2,819 / 2,718 / 2,750 |
| A1pa @247 / @463 / @649 | 16.82 / 14.85 / 15.25 | 3,827 / 3,390 / 3,400 | 0.202 / 0.213 / 0.211 | 20.4 / 18.1 / 18.7 | 19.2 / 16.8 / 17.3 | 2,955 / 2,807 / 2,830 |
| A1io @251 / @457 / @630 | 17.36 / 15.78 / 15.60 | 3,955 / 3,555 / 3,492 | 0.196 / 0.208 / 0.209 | 21.6 / 20.6 / 20.2 | 19.7 / 17.4 / 17.2 | 3,045 / 3,036 / 2,961 |
| Tin_c @213 | 17.76 | 4,017 | 0.207 | 20.6 | 21.5 | 3,041 |
| Tin3_c @218 / @338 / @518 | 18.55 / 18.30 / 18.08 | 3,944 / 3,832 / 3,769 | 0.193 / 0.197 / 0.198 | 21.5 / 21.4 / 21.1 | 22.3 / 22.0 / 21.8 | 3,023 / 2,862 / 2,773 |
| Tin3_e @214 / @402 / @592 | 17.46 / 15.91 / 15.51 | 3,883 / 3,696 / 3,612 | 0.197 / 0.198 / 0.201 | 20.7 / 18.4 / 17.9 | 20.9 / 19.1 / 18.9 | 3,261 / 3,329 / 3,237 |
| A2b @233 | 21.86 | 4,573 | 0.178 | 27.4 | 24.2 | 3,659 |
| A2_klcov @121 | 17.64 | 4,066 | 0.189 | 20.3 | 20.8 | 2,913 |

Two patterns:

- **Sharpening** (A2c's LB +38 %; A1 and A1-act overall): drive rises
  through about the same number of neurons.
- **Damage** (A1pa and A1io, the format-bug runs; Tin3_e, which trained while
  the actor played under energy): drive and spread fall, and overlap rises.
  Fewer, less exclusive neurons drive each key.

This makes the active fan-in a candidate token-health metric for the
random-init token check.

## Weight spectrum (`tools/spectrum_probe.py`)

Per language-model weight matrix, against the base: relative change, the
effective rank of the change, the share of the change in its top 8 singular
values, and intruder dimensions (Shuttleworth et al. 2025: top-10 singular
directions with best |cosine| < 0.5 to every pretrained direction).

| run | relative change (min 30 → 90) | effective rank of the change | top-8 share | intruders |
|---|---|---|---|---|
| A1 | 0.58 % → 0.91 % | 202 → 339 | 0.34 | 0 |
| A1-act | 0.56 % → 0.89 % | 185 → 288 | 0.35–0.38 | 0 |
| A1pa | 0.58 % → 0.85 % | 198 → 330 | 0.30–0.34 | 0 |
| A1io | 0.57 % → 0.85 % | 204 → 339 | 0.28–0.33 | 0 |
| A2c | 0.57 % → 0.89 % | 200 → 368 | 0.29–0.34 | 0 |
| Tin3_c | 0.30 % → 0.46 % | 313 → 357 | 0.34–0.37 | 0 |
| A2b | 0.59 % (min 23) | 200 | 0.32 | 0 |
| Tin3_e | 0.35 % (min 30) | 308 | 0.39 | 0 |

**No intruder dimensions, and no low-rank concentration:** the merged rank-8
updates add up to a change whose rank keeps growing. Over 2-hour runs, LoRA
eigenvector collapse is not happening. The weights move by under 1 %, so the
behavioural collapses come from which neurons activate, not from damaged
directions.
