# Track A training stack: checks before the screen stage (2026-11-26)

Steps A.9–A.11 of [2-methodology.md](../energy-controller/2-methodology.md).
Code: `sili_quartz/continual.py`, `energy_scan.py`, `actor_objective.py`,
`examples/gamescaling_trainer.py`, `examples/gamescaling_actor.py`
(`--replay-dir --sync-dir --energy --dose-file`). All checks on the sandbox's
P40; the 2060 was running the A-0 baselines. Raw numbers.

## Design as built

- **Actor** (2060, fp16, unchanged speed path): saves each decision's frame,
  logs each sampled token and its log-prob, applies the trainer's weight
  updates in place (16 tensors per decision), and runs energy at the dose the
  trainer writes.
- **Trainer** (P40, fp32): windows of 6 consecutive decisions from the last
  600; continual LoRA (rank 8) on **every** Linear (237, vision included,
  lm_head and its tied embedding included) plus direct training of every other
  parameter (3.6M); merged into the weights every 8 steps and shipped as exact
  fp16 diffs.
- **Loss:** −I(past; future) (gaussian_bc on layers[12] features projected to
  d = 32, standardized, plus unit noise so the estimate stays bounded) + a
  PPO-clipped policy term (ε = 0.2, ratio to the actor's sampling log-prob,
  reward = the pointwise MI) + λ·relu(floor − PR)/floor. Gradient norm clipped
  to 1.
- **Energy:** same sites as P0c (MLP activation and attention output of every
  layer), as one Triton scan per site and call; dose read from a tensor so the
  dual moves it inside CUDA graphs.

## Checks

| Check | Result |
|---|---|
| Unit tests (continual, energy scan, objective, gamescaling, controllers) | 62 passed |
| Trainer step, all layers, 2.7k tokens (6 frames) | 7.7–9.4 s, peak 14.9 GiB; warm-up (bank filling, forward only) 1.6 s |
| Merge + sync packet | merge 0.11 s; packet ≈ 1.1 GB (merges change most fp16 entries); the actor deletes packets once loaded, the trainer keeps a full checkpoint every 30 min |
| Actor applying a packet (offline, fp16) | 473 tensors over 30 decisions; **0** mismatches against fp16(trainer); decoder head copy refreshed |
| Energy at dose 0 (wiring check) | 24/24 actions parse |

**Policy-term stability** (100 steps; 37 warm-up, then training; per-token
log-prob of the model's own replayed actions):

| Policy term | log-prob, start → end | raw grad norm | MI (nats) |
|---|---|---|---|
| REINFORCE, both terms | −5.7 → −15.7 | 550–5,200 | 0.8 → 4.3 |
| REINFORCE only | −5.7 → −17.6 | 450–2,200 | 0.8 → 1.1 |
| representation only | −5.7 → −5.7 | 35 → 8 | 0.8 → 3.9 |
| **PPO clip, both terms** (kept) | −5.3 → −3.4 | 17–600 | 0.9 → 4.0 |

The first REINFORCE run had no gradient clip and an unbounded MI code
(30 nats); the rows above are after both fixes.

**Energy dose on the actor** (fp16, CUDA-graph decoder, 24 replayed frames,
cost multiples of P0c):

| dose × P0c | 0 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1 |
|---|---|---|---|---|---|---|---|
| actions parsed | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 6/24 | 0/15 |
| changed entries per decision | 9.9k | 13k | 34k | 127k | 340k | 1.26M | — |

Track A's dose unit is therefore 0.1 × P0c (`TRACK_A_COSTS`); the dual is
capped at 2 units. Energy adds about 18 % per decision on the P40
(0.85 → 1.00 s at fp16).

## Not checked yet

- The live actor with energy and sync on the 2060 (latency under Minecraft).
  The first screen session will show it in `actor.timing.jsonl`.
- Packet size could be cut (top-k or periodic dense instead of per-merge
  dense); not needed on one machine.
