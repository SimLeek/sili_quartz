# GameScaling-0.8B plays Minecraft through robonet: first run (2026-11-26)

Step A.7 of [2-methodology.md](../energy-controller/2-methodology.md). Code:
`sili_quartz/gamescaling.py`, `examples/gamescaling_actor.py`; log on the
sandbox at `~/runs/actor_smoke.jsonl`.

## Setup

- `Qwen3_5ForConditionalGeneration` (vision), fp32, P40; transformers'
  PyTorch fallback for the linear-attention layers (flash-linear-attention is
  not installed).
- Prompt and messages as in the model card: the standard system prompt for
  "Minecraft" (rules describe the screen and mechanics, no goal), 5 past
  (frame, action) turns, direct mode (prefill `<think>\n\n</think>`),
  temperature 1.
- Frames: robonet's AI frame at 854×480 (the "480p" input of GameScaling's
  own Minecraft evaluation), from the fullscreen 1920×1080 game.
- Execution: 6 steps of 33 ms through robonet; the chunk's mouse movement
  split over the steps as offsets from the screen centre.
- 20 decisions, daytime, standing in a jungle at spawn.

## Results

| Measure | Value |
|---|---|
| Actions that parsed | 20 / 20 |
| Seconds per decision | min 3.60, mean 4.29, max 4.82 (budget: 0.2) |
| Prompt tokens | 1,428 with no history, growing by about 450 per past frame to 3,700 with 5 |
| Frame age at decision | 0.01–0.02 s |
| Keys held (key-steps over 20 × 6 steps) | W 27, D 21, Space 7, A 4, LB 3 |
| Mouse per chunk (thousandths of the screen) | from −112 to 52 horizontally, −11 to 81 vertically |

Sample chunks: `52 -11 0 ; D W ; D W ; D W ; D W ; D W ; D W` (turn right,
walk forward-right), `0 8 0 ; ; ; ; LB ; LB ; LB` (look slightly down,
attack).

## Findings

- **The tokenizer has no action special tokens.** `<|action_start|>` and
  `<|action_end|>` are plain text in this checkpoint (six tokens each), so
  generation stops on the string, not a token id. The model still writes
  them correctly every time.
- **Latency is about 21× over the 200 ms budget**, so in real time the game
  runs about 4 s between chunks, with the last step's keys still held.

## Options for the latency (not chosen yet)

1. **Pause the game during inference** (time-paused evaluation, as GameScaling's
   own Minecraft evaluation does): SIGSTOP the game's Java process while the
   model thinks, SIGCONT for each 200 ms chunk. The game clock then only
   moves while actions run.
2. **Reuse the KV cache across turns** instead of prefilling 3,700 tokens per
   call (the reference streaming server keeps a per-session KV history):
   only the new frame (about 450 tokens) is prefilled.
3. **Fewer or smaller history frames** (for example 3 frames, or 640×360).
4. **Faster linear attention**: flash-linear-attention needs Triton, which
   may not support the P40 (sm_61); the 2060 (sm_75, fp16) could run the
   model if the game's share of its 6 GB allows.
