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

## Speedups (same day, after the first run)

Steady-state seconds per decision (5-turn history, 854×480 frames), from
`sili_quartz/gamescaling.py` with `session_turns=30, fast_decode=True`:

| Step | P40 fp32 | 2060 fp16 |
|---|---|---|
| First run (`generate()`, full prompt every call) | 4.3–4.6 | — |
| Own decode loop, full prompt every call | 2.58 | — |
| + KV cache kept across decisions (only the new frame is prefilled) | 1.07 | 0.93 |
| + CUDA-graph decode, action vocabulary, stop at the end tag | **0.60** | **0.41** |

Checks: cached vs fresh prefill KL 4e-6 and 1e-7 (fast path), same argmax;
graph decode logits within 7e-4 of transformers (30/30 argmax); the action
vocabulary (182 tokens) holds ≥ 99.98% of the model's probability at every
one of 203 decoding steps. FP16 is not used on the P40 (its fp16 rate is
1/64 of fp32).

Where the 0.41 s on the 2060 goes:

| Part | Seconds |
|---|---|
| new-frame prefill: render 0.003, processor 0.008, M-RoPE positions 0.007, vision 0.038, language model 0.112 | 0.18 |
| decode: 19 tokens × 11.6 ms | 0.22 |

- Decode per token is GPU-bound (the CUDA graph did not change it), at 8.9 ms
  with a 2k-position KV buffer and 11.8 ms with a 12k one (P40: 14.5 / 16.0).
- Language-model prefill of 430 tokens: 38% of its GPU time is the
  linear-attention chunk kernel (flash-linear-attention via Triton; on Turing
  it runs no faster than transformers' PyTorch fallback), 28% matrix
  multiplies, 11% attention over the 6k context.
- Flash-linear-attention made the P40 *slower* (decode 31 → 42 ms/token
  before the CUDA graph); the graph decoder does not use it.

## Real-time version: frames, length buckets, streaming, pipelining

- **640×360 frames** (robonet `ai_res`; GameScaling's own Minecraft evaluation
  rendered 640×360 and upscaled to 480p): 251 instead of 436 image tokens per
  frame.
- **Length buckets** for the decode graphs (1k, 2k, 4k, ... positions):
  attention reads only the KV in use. In fp16 on the 2060 the graph itself
  matters too (10.5 ms/token vs 21.5 ms without it).
- **`StreamingPolicy`**: the 18 linear-attention layers carry the past in
  fixed-size state, so the 6 full-attention layers keep only a sliding KV
  window (the 1,007-token system prompt as a sink, plus the last ~5 turns);
  positions keep counting up. Cost per decision is constant for unbounded
  play (the session version re-read 5 turns every 30 decisions, about 1 s, and
  its prefill grew with the context). Check: streamed vs fresh prefill of the
  same transcript KL 4e-6, same argmax, positions identical.
- **Pipelining** (`PipelinedExecutor`): the next decision is computed while
  the current chunk plays.

Live, 40 decisions, 2060 fp16 with Minecraft on the same card:

| Measure | Seconds |
|---|---|
| compute per decision (prefill 0.11 + decode 0.17) | 0.28 (max 0.35) |
| reaction time (frame captured -> first step of its chunk) | 0.28 (max 0.35) |
| interval between chunks (each chunk is 0.2 s of actions) | 0.28 (max 0.35) |

40/40 parsed. Reaction time is a property of this model: a decision has to
read the frame and write about 19 tokens. About 0.2 s is human reaction time
and is the target here; see the note on supermimic in the methodology.
