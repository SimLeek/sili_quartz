# 2. Methodology (draft, 2026-11-23)

Built from [part 1](1-introduction.md), mainly 1.11. Everything here is fixed
before any controller result exists. A change after results must say so and
why.

## 2.1 Rules for every experiment

- **No oracles.**
  - Track P may use its training targets.
  - Track A may use only its own screen, its keyboard/mouse stream and its
    internal state.
  - Game-side data (Minecraft statistics, a position-logging mod) is for
    **evaluation only** and is never fed to the agent.
- **Measured two ways.** Every arm reports its gain *and* its cost: wall time,
  GPU memory, and any extra forward passes.
- **Seeds.** Each seed is the same program run again. Track P uses at least 4
  seeds per arm for any claimed comparison. Track A uses the staged protocol
  in 2.4 (screen, repeat, repeat, then 3 seeds), because there is one P40. A
  lead needs an agreeing sign across seeds and a better worst seed.
- **Raw results first.** Each run's tables go to `docs/results/` with no
  verdict. Part 3 collects them, and part 4 draws conclusions.
- **Runs.** Sandbox only, under gpu_guard (80 °C abort), launched detached,
  with a watcher.

## 2.2 Shared parts (build once)

| Part | Choice | Status |
|---|---|---|
| Energy | `sili_quartz/energy.py`, `energy_hooks.py` (MLP and attention sites, decay = drive/40) | built |
| Effect meter | KL(p_energy ‖ p_clean) on next-step outputs, from a clean pass on a sampled subset of steps | to build |
| Surprise meter | running −log p of the model's own output | to build |
| Learning-progress meter | drop in loss over a short and a long window | to build |
| MI estimator | gaussian_bc (primary); InfoNCE (nonlinear arm, d ≤ 64); gaussian on the residuals of a nonlinear forward model (next arm) | benched ([results](../results/2026-11-23-mi-estimator-bench.md)) |
| Health stats | per-layer participation ratio, kept fraction vs PR, branching ratio, dormant fraction, energy shutoff fraction, optimizer second-moment mean and std, weight std | partly built (sili trackers) |
| Health floor | PR ≥ 0.8 × early PR; kept ≥ PR; branching ratio in [0.97, 0.99]; second moment and weight std within the measured shape (ratio to own early values) | from MQAR research, 1.11 |
| Dual update | λ ← max(0, λ + η·(floor − health)) | to build |
| Dose granularity | per region where the model allows it; number of controlled units set by the GPU memory left | to build |

## 2.3 Track P: prediction (accuracy is the extrinsic signal)

**Model:** GameScaling-0.8B's language-model head on text, as in 2c.
- This is a predictor we can measure directly: next-token loss and test
  perplexity on real text (20.7 at the start of 2c).
- It is a *text* predictor, not a frame predictor. In Minecraft (track A)
  GameScaling outputs actions only, with no frame-prediction output.

**Environment: text, WikiText-2** (raw, the same files as 2c, in
`~/collapse/data` on the sandbox). Track A's environment is Minecraft.
- *P-1:* looped and real prompts from the WikiText-2 test split.
- *P-2:* prompts and the real control's data are fresh 64-token blocks of the
  train split each generation. Test perplexity uses 128 fixed blocks of the
  test split.

**Experiment P-1, the loop test** (`loop_vs_temperature`-style; one run per
arm; fast).
- **Arms:** P0a no energy; P0b Mirostat; P0c fixed dose; P1; P1a; P1b; and
  ×R/D on P1.
- **Measures:**
  - loops broken (distinct-3 on 3 looped prompts);
  - on real text, distinct-3 and the original model's perplexity of what was
    written, against the true continuation;
  - the effect KL;
  - time per token.

**Experiment P-2, the collapse loop** (`examples/model_collapse.py`, 10
generations, 4 seeds, greedy teacher).
- **Arms:** P0c (rerun as the matched baseline); P1; P2; P2a; P2+A.
- **Measures:**
  - test perplexity per generation;
  - free-running distinct-3 and looping per generation, *and* during each
    generation's training run, to test the claim that the comfortable phase
    should end within one training run;
  - the training loss floor;
  - the dose trajectory;
  - health stats.
- **Pre-registered reading:**
  - *Better than the fixed dose* means test perplexity at g10 no worse than
    P0c (mean 78.5, worst 85.6), **and** free distinct-3 at g10 above P0c's
    0.086 in at least 3 of 4 seeds.
  - *The phase ends within a run* means free distinct-3 rising during a
    generation's training run while test perplexity holds.

## 2.4 Track A: actor (no extrinsic signal at all)

**Environment:** vanilla Minecraft on the sandbox, played through robonet on
localhost. The AI sees only the screen, through a viewport it zooms and pans
itself, and acts only through keys and the mouse.
- **Containment:**
  - Block OS key combinations (Alt+Tab, Alt+F4, Super) at robonet.
  - Block Esc, or use the no-quit Fabric mod.
  - A watchdog restarts the game into the same world if it exits.
- **Evaluation only:** Minecraft's per-player statistics file (distance,
  blocks mined, items), plus chunks visited if a mod is needed. None of it
  reaches the agent.
- **Environment, decided (user): vanilla Minecraft, Fabric and robonet.**
  - The version is the one the no-quit mod supports; the user is checking
    which.
  - *Setup (user, a short internet window):* sign in, install Minecraft, the
    Fabric loader and the mod.
  - *Mods needed:* the no-quit mod. Add a position/chunk logger only if the
    vanilla statistics file is not enough for evaluation.
  - MineStudio 1.1.6 (Minecraft 1.16.5) is GameScaling's own published
    evaluation route. It is not used here, but remains the reference for
    the model's published numbers.
- **Session length:** 2 hours (user).
- **Stuck or collapsed: the compression check (user).**
  - Every 1–10 minutes, compress the agent's own action stream (the action
    text, or the quantized actions, with zlib or a compressed npz). Record the
    compression ratio.
  - *Very compressible* means the agent is repeating itself: stuck, looped or
    collapsed.
  - *Barely compressible* means it is close to random noise.
  - Both ends trigger a check of the model. It is the same two-sided shape as
    boredom and confusion.
  - Run the same check on the screen frames, to catch the agent being
    physically stuck while its actions still vary.
  - *Proof of a permanent loop* is possible only in a narrow case: a
    deterministic agent and a deterministic game that revisit an exact
    (observation, internal state) pair will loop forever. With noise or
    energy it cannot be proven, so the window rule plus "no new statistics
    for the window" is the practical test.
  - It is a monitor and a check trigger, not a controller input. Using it as
    an input would be allowed (it is the agent's own data), but it would be a
    separate arm.

**Architecture: two, in this order (user).**
1. **GameScaling as it is,** kept as one whole model (model card,
   GameScaling-0.8B-S3).
   - It has no separate predictor *output*. But it is partly recurrent (3
     linear-attention layers to every full-attention layer), so it predicts
     internally: an internal world model, much closer to a brain (user).
   - Actor training on the whole model trains that internal predictor too.
   - *Base:* Qwen3.5-0.8B with a 12-layer vision encoder (patch 16).
   - *Training:* continued pre-training on 1,600 h of keyboard-and-mouse play,
     27% of it Minecraft.
   - *Each call:* the current frame plus a short history of frames and
     actions goes in, and the next **200 ms** of actions comes out as text
     between `<|action_start|>` and `<|action_end|>`. That is 6 key steps of
     33 ms, plus one mouse move and scroll for the chunk. The model may think
     before acting.
   - *Energy:* goes on its own layers. The objective's dual sets the dose.
2. **Dreamer/Plan2Explore style:** a world model, with an actor trained inside
   it on the reward-free objective. Plan2Explore is the reward-free
   exploration variant.

**Where MI is measured: two arms (user).**
- *Central:* the output of transformer block `n_layers // 2`. For
  GameScaling's 24 layers that is `model.layers[12]`, a linear-attention
  block. It is projected to d ≤ 64.
- *Output:* GameScaling's action-chunk stream, its only output in track A.
  It has no frame-prediction output to use.

**Experiment A-0, setup checks** (no training).
- GameScaling's vision path loads and runs on the P40, in the full
  `Qwen3_5ForConditionalGeneration`, not the text-only model used in 2c.
- robonet frames reach it on localhost, and its action text is parsed into
  robonet key and mouse events that reach the game.
- The untouched model's behavior is recorded as the reference for "does not
  degrade".

**Experiment A-1, reward-free exploration** (2 h sessions, one at a time on
the single P40; staged, below).
- **Arms:**
  - A1 (predictive information under the health floor; λ is the dose);
  - the ablations A1−act, A1+self and A1−λ;
  - A2 (empowerment);
  - A3 (region-to-region MI);
  - A1+bored.
- **Baselines:** untouched GameScaling; the fixed energy dose; no energy; a
  random policy.
- **Measures (evaluation only):** distance travelled, unique blocks and items
  seen and obtained, chunks visited, time alive, and how long it takes to
  get stuck (the compression check above, plus no new statistics for the
  window).
- **Measures (internal):** the I_pred estimate, health stats, the dose and λ
  trajectories, and wall time per step.
- **The four GameScaling questions:**
  1. Does training degrade the untouched model's play?
  2. Does it improve it?
  3. Can a new init approach it?
  4. Does whole-model actor training also improve its internal prediction?
     There is no prediction output to read. The only measure is a probe: a
     small readout trained on frozen centre-block states to predict the next
     frame's features, fitted before and after training. It is used for
     evaluation only and is never trained into the model. The Dreamer/
     Plan2Explore models stay separate (architecture 2).

**Staged protocol (user): one P40, so runs go one at a time.**
1. **Screen.** One 2 h session per arm and baseline. That is 11 sessions,
   about 22 h. Any arm that shows one of the following is ruled out:
   - collapse: the compression check trips and does not recover within the
     session;
   - a persistent health-floor violation;
   - NaN or divergence;
   - the game exits, or containment breaks.
2. **Repeat.** Rerun the survivors once, and rule out on the same criteria.
3. **Repeat again.** The same.
4. **Three seeds.** Train the remaining arms on three seeds while measuring
   performance with the evaluation-only statistics.

The elimination criteria above are fixed now. The compression thresholds come
from A-0 (normal play and a random policy). This replaces the general
"4 seeds per arm" rule for track A: survival is decided per run, and the
performance claims use the final three seeds.

**Experiment A-2, later:** sili's learned prefrontal regulator (L) setting
per-region targets, trained on the same objective.

## 2.4a Compute (user, 2026-11-23)

**Hardware:**
- *Sandbox (Bee):* the P40 (24 GB, fp32 compute, no fast bf16) and the RTX
  2060. The 2060 drives the display and also runs Minecraft.
- *ai0:* GPUs on the same LAN.

**Memory estimate for training GameScaling-0.8B on the P40.** These are
estimates, measured in A-0 before any choice:
- *Fixed cost (measured weights):* fp32 weights 2.9 GiB (tied embeddings),
  gradients 2.9 GiB, AdamW state 5.7 GiB. About 11.5 GiB before activations.
- *Activations:*
  - One 800 × 600 viewport frame is about 470 vision tokens (patch 16, merge
    2), and each call also carries a history of frames.
  - *Measured* ([results](../results/2026-11-23-gamescaling-train-memory.md)):
    - 8.7 MiB per token for the whole model in training, about 0.36 MiB per
      token per layer. The earlier "1 MiB per layer" was about 3× too high.
    - The largest share is the gated delta net (about 377 KiB per token per
      linear layer), from transformers' pure-PyTorch *chunked* fallback. It
      also runs in training: the state is kept per 64-token chunk, and the
      per-token cost is the chunk intermediates.
    - Next come the fp32 logits kept by the loss (about 1 MiB per token) and
      the MLPs.
  - *What fits:*
    - the fixed fp32 cost is about 11.5 GiB;
    - 1k tokens of backprop fits, at about 20 GiB;
    - 2k tokens does not, at about 29 GiB.
  - *With e-BPTT* (quartz PR #4, per-chunk backward with a carried
    Jacobian, flat memory in T):
    - only the current chunk is stored, e.g. the newest frame plus its action
      tokens: about 500 tokens, about 4.4 GiB;
    - older context is a detached state with carried credit;
    - a smaller chunk scales that down linearly.
    - It is not yet wired into the HF Qwen3.5 layers. That is a build item.
  - *Other levers:*
    - gradient checkpointing of the recurrence (recompute the states in
      backward);
    - flash-linear-attention's chunk kernel. It is a training kernel too,
      with its own backward that recomputes the within-chunk intermediates.
      It should cut the largest share; by how much is not measured. It needs
      Triton: P40 (sm_61) support is unconfirmed, and the 2060 (sm_75) should
      work.
    - Computing the loss in pieces, so the full fp32 logits (about 1 MiB per
      token) are never all held at once.

**If one P40 is not enough, in this order:**
1. **The P40 alone,** with e-BPTT chunks (or checkpointing) and split batches.
2. **Pipeline across the P40 and the 2060 (user).** The 2060 keeps headroom
   for Minecraft, which uses about 1–2 GB of VRAM.
3. **Pipeline across Bee and ai0 over the LAN (user).** At a stage boundary
   the activation is tokens × 1024 × 4 bytes, about 8 MB for 2k tokens, each
   way. Over gigabit that is about 65 ms per direction per micro-batch.

**Latency is a separate constraint.** The game runs in real time and does not
pause between calls, unlike GameScaling's own evaluation, which pauses the
game for each model call.
- Each call must return within the 200 ms chunk it plans, or the agent falls
  behind.
- Pipelining adds memory, not speed. For one live agent it *adds* latency,
  more so across the LAN.
- A-0 measures inference latency on the P40. If it is over budget, the
  levers are a smaller viewport, fewer history frames, or no thinking
  tokens.
- Training updates can run asynchronously from acting, on a replay of the
  last stretch of play.

## 2.5 Order

1. Build the shared meters (2.2).
2. P-1, then P-2 (prediction side, fast feedback, existing harnesses).
3. In parallel, the user does the Minecraft setup window; then A-0.
4. A-1.
5. A-2.

## 2.6 Still open

- The compression-ratio thresholds, from A-0's normal play and from a random
  policy.
- The exact Minecraft version: whichever the no-quit mod supports.
- Whether training fits on the P40 alone, and the inference latency (A-0).
