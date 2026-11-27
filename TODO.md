# TODO

Open work first. Finished items keep a one-line entry here and their full
record under [Done](#done) at the bottom; results live in `docs/results/`.

## 0. Working agreements
- `main` is branch-protected: work on branches, open PRs, the user merges.
- Every model family and every implementation of it lives side by side in
  `sili_quartz/models/`, so the torch and sili versions sit next to each other:

      transformer.py        A: base (chunked attention)         torch
      rmt.py                B: passed state tokens              torch
      titans.py             C: memory as weights (+ switches)   torch
      combined.py           B + C (ARMT-style)                  torch
      *_sili.py             same models on sili kernels, same tests

## 1. Memory-family tests (B, C, B+C, A as control) — Done → [details](#done-1)

## 1a. PLAN (user, 2026-11-16), in order
1. **Seeds** — Done → [details](#done-1a1). Seed 1 never trains for either
   family (JOURNAL row 32).
2. **poly2** — Dropped (user, 2026-11-17) → [details](#done-1a2)
3. **Ablation tests** (rows 14/15/24) — Done → [details](#done-1a3)
4. **Seed-1 failure** (JOURNAL row 32): combined fails on *data stream 1* alone; titans
   only with init 1 + data 1; poly2 r16 collapses on seed 2. Hypothesis (user): early training + Adam without stochastic
   annealing; industry would discard such runs, sili may recover them. A hard
   data stream is an ASSET: keep it as a fixture and re-run it under the sili
   optimizer later to see if the optimizer fix heals it. Test: every seed
   escapes the start within N steps (expected to fail under Adam — mark xfail
   with the reason, don't paper over it).
5. **40k seeds for the omega arms** — Done → [details](#done-1a5)
6. **Port all three to sili CUDA kernels (now part of 5)** — titans,
   combined, and the omega window (user, 2026-11-17: the data does not rule any
   of them out). Same tests per kernel; Adam is the known non-exact exception.
   The start-up fixture (1a.4) and the 2b tradeoff tests then run on sili.
No learning-rate schedules as fixes: that is an oracle solution.
No torch step-speed work: speed is not the limit now; speedups happen in sili.
No further B/C-family literature to fold in (user, 2026-11-16).

## 1b. Long-run B+C vs C — runs done, not separated (ledger row 30, open) → [details](#done-1b)

## 2. ATLAS switches — omega kept, poly2 dropped → [details](#done-2)

## 2b. torch + Adam → sili + importance: will our fp4 setup train?
Both planned changes are TRADEOFFS, not wins: each gets a two-axis
measurement (what it should improve, what it should cost), both numbers in the
ledger, the cost quantified and never hidden.

- **Adam → sili importance (single-state optimizer).** Expected: worse
  accuracy/steps until the mean/variance control and eigenvalue stochastic
  annealing land. Certain benefit: RAM, **1/2** of the Adam footprint (weight +
  one importance value; sili keeps no backprop state; a conventional engine with
  a one-value optimizer would be 2/3). Adam remains the diagnostic reference.
- **e-BPTT** — torch first pass done (PR #4) → [details](#done-2b-ebptt).
  Open items are under 3b.

1. **Bend proof first.** Prove that weights + optimizer state in fp4 with
   hadamard / additive LoRAs, plus annealing, can train and learn the recurrent
   MQAR, possibly requiring stochastic learning (stochastic rounding /
   selection). Not necessarily "as well" at equal size: information theory /
   compression may stop low bits from matching fp32 per parameter. If so, state
   the bound and the size that overcomes it: a larger fp4 model can trade
   precision-type complexity for more if-then (decision-type) complexity.
   Deliverable: the comparative model size in fp4 for equal capacity, and any
   other conversion facts that fall out (bits/parameter, rank of the additive
   correction).
   **Answered, narrowly (sili__new #58, corrected in #59):** stochastically
   rounded stored fp4 moves to the right average, with a noise floor √(d·e)
   that lr does not lower. The Hadamard/additive LoRAs and width are what
   lower it. End-to-end MQAR on that kernel is not proved. There may be no
   single parameter-count equivalence (precise work vs decision work): measure
   it per task.
2. **If the proof can't be done:** port a minimal fp4 hadamard / additive
   kernel to CUDA and train a transformer with it on the P40, then measure.
3. **If something is off in training:** look for NVIDIA's solution (their fp4
   training recipes) and biology's. Constraint: no more than **4 bits** of
   optimizer state per parameter. (The user had Gemini estimate how many bits
   per parameter each neurochemical is equivalent to.)

## 2c. Sili energy port and model collapse — Done (PR #6)
Results:
- [pilot](docs/results/2026-11-22-collapse-pilot.md)
- [site sweep + 4 seeds](docs/results/2026-11-23-collapse-site-sweep.md)
- [energy vs loops](docs/results/2026-11-22-energy-vs-loops.md)

Fixed-dose MLP + attention energy holds test perplexity at ~78 over 10
self-trained generations, against greedy's 2025 and matched noise's 2294; this
motivates 2d. Differences from the plan below:
- item 3 ran as the generational loop: the 2060 teacher writes, the P40
  student trains on it, and the student answers by handing its weights back
  (not text). Messages were 64 tokens, not 1k–4k.
- **Not doing: the 1k–4k-token dialogue (user, 2026-11-23), on compute grounds.**
  - Scaled from the measured sweep, 128 samples × 4k tokens is 32× the tokens:
    about 2.7 h per energy generation, about 100 h per seed for the 8 core arms,
    about 17 days for 4 seeds. 2k tokens halves that.
  - Backprop at about 1 MiB/token/layer needs about 96 GiB per 4k sequence, so
    the P40 needs checkpointing (about +1/3 training time).
  - 8 conversations instead of 128 cuts the compute, but gives noisy metrics
    and slow sequential generation (no flash-linear-attention on the 2060).
- item 5's dose sweep did not go down to 1e-6 and had no n + k pass rule.
  Doses run: .001–.1 MLP-only, and the per-site arms.

Model collapse: a language model trained on another LM's samples, chained
generation after generation, loses rare words first (the tails vanish) and
eventually repeats forever (Shumailov et al., Nature 2024, "AI models collapse
when trained on recursively generated data"). Raising sampling temperature does
not fix it: it keeps the tails but loses sentence structure. The claim to test:
even a small dose of neuron energy RL in the model that *generates* the data
(per-neuron temperature, not token temperature) stops the collapse.
Not stochastic annealing: that rescues the optimizer state in self-training.

1. **Torch energy: done (PR #6, `sili_quartz/energy.py`, parity-tested).** Port sili's `EnergyDynamics` (sili__new `sili/energy.py`;
   the older torch version is behind it). Mostly a library rename sili → torch.
   The one real difference: sili's `abs` backward has slope 1 at exactly 0
   (`sili/cpu.py` `abs_backward`, so neurons can wake from zero), torch's has 0.
   Use `torch.where(x == 0, x, x.abs())`. Unit tests: forward/backward match
   sili on fixed inputs, including h = 0.
2. **Model.** "Some model", kept outside the codebase: Qwen3.5-0.8B or 2B
   (hybrid recurrent: 3 linear-attention layers per full-attention layer;
   recurrent states are sensitive to small float errors, so harder to quantize
   and the more useful case). The user's download is GameScaling-0.8B
   (`~/claude_code/GameScaling-0.8B`, a Qwen3.5-0.8B finetune on games:
   keyboard/mouse + vision, for robonet later). Work on a COPY; keep the
   original as is (slow to re-download). P40: fp32 (no fast bf16/fp16); for
   fp16, keep fp16 in global memory and convert to fp32 in local memory per op.
   Later: A_lora ⊙ hadamard(B_low_bit) + C_lora for the recurrent parts.
Method and literature:
[docs/results/2026-11-22-model-collapse-method.md](docs/results/2026-11-22-model-collapse-method.md).
The generational pilot (`examples/model_collapse.py`) comes before the dialogue.

3. **Two models talking to each other.**
   - 2060: runs a copy for inference with neuron energy accumulating,
     occasionally forcing firing / acting as per-neuron temperature.
   - P40: trains with energy (LM loss partly mixed with the energy RL aux
     loss) on the 2060's text, and sends its own text responses back.
   - Turns: a simple end token ends a message and passes the turn; max message
     length a fixed limit in 1k–4k tokens.
4. **Collapse metrics, per round:** rare-token mass (bottom tail of the
   original model's unigram distribution), distinct-n, longest repeated span /
   repetition rate, test perplexity (the student's, teacher-forced
   on fixed real held-out text), and a structure check: the *original* model's
   perplexity on the *generated* samples, where LOW is not good (loops score
   10x better than real text; see docs/results/2026-11-22-energy-vs-loops.md).
   Plus free-running distinct-3: test perplexity always sees real context, so
   it cannot see the student looping on its own output.
5. **Arms:** energy off at T = 1 (expected: collapse by round n); temperature
   sweep (expected: tails kept, structure lost); energy dose sweep from tiny
   (1e-6, expected useless) up, to find the line. Pass = no collapse through
   n + k rounds (k a fixed budget) at some dose, structure kept. Seeds = same
   program run N times. Results to `docs/results/`, ledger row.

## 2d. Minimal energy controller (before sleep, user 2026-11-22)
Write-up in paper order: [docs/energy-controller/](docs/energy-controller/README.md).
Track A environment (2026-11-25): Minecraft 1.20.1 + Forge on the sandbox
(Prism Launcher), driven through robonet on localhost (robot_network PR #5:
bridge actions, headless monitor, key blocklist, kill switch);
`tools/minecraft.py` stops, seed-resets and relaunches the world. Steps
A.1, A.2, A.2b, A.4 done; A.5 needs the relaunch watchdog. See
[2-methodology.md](docs/energy-controller/2-methodology.md) section 2.5.
Part 1, introduction and related work, is a first pass (2026-11-23).
A fixed energy dose is wrong most of the time: raise it to try new things and
to learn new things, lower it to focus and get accurate output. Build the
smallest controller that sets the dose, working before 3 and 4.
- **Signals measured so far** ([energy vs loops](docs/results/2026-11-22-energy-vs-loops.md)):
  - repetition / low surprise: energy switches off 2 to 13x more neurons on a
    loop than on real text, a free "bored" signal → raise the dose;
  - high surprise / incoherence → lower it;
  - an external reward present → lower it (energy RL added nothing on
    dense-reward MQAR);
  - amortized health: dormant fraction, rank collapse, branching ratio
    (`BranchingRatioTracker` can already set energy's density per call) →
    raise it while learning.
- **First test:** the loop test with a surprise-targeting dose, against fixed
  doses and against Mirostat (Basu et al., ICLR 2021), which targets surprise
  through token temperature: the strongest temperature-style baseline.
- **Then:** the collapse loop (2c) with the controller in place of a fixed cost.
- **Why a controller, from the collapse sweep (user, 2026-11-23,
  [site sweep](docs/results/2026-11-23-collapse-site-sweep.md)):**
  - With no outside data, greedy + MLP/attention energy at a fixed dose holds test
    perplexity at 75 → 78.5 from g5 to g10 (4/4 seeds), against greedy's 2025
    and matched noise's 2294. It does this with almost no tuning. The plateau
    is the signal: energy is useful while awake, especially when outside
    training data is not available.
  - Its free-running output stays loopy (distinct-3 0.086). The reading: low
    complexity is energy's "comfortable" phase before learning. That phase
    should end *within one training run*, not over generations, and a fixed
    dose cannot end it. That is what the controller is for.
  - The training loss falls to 0.008 at a fixed dose, against greedy's 0.074.
    Energy should keep a noise floor; lowering that floor once the model
    predicts well enough should let it learn more.
- **Scope:** a controller for practically all neurons, not just the MLP and
  attention sites. Open: how much of it is a simple program and how much an
  AI (learned, e.g. the actor-critic idea).
- **Where it should matter most:**
  - RL with very sparse rewards;
  - stuck training even with next-input prediction. energy_rl was designed to
    break stable attractors and zero-init deadlocks (all-zero RNN synapses with
    all flow into one line trained once energy was on).
- **Test within one run:** the controller should take free distinct-3 up
  during a run while keeping the plateau, against the fixed dose.
- **When this lands, check whether 6 (energy for RL task solving) belongs
  inside it.** Note the two are different "energies": 2c/2d is sili's
  per-neuron homeostatic energy; the Energy-Based Reasoners paper means a
  learned scalar over reasoning states, minimized by looping. That scalar may
  still be a useful signal for the controller (converged → focus).

- **Track A: tether the state to its input (user, 2026-11-27).** Hypothesis:
  something has to keep the model tied to its input (the world), as cortical
  columns do. GameScaling's own training had no prediction loss: behaviour
  cloning on assistant tokens, all weights trained (arXiv 2610.28698 §3).
  Its tie to the world was the label stream: human actions, or VPT-style
  inverse-dynamics labels that saw later frames (which subsets it used is not
  stated), plus ~6 % post-hoc rationales written with the action visible.
  RL drops that tie. Cloning coherent play means the weights carry an
  implicit world model (actions with high MI with the game); our objectives
  (MI between internal projections, entropy floor, energy) do not require
  that coherence, so nothing defends it. Variants, as arms:
  - Not an arm: KL to a frozen copy of the untouched policy. That is the
    behaviour-cloning target itself, an oracle (user, 2026-11-27).
  - `--tether head`: the centre state after acting on frame t predicts frame
    t+1's mean image-token input (detached) through a linear readout;
    representation loss only, never a reward, so pausing is not paid for.
  - `--tether col`: neural columns, no readout: the state's neurons in
    columns of 8, each column's average matches one value of the input
    (reduced from 1024 to 128 values by a fixed random orthogonal projection,
    since the state is no wider than the input here). A column of one neuron
    would only copy the input (user).
  - `--tether chunks` (user): columns across depth, no extra width. Layers in
    chunks of 4; per dimension, the average of a chunk's layer outputs after
    acting must match the chunk's input at the next frame, so each layer can
    correct the average and the signal is re-anchored every 4 layers instead
    of having to survive the whole depth (72+ layers in larger models).
  - Output side (user): the output neurons decide the output, as the input is
    the input; the last chunk's layers average to it (`--tether-out`: at each
    emitted action token, the last 4 layers' average matches that token's
    embedding, the head being tied). Later variant: the whole depth average
    *is* the output. It replaces the pretrained readout, so it needs a gradual
    hand-over from the final layer, not a switch.
  - Removed (user, 2026-11-27): the action-entropy guard. The right entropy
    depends on the moment (low while chopping wood, high for quick actions),
    and the penalty trained the output to be random, mostly in the mouse
    digits. Energy keeps neurons in a useful band; the averaging keeps a few
    forced or exhausted neurons from changing what the model does.
  - Full setup (user): chunk tether + energy on MLP/attention/input/output
    with the dual dose + PR guard + input-share band
    (`--health pr+in`: below its floor the dose is scaled down, so energy's
    waking "dreaming" cannot drown out the input) + amortized row-norm band
    (`--renorm`) + Adam reset at merges. The freeze-gated sleep phase is TODO 3.
  - Energy self-maintenance instead of a loss: energy on input-driven state
    holding its setpoint has to anticipate the input. The energy-only arm
    (Eio_fixed1g) tests this; scored by the probe below.
  - Later: columns on the Gated DeltaNet recurrent state (16 heads x 128 x 128
    per layer, much wider than the input): column means of the state match
    the input. Needs the state at each turn boundary, which the fused kernel
    does not return.
  - Short trained thoughts (user, 2026-11-27): a goal of 5-8 tokens
    ("I must get wood." -> "have wood, crafting table") carried in context and
    trained with the RL policy. That costs ~50-100 ms per decision, not the
    1-2 s of the base model's 50-200-token rationales, and gives the policy a
    persistent intent. **The thought length is a knob to test** (user: 5-8
    tokens, "maybe a few more"): sweep it, e.g. 8 / 16 / 32 tokens, and report
    worst-case regret across the range rather than one tuned value.
    First: `tools/thought_influence.py` measures whether the
    base model's thoughts change its actions at all.
  - Keeping the output repertoire under RL (user, 2026-11-27; A2c collapsed to
    one key): Clip-Cov / KL-Cov built (`--cov`). Next ideas, all knowing the
    action tokens:
    - energy on tokens as virtual neurons: each action token has an energy
      state, with feedback when it is used too much or too little. Needs a
      per-token set point (the site-mean set point silences above-mean units at
      equilibrium);
    - sleep-phase fan-in consolidation: for each output token, trace and
      strengthen the path that can trigger it back through the network
      (possibly a computed fan-in), then make it coherent with the rest of the
      model during sleep, so every token stays reachable. The fan-ins must be
      **exclusive** (user): strengthening one token's path must not raise the
      others, so tokens stay mutually exclusive. Measure how much each token's
      fan-in overlaps the other tokens' (and the shared non-token neurons'),
      and consolidate the exclusive part. Related: lateral inhibition, and the
      channel-orthogonality penalty from sili_peridot's rank control.
  - Every arm is scored by `tools/world_model_probe.py`: ridge from the
    trained state to the untouched model's next-frame input on one fixed
    reference replay, against copy and current-frame baselines.

## 3. Stochastic annealing + health pass: "sleep"
- **Sleep window (≤ 8 h/day, once a day) = annealing + health passes that
  improve the optimizer state.** Sparse backprop runs during BOTH wake and
  sleep. fp4-fix is a separate, later thing (only once we are at fp4). So the
  memory tier does not have to hold a whole day alone: weights learn all day;
  sleep adds the optimizer-state repair. Design the schedule around that.
- Sparse amortized learning with health/annealing may run far below max
  learning speed: closer to inference in some phases, to training in others.
  Memory-family results must be re-checked under THAT rate, not Adam's.
- Stochastic selection has won every training-related comparison so far; it
  will change these properties too: include it as an arm, not a footnote.

## 3b. Sparse row selection without starvation (before 4: its 1000-row subset needs good sparse training)
Formerly 7, random top-k with error feedback. Error feedback (Seide 2014, Stich
2018) stores a residual of the gradient's size, one value per synapse, which
breaks the single-state budget (weight + importance, about 1 byte per synapse).
The open question stays: is there a cheap, correct substitute? See sili__new
`bend_proofs/noise_selected_credit/README.md` (error feedback section).

**Hypotheses (user, 2026-11-22), not measured:** keep one number per input or
output neuron, a selection penalty that rises each time the row is picked and
decays otherwise. Add it to the selection score only, as in DeepSeek-V3's
auxiliary-loss-free MoE load balancing (a per-expert bias, nudged by load).
- **H1, coverage:** |G| top-k + penalty stops the stalls. In the e-BPTT sweep
  every |G|-scored arm had a seed that stalled or collapsed, and noise at
  p = .25 matched dense (9/11). Starvation of the same rows is the suspected
  cause, not shown.
- **H2, unbiased without a buffer:** sample rows with known probability p
  shaped by |G| and the penalty, and divide each update by p. That is unbiased,
  with variance as the remaining cost vs error feedback, not bias. The penalty
  also keeps p from getting tiny, where 1/p blows up.
- **Known gap vs error feedback:** a skipped row's gradient is dropped, not
  deferred. Plain top-k + penalty is still biased: each row learns at a rate
  set by how often it is picked, which the penalty only evens out.
- **The same mechanism as energy:** it is EnergyDynamics' activation cost applied
  to *being selected*. The torch port (`sili_quartz/energy.py`) may drive it as is.

**Test:** the e-BPTT credit sweep with two added arms, (1) |G| top-k + penalty
and (2) penalty-shaped sampling divided by p, against noise p = .25, |G| top-k
and dense. Measure seeds reaching 0.95, worst seed, steps and wall time. Then
small MQAR recurrent transformer models.
- e-BPTT items still open: clean in-window test (episode inside one chunk);
  more steps / seeds at gap 64; larger d for the wall-time saving; K-side
  Jacobian.

## 4. HDD hippocampus
1. Port sili cpu so that forward can add contribution to one subset of input
   rows while backward adds gradient + optimizer updates to another subset.
2. fp4 hadamard working in VRAM on an MQAR example.
3. **GPU stand-in first (user, 2026-11-22): show the GPU case is similar to a
   reasonable HDD case before the slow run.**
   - **Theory says whether it should learn.** Each step trains n rows (backprop)
     and activates m rows (forward) out of N. Treat a trained subset as a write
     and an active subset as a read, and compute the true signal and the random
     false overlap the block should get:
     - *Willshaw* (clipped Hebbian, binary): after M stored pairs a synapse is
       set with p1 = 1 − (1 − (k_in·k_out)/(N_in·N_out))^M. A true target
       collects all k_in of the cue's inputs; a false one collects
       Binomial(k_in, p1). Capacity peaks near p1 = 0.5, at about
       ln 2 · N_in·N_out / (k_in·k_out) pairs.
     - *Kanerva SDM*: a location is active with probability p. Write and read of
       the same item share about p·H of H locations; an unrelated write shares
       about p²·H, the false overlap. The best p for T stored items is about
       (2·H·T)^(−1/3).
     - Map n, m, N and the items stored per window onto these models. Then
       predict signal vs false overlap, and with it learns / doesn't, at GPU size
       and at HDD size.
   - **HDD sparsity on the GPU:** a banded matmul, nonzero only within ~y above
     and ~y below the diagonal, else 0. This mimics the HDD layout, where
     nearby time rows are chosen together and read sequentially.
   - **Checks on the GPU block:** does it learn MQAR and ordinary training? Do
     other transformer blocks end up reading from it? Does the measured signal
     and false overlap match the prediction? If so, the HDD size is predicted,
     not guessed.
   - Needs 3b: training only n rows is sparse selection.
4. A million × million recurrent transformer block, backprop through a random
   subset of ~1000 rows where the subset is time-related (nearby time entries are
   chosen together, as in e-BPTT's smoothed noise). Backed by the sili database
   in HDD memory. ~1000 rows to keep reads/writes to around page size. both row
   reads/writes and seeking to the next selected row should be sequential.
   The networks should be very sparse, with 1000-10000 synapses,
   to limit to 20-40MB IO per step.

## 5. Port kernels
Port the required kernels and code one by one, each passing the same tests
(Adam is the known non-exact exception). Includes 1a.6 (titans, combined,
omega window on sili CUDA) and the peridot-style sili Tensor port. So that we can:
1. Convert Qwen3.8-27B to fp4 with fp4 + hadamard + additive as close to the
   original weights as possible (and the optimizer state if we can get it,
   otherwise a fake, relatively high optimizer state). Download models onto
   the sandbox (Bee), which has the free space.
2. Get it running by transferring kernels one by one.
3. Train it with intrinsic RL: energy, stochastic annealing.

### The fp4 kernel version to port (checked on sili__new acc7916)
Evidence: `bend_proofs/fp4_trainability` (#58, corrected in #59). Proved =
one-update unbiasedness and its variance; the rest was measured on the real
headers or read from the call sites.

- **SIMD trap, read this before vectorizing for CUDA.** Do NOT port
  `block4_vec_quantize_stochastic_fp4_live` (`block4.hpp`). Despite the name it
  keeps the value's own sign, so under |v| < 0.5 it is deterministic, biased by
  up to 0.5, and never crosses zero (0/1000 runs; the scalar: 1000/1000). Only
  a unit test calls it. Vectorize the two *scalar* functions below, and test
  E[Q(v)] = v and zero crossing on the GPU.
- **Weight:** `fp4_quantize_stochastic_live`: unbiased on [-6, 6], saturates
  outside, rounds across ±0.5 near zero.
- **Importance is a different quantizer:**
  `fp4_quantize_stochastic_live_nonneg`. Never reuse the weight's (or its SIMD
  form) here: the cross-sign rounding makes importance negative and sqrt(ci)
  NaN. Below 0.5 it deliberately returns 0.5 (a floor, so biased there).
- **Empty slot:** block4 uses the plain `fp4_quantize_stochastic` pair when
  weight and importance were both code 0 (`was_live` false), as in
  `Block4Codec<FP4BiPacked>::encode<true>`.
- **Update:** `disldo_backward<..., RMSpropScalePolicy, DeferredScaleWrite=false,
  StochasticRounding=true, BoundedRMSpropSynapsePolicy>`: one batch-aggregated
  update, then quantize. Defaults of the fp4 binding: beta2 .999, eps 1e-8,
  max_abs_delta 2, max_ci 100, zero_escape_eps .1, scale_invariant off,
  max_abs_grad off (1e30: the binding's 8 is never passed for fp4).
- **Effective weight:** W = Q ⊙ S + A, with the rank-N scale S and the additive
  A in fp32. Activations and state are fp32.
- **Nibble order differs, for no recorded reason: use one order in the port.**
  Scattered `FP4BiPacked` = weight high (3cb16b7, 2026-07-01). Block4 = weight
  low (8fb1583, a Claude commit, 2026-07-28, no rationale given). Today's
  conversions all decode and re-encode correctly (read, not tested). Two
  comments state block4's order backwards or ambiguously: `block4.hpp` (the
  Block4Tile8 comment) and `delta_csr_types.rst` ("weight|imp<<4").
- **Not for learning updates:** `fp4_quantize` / `fp4_quantize_live` (nearest).
  The column-rescale pass in `delta_csr_types.hpp` still writes with
  `fp4_quantize_live`, so small rescales are lost or biased: check it before
  porting that pass.

## 6. Energy for RL task solving
Get energy working for RL task solving too. Reading: "Energy-Based Reasoners:
Your Recurrent Reasoner Should Be an Energy-Based Model" (Chen, Gladstone, Li,
Ji, Du; arXiv 2611.08977). Its "energy" is a learned scalar over latent
reasoning states, not sili's neuron energy: see 2d, which may absorb this.

## Finale: train on new video games
Test by training on video games the model hasn't seen: actual games, not
Atari. robonet (general remote-desktop control) or whatever per-game harness is
simplest for one or more games; GameScaling-0.8B already has keyboard/mouse +
vision. Choose the harness when we get here.

## Backlog: integration tests + advancements from other models
- Omega distractor test (row 20), chunk sweep, MIRAS losses, MLP memory, Muon,
  XL cache (A) as a live control.

## Reference: compute
- Router LAN (DHCP, 192.168.1.0/24). Hosts are found by MAC, not IP:
  `tools/lan_hosts.py [name]` reads `~/.config/sili_quartz/lan_hosts` (kept out of
  this public repo), checks the ARP table, sweeps the wired /24 if needed.
  `run_batches.py` resolves `sandbox` and `ai0` this way: 5 GPU batches.
  `tools/lan_up.sh` is only for a direct cable with no router.
- GPUs: laptop 1660 Ti 6 GB; sandbox P40 24 GB + 2060; ai0 GTX 1080 8 GB +
  1660 SUPER 6 GB (driver 580). ai0 is the user's ttrpg machine: files in
  `~/claude_code` only, ask before sudo, GPU jobs only if asked.
- If we scale out, **track watts per job** alongside wall-clock in the batch
  logs, from the start.
- P40 long runs: power cap 200 W (51 °C at 150 W is far below its limit); watch
  the guard csv. The 1660 Ti is the card nearest its abort (75 °C).

---

# Done

<a id="done-1"></a>
## 1. Memory-family tests: B, C, B+C, and why A fails them (PR #2)
Result: **B cannot bind; C and B+C are the path.** Ledger rows 18/19, 29;
`docs/ARCHITECTURES.md` for the taxonomy.
- Built **B** (`models/rmt.py`, RMT register of m memory tokens read/written
  per chunk, gradient via `tbptt_chunks`), **C** (`models/titans.py`,
  `TitansMAC`), **B+C** (`models/combined.py`), **A** (`models/transformer.py`)
  as the control that must fail beyond its reach. One `segment_mask` serves all.
- Shared task: recall with gap; sweep gap and live bindings.
  `examples/capacity_sweep.py` trains once and evaluates longer streams / more pairs.
- Prediction tested: C falls off at O(d_k) bindings (confirmed, C ≈ d_k/4 ≈ 16);
  B at m×d (falsified — B never binds: plateaus at 0.33, a bag of values for any
  m or curriculum).
- Family capacity at 32 bindings: rmt 1.5, titans 19.4, combined 23.7
  (`tests/test_mac.py::test_family_capacity_probe`).

<a id="done-1b"></a>
## 1b. Long-run B+C vs C (PR #3) — combined narrowly ahead; still open (ledger row 30)
- 230-min runs, 32-pair probe: titans 30.2 vs combined 30.7 of 32 (windowed).
  [docs/results/2026-11-15-long-family-runs.md](docs/results/2026-11-15-long-family-runs.md)
- Equal 40k steps, 128-key vocab, train 16 pairs, stress 64 pairs: 30–40k
  window titans 55.4 ± 3.1 (min 43.4) vs combined 56.4 ± 1.5 (min 53.7).
  [docs/results/2026-11-16-equal-steps-64-pair-probe.md](docs/results/2026-11-16-equal-steps-64-pair-probe.md)
- Reading: combined is slightly higher in both experiments and far more stable
  (half the variance, much better worst case). Seeds: see [1a.1](#done-1a1).
- Lesson: compare windowed statistics (mean/sd/min over 10k steps), never single evals.

<a id="done-2"></a>
## 2. ATLAS switches: poly2 keys + Omega window (377db7e)
- `features="poly2"`: keys `[k; upper(kkᵀ)]`. `window=c`: one joint step over
  the chunk's c pairs, per-key normalized, scaled by 1/λ_max of the normalized
  Gram (differentiable). Ledger row 1 re-derived as "window stores exactly when
  keys are independent" before switching; that test doubles as the capacity probe.
- At d_k=64, 32-binding probe: poly2 saturated 32/32; omega alone 28.3 vs base ~19.
- At scale (128-key, 16 pairs) poly2 + omega **fails to train** (JOURNAL row 31,
  train acc 0.74 at 40k) — open as 1a.2.

<a id="done-1a1"></a>
## 1a.1 Seeds 0–3, titans vs combined (PR #3)
[docs/results/2026-11-16-seeds-and-poly2-at-scale.md](docs/results/2026-11-16-seeds-and-poly2-at-scale.md)
- Same program, `--seed` 0–3, 40k steps, 64-pair stress probe; seeds 1–3 three
  processes per GPU (titans P40 ~279 min, combined 2060 ~220 min).
- Seeds that trained (0, 2, 3), 30–40k windowed: titans 54.7, combined 56.3 of
  64; worst eval 43.4 vs 53.7. Combined ≥ titans for every trained seed.
- Seed 1 never trains for either family (combined collapses to a constant
  predictor; titans creeps to 0.07 train acc). titans s3 plateaus until ~10k.
  Spread is *whether/when training escapes the start* → JOURNAL row 32, plan 1a.4.

<a id="done-1a3"></a>
## 1a.3 Ablation tests for the required settings (PR #3)
`tests/test_mac.py::test_ablation_breaks_long_gap_recall` switches one setting off
in the verified long-gap config (gap 16, 3 pairs, 800 steps, seed 0, base 0.074;
full config 1.000) and requires accuracy below base + 0.5, the same line the
positive test must clear:
- previous-token keys off (row 14): 0.383 — partial, still far below 1.000.
- detach every chunk, so no gradient across the gap (row 15): 0.061, chance.
- alpha 0.5, a random-init-like forget gate (row 24): 0.061, chance.
`test_default_gates_are_not_random` (row 24, unit): defaults alpha = eta = 0,
not learnable parameters, identical across seeds.

<a id="done-1a2"></a>
## 1a.2 poly2 at scale — dropped (user, 2026-11-17)
[docs/results/2026-11-17-seven-seeds-split-omega.md](docs/results/2026-11-17-seven-seeds-split-omega.md)
- poly2 + omega stalls at full size (d_feat 2144, 13 GB) and at `poly_rank=16`
  (200 features, 1.06 GB).
- Alone: full poly2 trains 2/2 (2–4k 58.1, 48.2); rank 16 trains 1/2 (59.5, the
  best run, but seed 2 collapses to a constant on a stream every other arm trains).
- Omega alone trains 3/3 (2–4k 51.8 on each), so omega stays.
- Removed: `features`, `poly_rank`, `poly2()`, the poly2 unit/integration tests and
  the `--features/--poly-rank` flags.

<a id="done-1a5"></a>
## 1a.5 40k seeds for the omega arms (2026-11-17)

Titans + omega and combined + omega, seeds 0–6, 40k steps, same spec as the
7-seed runs. Results: [docs/results/2026-11-17-omega-forty-k-seeds.md](docs/results/2026-11-17-omega-forty-k-seeds.md).
- titans + omega: 7/7 train, 30–40k 56.9 ± 2.0 of 64, six seeds at 57.6–58.0.
- combined + omega: 6/7 (seed 5 stuck at acc 0.09), 56.4 ± 2.1, best worst
  case (window min 49.7).
- Plain: titans 55.2 ± 1.8, combined 55.9 ± 1.5, each 6/7.
- Start-up is not reproducible across GPUs: combined + omega s2 stalled on ai0
  (10k run) and trained at 4.5k on the 2060. The 1a.4 fixture needs pinned
  hardware or a failure count over many seeds.
- 1080 peak 77 °C (3 under the abort), 1660 SUPER 70, 2060 63, P40 51.

<a id="done-2b-ebptt"></a>
## 2b. e-BPTT torch first pass (PR #4, 2026-11-17)
[docs/results/2026-11-17-ebptt-credit-sweep.md](docs/results/2026-11-17-ebptt-credit-sweep.md)
`models/ebptt.py`. Overriding a saved input during backward works
(`saved_tensors_hooks`; tested on starmap (no temporal association), 2D slices of smoothed 3D gaussian noise, temporal-only smoothed noise, white noise).
On row 15 with the state detached every chunk: credit from a plain sum of
remembered inputs 0.268, buffer 0.212. With the V-side Jacobian carried
(I − c k kᵀ): trace 0.969, buffer-16 0.891 (pass). W_V-only credit 0.061.
Sparse credit backward over a noise-chosen 1/4 of memory dims matches the
dense credit (9/11 vs 6/8 seeds ≥ 0.95). |G|-chosen stalls on 2 of 3 seeds
and its choice is biased (measured: top 8 dims take 29 % of picks), but
that the bias CAUSES the stall is not shown. The Bend proof
(sili__new bend_proofs/noise_selected_credit) only shows the noise
choice is unbiased; error feedback is the known fix for biased top-k. Per-chunk
backward (`stream_backward`) makes memory flat in T: 22 MiB at T = 28…1036
vs 45…1582 MiB for full BPTT. Row-15 test at constant memory: 0.980.
Cost axis: steps to 0.95 vs full BPTT ×5–10 at gap 16; at gap 64 BPTT 3/3
by 250, e-BPTT 1/3 by 750 and 2/3 stuck ~0.4 (cost grows with gap).
Open: clean in-window test (episode inside one chunk); more steps / seeds
at gap 64; larger d for the wall-time saving.
