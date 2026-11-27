# Track A: evolution — methodology, results and plan (started 2026-11-27)

The methodology and results for track A, continuing the
[introduction](1-introduction.md) and the original
[methodology draft](2-methodology.md). Track A moved from trying
combinations to evolving a lineage: each run adds one piece to a proven base.
The tree below is the record of *why* each piece counts as required, kept
off, or rejected.

Reading order:

- core requirements;
- the method;
- the tree (results);
- the run plan and its options;
- the pieces, as **still to test** (plausible), **results** (required, keep
  off) and **won't do** (rejected);
- the measurements.

Update this file whenever a result moves a piece: link the result doc and add
a row to the tree.

## Core requirements (user, 2026-11-27)

Two things the learning itself must provide. Everything else (health
metrics, renorms, sleep, guards) is an **improvement**. An improvement joins
the required set only if it is shown to help in practically all cases.

1. **Coherent token output.** The model keeps (and, from scratch, learns) the
   output tokens it can use: the action keys, not a collapse onto one key or a
   spread over all 74.
   - **Check (not built):** the acceptance test for whichever token-memory
     method is chosen (token energy, exclusive fan-in consolidation,
     Clip-Cov/KL-Cov, ...). It is the same method we would rely on to *keep*
     the tokens under RL, applied from a random start:
     - **Start:** re-initialize the 74 key tokens' rows of the tied
       embedding/output head (random, matched norm). The grammar mask still
       keeps every action well-formed; only which keys to use is lost.
     - **Stage A, the method alone, no reward** (offline, on replay frames):
       sample actions, apply the method's updates. Pass if the sampled actions
       come to use the key set at all (distinct keys and key entropy back in
       the base model's range) instead of staying on a few keys. This tests
       reachability.
     - **Stage B, the method plus the candidate drive** (live): pass if the
       keys used in play return to the base range *and* depend on the screen
       (different keys in different states). This tests that the tokens are
       learned as usable, not just reachable.
     - Run the same test on the base model's own rows as the control: the
       method must not erase a repertoire that is already there.
   - **Measured meanwhile:** key repertoire in play, and
     `tools/repertoire_probe.py` against the base model's 11–25 keys per
     10 minutes.
2. **Coherent, input-driven action choice.** The output side has a drive that
   makes its actions depend coherently on the world input (any input-coherent
   RL).
   - **Most promising candidate:** energy curiosity, unblocked by
     column-averaged energy (below).

## Method: evolution, recorded as a phylogenetic tree

- **Rule:** each run uses every proven required piece, respects every keep
  off, and adds **one** plausible piece (a trait).
- **Root:** the tree starts from a subset of the required pieces in
  combination, to confirm each is actually required (drop one, check it
  breaks).
- **Advancing:** a run that does well on the metrics (repertoire, input
  coherence, world-model probe, game stats) advances: its children each try a
  further trait. One that degrades marks that trait keep off (where) or
  rejected.
- **Recording:** the tree is a table. The node name is the chain of
  additions; depth is shown with leading dashes; rows are in tree order. A
  child row sits under its parent.

### The tree (results)

The runs before this ledger were not planned as a tree; they are placed as
branches from the base model.

| node | run(s) | result | status |
|---|---|---|---|
| base + grammar | A-0 baselines (no grammar), grammar smoke | parses 100 %; 11–25 keys per 10 min | root |
| - +A1 MI reward | A1, A1c | narrowed to loops, then static screens | rejected |
| - +A1-act MI reward | A1-act | pause menu for 85 min | rejected |
| - +A2 reward | A2c | one key (hold LB) from minute 30 | rejected (as built) |
| -- +one-sided entropy guard | A2g | ran away to random | keep off |
| -- +two-sided entropy guard | A2b | oscillated between bounds | keep off |
| -- +KL-Cov (rolling 0.2 %) | A2_klcov | ~0.2 tokens per step; still narrowed (confounded) | too weak as configured |
| - +energy, site-mean set point, dose 1 | Eint_c (no training) | random within 5 min | keep off |
| - +input tether, all 6 chunks | Tin_c | policy a by-product of prediction; death screen | keep off (output-side chunks) |
| - +input tether, layers 1–12 | Tin3_c | no random output; fewer clicks, 9 deaths | plausible (needs a drive) |
| -- +energy (input and interior), site mean | Tin3_e | random once at dose 1 | keep off |
| --- +energy curiosity reward | Tin3_curio | drive read 0 (trainer energy from rest) | stopped; retry after column energy |
| - +energy, column-averaged (8), site mean, dose 1 | Ecol8_c (no training) | still mashing (8.3 distinct keys per decision, mouse 41) | keep off: play-time energy on an unadapted model (user: always gibberish) |
| - (root, reference) | Root_g: untouched + grammar, no energy, no training | running | scorecard reference for grammar runs |
| -- +input tether layers 1–12, +energy in training only (fixed:33, actor clean) | Tin3_et | queued | |
| --- +energy-curiosity reward | Tin3_et_curio | queued | |
| - +input and output tether, norm band | FULL_c | output random from minute 10 | keep off (output tether without a drive) |
| -- +energy on the output neurons | FULLe_c | random from the start | keep off |
| - (format bug, pre-grammar) | A1pa, A1io | broken completions, no gradient | led to grammar (required) |

## Run plan

### Queued now (on the sandbox)

| order | node | adds (one trait) | what it tests |
|---|---|---|---|
| 1 | Root_g (running) | nothing: untouched model + grammar, no energy, no training, 1 h | the scorecard reference for every grammar run |
| 2 | smoke test, then Tin3_et | energy in training only (fixed:33, the text runs' dose; actor plays clean), on the Tin3 node | whether training under energy hurts or helps play when the actor itself plays clean |
| 3 | Tin3_et_curio | the energy-curiosity reward on the output side | the first candidate drive for core requirement 2 |

After that the queue ends and the sandbox returns to desktop mode.

### How a node is judged

- **Scorecard:** `tools/run_scorecard.py` against Root_g: healthy windows,
  minute of the first flag (loops, collapsed or spread keys, mashing), game
  rates.
- **Checkpoint probes:** repertoire, and the active fan-in (no damage pattern:
  drive and spread falling while overlap rises); world-model probe.
- **Advance:** no worse than the parent on these, and better on at least one
  core requirement. Examples: keys that depend on the screen; a repertoire in
  the base range.
- **Otherwise:** the trait goes to keep off (with where) or rejected.

### Next traits, by outcome

- **If Tin3_et_curio advances**, its children each try one of:
  - a lower learning rate, or several windows per step;
  - play-time energy bursts on loop flags;
  - dynamic depth connections at chunk boundaries;
  - short trained thoughts (length as the knob);
  - the output tether at low weight;
  - Clip-Cov/KL-Cov with a larger fraction.
- **If it fails:** Tin3_et alone shows whether training-time energy is the
  problem. Alternative branches:
  - the curiosity reward without the tether;
  - state empowerment, once the world-model probe improves.

### Token memory from random (core requirement 1): not started

An offline track; it does not need the game.

1. **Scramble:** built (`tools/scramble_keys.py`). It re-initializes the
   74 key rows of the tied embedding/output head (random direction, each
   row's own norm kept). **The no-method control, measured:**
   - **Keys:** the scrambled model presses no key at all (0 distinct keys in
     120 sampled actions, against 13 for the untouched model).
   - **Actions:** 41 of the 120 are the pure no-op; mouse moves survive.
   - **Active fan-in:** drive toward the emitted keys falls from 18.6 to
     4.0, spread over ~15,900 neurons (untouched ~4,300), overlap 0.08.
   - **Pass mark for a method:** the key repertoire and k80 back in the
     untouched model's range (about 6 keys covering 80 % of ~30 used).
2. **Stage A harness:** on replay frames, sample with the grammar, apply the
   method's updates, and measure the repertoire and the active fan-in every
   N steps.
   - **Control, no method:** expected to stay stuck.
   - **Control, the method on the base model's own rows:** it must not erase
     an existing repertoire.
3. **Methods, one node each:**
   - exclusive fan-in consolidation (a weight update by itself);
   - token energy (virtual neurons; it changes sampling only, so it needs a
     drive's gradient and belongs in stage B);
   - Clip-Cov/KL-Cov with a larger fraction.
   - **Settled (user): no target usage distribution.** Forcing uniform usage
     would override the model's choices. The method keeps each key
     *reachable*: an exclusive set of driving neurons with enough capacity
     that the policy can pick the key. How often it is picked stays the
     policy's choice.
     - From random init, every key starts equal, so usage comes out uniform.
     - From a trained model, the existing usage statistics are kept.
     - Fan-in consolidation fits this; a uniform-usage target does not.
   - **Measured shapes (untouched model):** per-key *capacity* is nearly
     uniform, while *usage* is heavy-tailed. Usage skew comes from context and
     policy, not from capacity.
     - Capacity: static fan-in strength log-σ 0.08 (min 0.83× the median);
       active drive log-σ 0.18.
     - Usage: roughly log-normal, log-σ 2.2–2.4, skew −0.02 to +0.64. With σ
       that large the log-normal mode is under one press, so the mode cannot
       serve as a floor (the user's first suggestion).
     - Floor for consolidation (proposed): a key is fading when its capacity
       falls below the untouched model's weakest-key ratio, 0.83× the current
       median capacity. A measured anchor.
     - Usage guard: k80, the number of keys covering 80 % of usage in a fixed
       30-minute window. It is about 6 (of ~30 used) for the untouched model,
       i.e. 80-20 (user). A2c's collapse went to 1 (99-1); energy's mashing to
       31 of 74. Added to the scorecard.
     - Research: no study found fitting per-key usage in games (closest:
       W-A-S-D concentration, heavy-tailed player behaviour, log-normal
       achievement counts and action timing).
4. **Stage B:** the winning method plus the advancing drive, live.

## Plausible (still to test), roughly by how directly each addresses a known failure

| piece | addresses | status |
|---|---|---|
| **A working drive for the output side.** Candidates: state empowerment through the world model; energy curiosity (homeostatic drive reduction on the input side); KL-Cov/Clip-Cov-guarded rewards. Not candidates: see rejected | the output side has no reason to act once trained: with only an input tether, it drifted and degraded ([Tin3_c](../results/2026-11-27-screen-Tin3_c.md)) | That a drive is needed is shown; which drive works is not. Empowerment's estimator does not yet separate static screens from the open world (base, Tin_c, Tin3_c). Energy curiosity is blocked by the energy problem below |
| Energy at play time as designed: only to break attractors (bursts triggered by the scorecard's loop or collapsed-key flags, shorter than its time to equilibrium, then reset) or as occasional short bursts for rare states. Energy was built to heal zero-init and pathological attractors during training, not to run constantly at play time (user) | play-time loops (A2c, A1) are pathological attractors in that sense | Idea |
| Energy in training only (weights train under it, actor plays clean), at the text runs' dose (fixed:33) | energy's designed role; the constraint that set track A's 0.03× dose came from play-time energy on an unadapted model | Queued: Tin3_et, then Tin3_et_curio (smoke-gated) |
| Energy that does not silence the most active neurons at equilibrium. Options: column-averaged forward readouts, so a silenced or forced neuron moves a column average by 1/n (user); a per-neuron set point (user: could drift); a decay ratio that never reaches equilibrium within a session | energy in a long-running actor | At equilibrium the site-mean set point shuts off every neuron more than 5 % above the site mean, at any dose ([Eint_c](../results/2026-11-27-screen-Eint_c.md)). That is systematic (all above-mean neurons), so averaging may absorb only part. The current columns are training targets, not forward readouts, so they do not absorb it in the forward pass yet |
| Dynamic depth connections at chunk boundaries (MUDDFormer-style, Xiao et al. 2025): each chunk's input is the residual plus a learned, per-position selection over the previous chunk's layer outputs (or all earlier ones), initialized to add nothing | passing only a column's average forward would pass only the next-frame prediction, while internal neurons may carry longer-horizon information (user). The cortex sends L2/3 prediction errors forward into the next area's L4 and L5/6 predictions back (Bastos et al. 2012); selection from many columns stands in for a sparse connectome | Idea (user). MUDDFormer reports matching 1.8–2.4× compute (self-reported). Distinct from column-averaged energy, which averages only energy's perturbation, not the signal |
| Sleep annealing (sili_peridot): forced-firing energy plus WeightRenorm in a sleep phase, gated on an amortized measure | directional (rank) collapse; LoRA intruder dimensions (Shuttleworth et al. 2025) | No intruders and no low-rank concentration in any 2-hour checkpoint: the change's effective rank grows to 290–370 ([checkpoint probes](../results/2026-11-27-checkpoint-probes.md)). Not needed for this failure at this length; keep for longer runs and generations |
| Clip-Cov / KL-Cov (Cui et al. 2025) | entropy collapse under RL (A2c went to one key) | KL-Cov with the paper's 0.2 % fraction held over a rolling window selected ~0.2 tokens per step and did not stop A2 from narrowing (A2_klcov, stopped at minute 35; confounded by a slower trainer). As configured, too weak for our ~125 tokens per step. Clip-Cov (a tenth as many tokens) was dropped untested |
| Lower learning rate or more tokens per step | fast narrowing | We take ~2,000 tokens per step against GameScaling's 2.36 M, at a higher learning rate; one 2-hour run is ~10 % of their training steps |
| Input tether (depth columns) on the input side, layers 1–12 | grounding the state; world model for empowerment | Keeps the output from going random (unlike the output tether), but the state change still reaches the middle layers ([Tin3_c](../results/2026-11-27-screen-Tin3_c.md)). Not shown to help on its own |
| WeightRenorm toward the starting model's per-row statistics | magnitude drift | The 0.8–1.25 band is in place as a stand-in (see keep off) |
| Short RL-trained thoughts (5–8+ tokens), with length as a knob to sweep | persistent intent; grounded thoughts | The base model's thoughts move actions by up to ~4 nats but often misdescribe the screen ([thought influence](../results/2026-11-27-thought-influence-base.md)) |
| Energy on tokens as virtual neurons | output repertoire | Idea (user); has the same set-point question as neuron energy |
| Exclusive fan-in consolidation in sleep | output repertoire; token reachability | Idea (user). Static fan-in is unchanged in every checkpoint (≤ 0.13 %), so it would act on activations |
| Neuron diversity (lateral inhibition, Gaussian max-kernel inhibition, variational) | the couch-potato failure of curiosity | Idea (user) |
| PR guard as a loss | representation collapse | In peridot, PR did not tell stuck runs from healthy ones; probably better as a measurement |

## Required (results)

| piece | applies to | evidence |
|---|---|---|
| Grammar-constrained decoding, with the same mask in the trainer's log π and PPO ratio | action output | Without it, broken completions carried no gradient and became absorbing (A1pa, A1io: 70 % unparsed, 0 PPO turns). With it, every decision parses. [A1pa/A1io](../results/2026-11-27-screen-A1pa-A1io.md) |
| Finite masked fill in the masked log-softmax | trainer | −inf gave NaN gradients on the first update (grammar smoke run) |
| The trainer replays energy from the actor's saved state | every energy arm | From rest, the trainer's energy barely acted while the actor's had accumulated: PPO ratios compared different forward passes, and the curiosity drive read 0 ([Tin3_e](../results/2026-11-27-screen-Tin3_e.md)) |

## Keep off (results)

| piece | keep it off | evidence |
|---|---|---|
| Energy with the site-mean set point at a fixed dose | any long-running actor (until the energy row above is solved) | Actions were random within minutes, even with no training. With energy off, the same run's weights were healthy (repertoire probe). [Eint_c](../results/2026-11-27-screen-Eint_c.md), [Tin3_e](../results/2026-11-27-screen-Tin3_e.md) |
| Energy | the action-choice (output) neurons | Random from the first minutes ([FULLe_c](../results/2026-11-27-screen-FULLe_c-stopped.md)) |
| Input tether | output-side chunks (tether only `--tether-chunks 3`, layers 1–12) | Tethering all six made the policy a by-product of input prediction: confident, no clicks, stuck on its death screen ([Tin_c](../results/2026-11-27-screen-Tin_c-stopped.md)) |
| Output tether | the output side, unless a policy drives it; then only at low weight | It regressed the last chunk onto the model's own samples and flattened the choice ([FULL_c](../results/2026-11-27-screen-FULL_c-stopped.md)) |
| Policy rewards that pay for predictability | policy term | MI rewards narrow to static screens and loops: the dark room (A1, A1-act, A1c) |
| Policy rewards that contain the policy's own randomness | policy term | A2's reward contains H(A\|S): mashing with a guard (A2g), collapse to one key without (A2c) |
| Direct entropy penalty or guard | everywhere | The right entropy depends on the moment; one-sided ran away, two-sided oscillated ([A2g](../results/2026-11-27-screen-A2g-stopped.md), [A2b](../results/2026-11-27-screen-A2b-stopped.md)) |
| KL to a frozen copy of the untouched policy | everywhere | An oracle: it is the behaviour-cloning target itself (user) |
| Arbitrary bands (0.8–1.25 × early value) | guards and renorms | Replace with measured anchors or derived quantities (user) |

## Rejected (won't do, in this role)

| piece | role it failed in |
|---|---|
| Predictive information of the stream (A1, A1-act) | policy reward (dark room). Kept as a measurement |
| Policy-dependent I(A; S') (A2 as built) | policy reward |
| Action-entropy guard | any role |
| MI representation term with both ends learned | world model: internal slow variables can raise it |

## Measurements in place

| tool | measures |
|---|---|
| `tools/world_model_probe.py` | how well each checkpoint's state predicts the untouched model's next-frame input |
| `tools/empowerment_check.py` | the state-empowerment estimate on static screens against the open world |
| `tools/policy_drift.py` | base model against a checkpoint on the same frames: actions and per-layer state cosine |
| `tools/repertoire_probe.py` | sampled-action repertoire per checkpoint with energy off: keys, entropy, LB share |
| `tools/fanin_probe.py` / `tools/fanin_active.py` | static and activation-weighted fan-in per action key: strength or drive, spread, overlap |
| `tools/spectrum_probe.py` | weight-change rank, top-8 share, intruder dimensions |
| `tools/run_scorecard.py` | one scorecard per run: per-window repertoire, distinct keys per decision, mouse, certainty, flags against the reference, game rates |
| `tools/thought_influence.py` | whether thoughts change actions |
| key repertoire per 10 min | distinct keys used in play, from the actor logs |
