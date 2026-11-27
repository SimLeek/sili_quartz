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
| 1 | Root_g (done) | nothing: untouched model + grammar, no energy, no training, 1 h | the scorecard reference: k80 4–7, distinct ratio ~0.8, 0.6–0.7 distinct keys per decision, mouse 11–23; mined 154 per hour of play |
| 2 | smoke test, then Tin3_et | energy in training only (fixed:33, the text runs' dose; actor plays clean), on the Tin3 node | whether training under energy hurts or helps play when the actor itself plays clean |
| 3 | Tin3_et_curio (commit 0476832) | the energy-curiosity reward on the output side | the first candidate drive for core requirement 2. The smoke run showed the drive building up with frame position (energy starts at rest each window: 0.025 → 0.094), so each position's running mean drive reduction is subtracted |

Then queue 24 runs unattended: the **drive generation**, as siblings.

All siblings share one parent setup (input tether on layers 1–12,
training-time energy at fixed:33, actor clean) and differ only in the drive.

| sibling | drive | kind |
|---|---|---|
| Tin3_et_curio (queue 23) | energy curiosity, layers 1–12 | energy family |
| curio_input | energy curiosity, input neurons only (pixel-level; roughly episodic novelty) | energy family |
| curio_all | energy curiosity, all 24 layers (concept-level) | energy family |
| episodic_int | episodic novelty on layer-12 frame representations: mean distance to the 10 nearest *earlier* frames in a rolling 2,048-frame memory (a poor estimate until the hippocampus, TODO 4) | different estimator (joint) |
| rnd_int | random network distillation on the same representations | different estimator (generalizing) |

1. **Smoke tests.** Each new drive gets a trainer-only smoke run; one that
   fails is skipped.
2. **Evaluate** every sibling (`~/runs/eval_node.sh`).
   - **Combination siblings** (user: the estimators are not mutually
     exclusive): built from the singles that pass the health gate
     (`~/runs/plan_combos.py`).
     - The pool is {the best healthy energy setting, episodic, RND}. Energy
       settings cannot combine with each other (one energy module).
     - Combinations: every pair, plus all three when all are healthy.
     - Rewards are combined without weights (`RewardMix`): each drive
       standardized by its own running statistics, then summed. Equal weight
       is the knob-free start; if a combination wins, its weights become a
       trait for its children.
     - Each combination is smoke-tested, run, and evaluated.
3. **Select** across singles *and* combinations (`~/runs/select_sibling.py`). Health is absolute, against the
   untouched model:
   - at least half the scorecard windows healthy;
   - repertoire: distinct keys ≥ 0.7× 13, LB share ≤ 0.6;
   - no active fan-in damage.
   - Among the healthy, the highest **input coherence** wins (key-set
     similarity within a state minus across states, from the repertoire
     probe); ties go to the smaller median deviation.
4. **The winner's children**, one trait each: learning rate 1e-5; output
   tether at 0.1; KL-Cov at k = 0.02. Each is decided against the winner.
5. **Desktop mode.**

The decisions are logged as `DECIDE` lines in `~/runs/arms_queue3.log`; the
tree and the pieces here are updated from them.

### How a node is judged

- **Scorecard:** `tools/run_scorecard.py` against Root_g.
  - **Median deviation:** per window, the mean relative distance from Root_g
    over the six repertoire metrics. It orders the runs: Root_g 0.03, A1c
    0.24, Tin3_c 0.32, A2_klcov 0.38, Tin_c 0.41, FULL_c 0.73, A2c 0.80,
    play-time energy 10–12.
  - **Severe flags:** a metric at least 50 % off in the failing direction.
    Min/max of the 1-hour reference flagged everything.
  - **Game rates.**
- **Checkpoint probes:** repertoire, and the active fan-in (no damage pattern:
  drive and spread falling while overlap rises); world-model probe.
- **Advance:** no worse than the parent on these, and better on at least one
  core requirement. Examples: keys that depend on the screen; a repertoire in
  the base range.
- **Otherwise:** the trait goes to keep off (with where) or rejected.

### How a result counts (user, 2026-11-27)

- **Better, beyond "not broken":**
  - capability in the game: per hour of play, blocks mined, items picked up,
    distance, deaths; plus distinct block types, crafting, and tutorial
    progress;
  - **learning within the run:** the last third of the windows against the
    first third, and the second half's game rates against the first half's.
    The first minutes are the base model, so each run is its own control.
    These trends are read against the untouched runs' own trends; even Root_g
    mines less in its second half (the trees near spawn are used up first).
  - health is a constraint, not the goal.
  - **The dark-room signal** (user): a policy should not choose the pause menu
    and stay there.
    - **Pauses per hour:** from the game log's "Saving and pausing game"
      lines. Earlier sessions logged 66–247 pauses each.
    - **Esc resumes:** when the screen stays frozen for 3 minutes on a pause,
      the watcher presses Esc once and counts it, so returning to the pause
      menu shows as a rising count.
    - **Ended by hang:** a frozen screen that is not a pause is a hang; the
      watcher kills the game and ends the run (it stops the actor), recorded
      as the run's ending reason.
    - Before this change a parked pause was treated as a hang and the game was
      killed and relaunched mid-run: Tin3_et_curio lost about 5–8 minutes at
      22:29–22:37 UTC (the first relaunch timed out after 240 s) while the
      actor kept acting on the desktop.
- **Noise floor:**
  - three untouched runs (Root_g, Root_g2, Root_g3) give the run-to-run spread
    of every play metric;
  - three sampling seeds of the repertoire probe on fixed weights give the
    probe's own sd.
  - A difference inside the floor is not evidence: if the winner's lead over
    the runner-up is inside it (coherence lead under 2× the probe sd, or
    deviation inside the untouched spread), both are rerun with a second
    training seed and the averages decide.
- **Method or setting?**
  - Is the drive's signal alive (reward variance non-zero and moving with
    events)?
  - Did the policy respond (log-probs shift, PPO turns matched)?
  - Is the result robust to its knob? The winner is swept at learning rate
    and dose ¼×, 1×, 4×, and the **whole spread** is reported: every point,
    best (optimality), worst (robustness), range, and whether the best sits
    at an edge (then the sweep should extend). If one value is best on every
    metric and stays best in other contexts, it is a setpoint, not a knob
    (user).
  - A dead signal is a bug or a setting. A live signal with worse behaviour
    is the objective.

Scripts: `tools/lineage/` (copies of the sandbox orchestration).

### Random-init checks (queue 25, after the drive generation)

The base model is behaviour-cloned and near optimal, so a method can look good
by not breaking it (user). Fully random init of the 0.8B vision-language model
cannot learn vision in 2-hour runs. Two middle grounds instead:

1. **Random policy on pretrained perception:** the last chunk (layers 21–24)
   re-initialized (`tools/reinit_layers.py`) while vision and the early layers
   keep their weights. The drive must build a policy from scratch. Tests core
   requirement 2.
2. **Scrambled keys** (`tools/scramble_keys.py`). Tests core requirement 1.

Each start gets a 1-hour **no-training control** (the same random start,
no updates). The generation's winner and runner-up are then trained from it
for 2 hours (`--init` on actor and trainer). A method passes if it improves
*from the random start*: scorecard trend, coherence, keys and k80 moving toward
the untouched model's range, against the control
(`~/runs/randinit_report.py`).

Caveat: the trainer uses rank-8 LoRA merged every 8 steps. The change does
accumulate to high rank (spectrum check), but learning a whole chunk this way
is slow, so a null result may mean "too slow in 2 hours", not "cannot". The
rate of improvement is logged, not only pass or fail.

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
3. **First result, fan-in consolidation v1** (`tools/fanin_consolidate.py`,
   4 rounds, floor 0.71× median):
   - **Untouched model (control):** repertoire kept (15 keys, entropy 2.73,
     LB share 0.375; base 13, 2.46, 0.38). Passes: consolidation does not erase
     a repertoire.
   - **Scrambled keys:** capacity rose to the untouched model's level (median
     0.60 → 0.94, keys below the floor 25 → 6), but the model still almost
     never presses a key (2 presses in 120 actions; 0 before).
   - **Reading:** exclusive sets carry each key's *identity*, not the shared
     "press a key now" pathway. At a key slot the choice is first *a key
     versus a separator or space*. The untouched model's key rows share a
     component that many neurons push together; scrambling (or random init)
     removes it.
   - **Not a shared class pathway (user):** neurons that push all keys
     together fail mutual exclusivity one layer back. The fix keeps strict
     exclusivity but defines it against **every token that competes at the
     slot**: the other keys *and* the space, separator and end-tag tokens. v1
     only used the other keys, so nothing made a key beat the separators.
   - **v2** (`--competitors vocab`): exclusive drive against all
     action-vocabulary rivals, and updates with every rival's direction
     projected out.
     - **Control:** repertoire kept (14 keys, LB share 0.43).
     - **Scrambled:** capacity did not rise (median 0.55 → 0.55, 23 keys
       below the floor), and still 2 key presses in 120 actions.
   - **Shared flaw of v1 and v2:** they only rotate write columns, never what
     makes a neuron fire. Push is activation × path, and the chosen neurons
     are mostly not active at key slots.
     - **Next version:** for each key, choose unowned neurons that are active
       at *key slots* (positions where the grammar allows a key), then make
       their paths exclusive to it. Label-free: it uses slot positions, not
       which key was pressed.
     - **Also:** scrambling breaks reading (the rows are tied), so the model
       cannot recognize its own past keys in the history either. Random init
       has the same problem; rebuilding the output side alone may not be
       enough.
4. **Methods, one node each:**
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
       - The floor is on *capacity* (weights, through the key's direct
         fan-in), not on presses: it has no time window and never makes the
         policy press a key. Keys that do nothing in the current situation
         may stay very rare (user: some keys go a whole day unpressed).
       - Forced use of rare keys (to check they still come out and to learn
         associations) is a separate exploration trait. If added: a small
         per-key budget, never during a committed sequence (holding LB for
         the 60–100 actions that break a wood block), only between holds;
         or done offline on replay frames as a reachability test (can the
         key's logit be raised) with no game time at all.
       - Its natural form (user): token energy with a very long time
         constant. A key's energy builds very slowly while it goes unused
         and forces one press after hours or days of disuse, then resets,
         like a denervated neuron's spontaneous twitching.
     - Usage guard: k80, the number of keys covering 80 % of usage in a fixed
       30-minute window. It is about 6 (of ~30 used) for the untouched model,
       i.e. 80-20 (user). A2c's collapse went to 1 (99-1); energy's mashing to
       31 of 74. Added to the scorecard.
     - Research: no study found fitting per-key usage in games (closest:
       W-A-S-D concentration, heavy-tailed player behaviour, log-normal
       achievement counts and action timing).
5. **Stage B:** the winning method plus the advancing drive, live.

## Plausible (still to test), roughly by how directly each addresses a known failure

| piece | addresses | status |
|---|---|---|
| **A working drive for the output side.** Candidates: state empowerment through the world model; energy curiosity (homeostatic drive reduction on the input side); KL-Cov/Clip-Cov-guarded rewards. Not candidates: see rejected | the output side has no reason to act once trained: with only an input tether, it drifted and degraded ([Tin3_c](../results/2026-11-27-screen-Tin3_c.md)) | That a drive is needed is shown; which drive works is not. Empowerment's estimator does not yet separate static screens from the open world (base, Tin_c, Tin3_c). Energy curiosity is blocked by the energy problem below |
| Energy at play time as designed: only to break attractors (bursts triggered by the scorecard's loop or collapsed-key flags, shorter than its time to equilibrium, then reset) or as occasional short bursts for rare states. Energy was built to heal zero-init and pathological attractors during training, not to run constantly at play time (user) | play-time loops (A2c, A1) are pathological attractors in that sense | Idea |
| Energy in training only (weights train under it, actor plays clean), at the text runs' dose (fixed:33) | energy's designed role; the constraint that set track A's 0.03× dose came from play-time energy on an unadapted model | Queued: Tin3_et, then Tin3_et_curio (smoke-gated) |
| Energy that does not silence the most active neurons at equilibrium. Options: column-averaged forward readouts, so a silenced or forced neuron moves a column average by 1/n (user); a per-neuron set point (user: could drift); a decay ratio that never reaches equilibrium within a session | energy in a long-running actor | At equilibrium the site-mean set point shuts off every neuron more than 5 % above the site mean, at any dose ([Eint_c](../results/2026-11-27-screen-Eint_c.md)). That is systematic (all above-mean neurons), so averaging may absorb only part. The current columns are training targets, not forward readouts, so they do not absorb it in the forward pass yet |
| Drives: energy curiosity by site (input neurons, layers 1–12, all layers) as settings of one drive; episodic novelty on internal features and RND on internal features as outside alternatives | core requirement 2 (a drive) | Math checked in Bend (`bend_proofs/energy_vs_episodic_novelty`, 10 laws, plus the real kernel). Per-neuron energy is factorized habituation: it cannot see joint novelty, and no per-neuron moment of any order fixes that; conjunctive units or a memory of whole states can. RND's distinct part is inductive bias. So: one energy module configured by site (and conjunctive units if needed); episodic memory goes with TODO 4 (hippocampus); RND only as a check |
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
