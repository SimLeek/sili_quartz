# Track A: required pieces (ledger, started 2026-11-27)

Track A has moved from trying combinations to collecting the pieces a working
setup needs. The sections read as **still to test** (plausible), **results**
(required, keep off) and **won't do** (rejected). Update this file whenever a
result moves a piece, with a link to the result.

**How runs are set up from here (evolutionary):** each Minecraft run uses
every *proven* required piece, respects every keep-off, and adds **one**
plausible piece. If the piece improves the run, it moves to required; if it
degrades it, it moves to keep off or rejected (with where and why).

## Plausible (still to test), roughly by how directly each addresses a known failure

| piece | addresses | status |
|---|---|---|
| **A working drive for the output side.** Candidates: state empowerment through the world model; energy curiosity (homeostatic drive reduction on the input side); KL-Cov/Clip-Cov-guarded rewards. Not candidates: see rejected | the output side has no reason to act once trained: with only an input tether, it drifted and degraded ([Tin3_c](../results/2026-11-27-screen-Tin3_c.md)) | That a drive is needed is shown; which drive works is not. Empowerment's estimator does not yet separate static screens from the open world (base, Tin_c, Tin3_c). Energy curiosity is blocked by the energy problem below |
| Energy that does not silence the most active neurons at equilibrium. Options: column-averaged forward readouts, so a silenced or forced neuron moves a column average by 1/n (user); a per-neuron set point (user: could drift); a decay ratio that never reaches equilibrium within a session | energy in a long-running actor | At equilibrium the site-mean set point shuts off every neuron more than 5 % above the site mean, at any dose ([Eint_c](../results/2026-11-27-screen-Eint_c.md)). That is systematic (all above-mean neurons), so averaging may absorb only part. The current columns are training targets, not forward readouts, so they do not absorb it in the forward pass yet |
| Sleep annealing (sili_peridot): forced-firing energy plus WeightRenorm in a sleep phase, gated on an amortized measure | directional (rank) collapse; LoRA intruder dimensions (Shuttleworth et al. 2025) | Spectrum check running on all checkpoints. First four checkpoints: 0 intruders, weight change 0.6–0.9 %, effective rank of the change 200–290. The fix is about as easy as the measurement (user) |
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
| `tools/thought_influence.py` | whether thoughts change actions |
| key repertoire per 10 min | distinct keys used in play, from the actor logs |
