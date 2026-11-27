# Track A: required pieces (ledger, started 2026-11-27)

Track A has moved from trying combinations to collecting the pieces a working
setup needs. Each piece has a status, the part of the model it applies to (or
must stay off), and the evidence. Update this file whenever a result moves a
piece.

Statuses:

- **Required:** evidence shows the setup breaks without it.
- **Keep off:** evidence shows it breaks the setup on the named part.
- **Plausible:** a reason or partial evidence, not yet shown on this setup.
- **Measurement:** a probe or meter, not a training signal.
- **Rejected:** shown not to work in this role.

## Required

| piece | applies to | evidence |
|---|---|---|
| Grammar-constrained decoding, with the same mask in the trainer's log π and PPO ratio | action output | Without it, broken completions carried no gradient and became absorbing (A1pa, A1io: 70 % unparsed, 0 PPO turns). With it, every decision parses. [A1pa/A1io](../results/2026-11-27-screen-A1pa-A1io.md) |
| Finite masked fill in the masked log-softmax (not −inf) | trainer | −inf gave NaN gradients on the first update (grammar smoke run) |
| Trainer replays energy from the actor's saved state, not from rest | every energy arm | From rest, the trainer's energy barely acted while the actor's had accumulated. PPO ratios compared different forward passes, and the curiosity drive read 0. [Tin3_e](../results/2026-11-27-screen-Tin3_e.md) |
| A per-neuron (or per-unit) energy set point, instead of the site mean | every energy site in a long-running actor | At equilibrium the site-mean set point shuts off every neuron more than 5 % above the site mean, at any dose, and actions go random. Shown with no training at all. [Eint_c](../results/2026-11-27-screen-Eint_c.md), [Tin3_e](../results/2026-11-27-screen-Tin3_e.md). **Decision pending (user).** |
| A drive for the output side (a policy term), separate from the representation losses | output-side layers | With only an input tether, the change in layers 1–12 carried into layers 13–16 and the untrained output side lost clicks (0.27 → 0.10) and died 9 times. [Tin3_c](../results/2026-11-27-screen-Tin3_c.md) |
| The policy reward must not pay for predictability or for the policy's own randomness | policy term | MI rewards (A1, A1-act, A1c) narrow to static screens and loops: the dark room. A2's reward contains H(A\|S). [objective math](track-a-objective-math.md) |

## Keep off

| piece | keep it off | evidence |
|---|---|---|
| Input tether (depth columns) | output-side chunks. Tether only the input side (`--tether-chunks 3`, layers 1–12) | Tethering all six chunks made the policy a by-product of input prediction: confident, no clicks, stuck on its death screen. [Tin_c](../results/2026-11-27-screen-Tin_c-stopped.md) |
| Output tether | the output side, unless a policy term drives it; then at low weight (`--w-tether-out`) | Regressing the last chunk onto the model's own samples flattened the choice (entropy 0.45 → 1.9). [FULL_c](../results/2026-11-27-screen-FULL_c-stopped.md) |
| Energy | the action-choice (output) neurons, until per-token set points exist | Actions were random from the first minutes. [FULLe_c](../results/2026-11-27-screen-FULLe_c-stopped.md) |
| Direct entropy penalty or guard | everywhere | The right entropy depends on the moment. One-sided, it ran away (A2g); two-sided, it oscillated (A2b). [A2g](../results/2026-11-27-screen-A2g-stopped.md), [A2b](../results/2026-11-27-screen-A2b-stopped.md) |
| KL to a frozen copy of the untouched policy | everywhere | An oracle: it is the behaviour-cloning target itself (user) |
| Arbitrary bands (0.8–1.25 × early value) | guards and renorms | Replace with measured anchors (the starting model's own statistics) or derived quantities (user) |

## Plausible (in order of how directly they address a known failure)

| piece | addresses | status |
|---|---|---|
| Clip-Cov / KL-Cov (Cui et al. 2025), the paper's token fraction held over a rolling window | entropy collapse under RL: A2c went to one key | Built (`--cov`); smoke test, then A2 arms |
| Sleep annealing (sili_peridot): forced-firing energy plus WeightRenorm in a sleep phase, gated on an amortized measure | directional (rank) collapse; LoRA intruder dimensions (Shuttleworth et al. 2025) | Spectrum check running on all checkpoints; the fix is about as easy as the measurement (user) |
| WeightRenorm toward the starting model's per-row statistics | magnitude drift | The 0.8–1.25 band is in place as a stand-in |
| Lower learning rate or more tokens per step | fast narrowing | We take ~2,000 tokens per step against GameScaling's 2.36 M, at a higher learning rate; one 2-hour run is ~10 % of their training steps |
| Energy curiosity reward (homeostatic drive reduction on the input side) | world model; intermediate novelty without a set-point constant | Built (`--reward energy`); needs the per-neuron set point first |
| State empowerment through the world model | effective action, with no dark room | Built (`--reward empower`); the estimator does not yet separate static screens from the open world (base, Tin_c, Tin3_c). Needs a better world model first |
| Short RL-trained thoughts (5–8+ tokens), with length as a knob to sweep | persistent intent; grounding thoughts | Base model's thoughts move actions by up to ~4 nats but are often wrong about the screen. [thought influence](../results/2026-11-27-thought-influence-base.md) |
| Energy on tokens as virtual neurons (per-token set point) | output repertoire | Idea (user) |
| Exclusive fan-in consolidation in sleep | output repertoire; token reachability | Idea (user). Static fan-in is unchanged in every checkpoint (≤ 0.13 %), so it would act on activations |
| Neuron diversity (lateral inhibition, Gaussian max-kernel inhibition, variational) | couch-potato failure of curiosity | Idea (user) |
| PR guard as a loss | representation collapse | In peridot, PR did not tell stuck runs from healthy ones; better as a measurement |

## Measurements in place

| tool | measures |
|---|---|
| `tools/world_model_probe.py` | how well each checkpoint's state predicts the untouched model's next-frame input |
| `tools/empowerment_check.py` | the state-empowerment estimate on static screens against the open world |
| `tools/policy_drift.py` | base model against a checkpoint on the same frames: actions and per-layer state cosine |
| `tools/repertoire_probe.py` | sampled-action repertoire per checkpoint with energy off: keys, entropy, LB share |
| `tools/fanin_probe.py` | static direct fan-in per action key, with strength and overlap |
| `tools/fanin_active.py` | activation-weighted fan-in per key: drive, spread, overlap |
| `tools/spectrum_probe.py` | weight-change rank, top-8 share, intruder dimensions |
| `tools/thought_influence.py` | whether thoughts change actions |
| key repertoire per 10 min | distinct keys used in play, from the actor logs |

## Rejected (in this role)

| piece | role it failed in |
|---|---|
| Predictive information of the stream (A1, A1-act) | policy reward (dark room). Kept as a measurement |
| Policy-dependent I(A; S') (A2 as built) | policy reward: collapse without a guard, mashing with one |
| Action-entropy guard | any role |
| MI representation term with both ends learned | world model (can be raised by internal slow variables) |
