# 1. Introduction and related work (2026-11-23)

**Status:** second pass, written before any methodology was chosen.
- **1.1–1.3** (first pass) come from web-search snippets, abstracts, and papers
  already cited in sili (`sili__new/CITATIONS.md`, `energy-params.md`).
- **1.4–1.8** (second pass) read four papers in full:
  - Mirostat;
  - SAC, section 5;
  - Plappert et al., section 3 and appendix C;
  - Aston-Jones & Cohen 2005.

  The rest is from abstracts.
- Entries say when only a snippet or abstract was read. *Unverified* marks
  recent preprints whose results were not checked.

## 1.1 The question

A fixed energy dose is wrong most of the time. In the collapse sweep, a fixed
MLP + attention dose held test perplexity at about 78 over 10 self-trained
generations, against 2025 for plain greedy and 2294 for matched noise. But
free-running output stayed loopy, and the training loss fell to 0.008
([site sweep](../results/2026-11-23-collapse-site-sweep.md)). Raise the dose
to try and to learn new things; lower it to focus.

Before designing a controller, this pass asks three things:
1. Has anyone built a mechanism like sili's energy?
2. Did they build a controller for it?
3. How much of such a controller is a simple program, and how much a learned
   system?

## 1.2 What our energy is, in plain mechanism terms

Each neuron keeps a scalar energy *e*, updated every step from its output *h*
(`sili_quartz/energy.py`, a port of sili `EnergyDynamics`):

    e ← e·(1 − decay) + drive + noise − activation_cost·|h|

The search used two kinds of names for this. Names from the code: drive,
activation cost, decay, exploration, setpoint, reactivity, density, precision,
p, fire, shutoff, wake gate. Names for what it *does*, which the code never
uses:

| Code | What it does | Names it goes by elsewhere |
|---|---|---|
| `drive` − `activation_cost`·\|h\| | a leaky integrator of "unused budget": a quiet neuron gains, a busy one pays | metabolic budget, energy budget, drive (Hull), resource pool |
| fire at *e* ≥ 2 (output forced to 2) | a neuron that has been unused too long is forced active | boosting, excitability rising with disuse, homeostatic excitability, spontaneous or metabolically driven spikes, adaptive threshold (downward) |
| shutoff at *e* ≤ −2 | a neuron used too much is shut off | fatigue, spike-frequency adaptation, habituation, repetition suppression, refractoriness, adaptive threshold (upward) |
| `fire_reset_to_zero`, `wake_gate_steps` | reset and hold after firing | refractory period |
| `exploration` (noise in *e*) | random timing of fire and shutoff | stochastic resonance, intrinsic noise, exploration noise |
| `decay` | leak toward rest; a dead band in which nothing ever fires | leak, forgetting, the dead band of a deadzone controller |
| `reactivity` (setpoint loss) | per-neuron push of *e* toward a setpoint, with a gradient to *h* | intrinsic plasticity, per-neuron homeostasis, lifetime sparsity |
| `precision`, `density` (KL term) | population push of the active fraction toward a target | population sparsity, KL sparsity (sparse autoencoders) |
| `p` (hard top-p) | a hard ceiling on the active fraction | k-winners-take-all, k-sparse |

In those terms, sili's energy is a **per-neuron, two-sided homeostatic
excitability rule with noise**, applied at inference *and* in training to a
backprop-trained network. Two-sided means it boosts on disuse and suppresses
on overuse. The "dose" a controller would set is mainly `activation_cost`,
with drive tied to it as cost × mean |h|, plus the noise and decay.

## 1.3 Mechanisms like it

### Homeostatic excitability in trained networks (the disuse side)

- **HTM spatial pooler boosting and k-winners with boosting.** Ahmad &
  Scheinkman 2019, "How Can We Be So Dense?", arXiv 1903.11257.
  - A k-winners layer keeps a duty cycle per unit: a running average of how
    often it wins. Units that win too rarely get a multiplicative boost so they
    win again.
  - The closest deep-learning match to the forced-firing half, and used with
    backprop (nupic.torch). One-sided: it boosts the starved and does not shut
    off the overused, beyond k-WTA itself.
  - The boost strength is a fixed hyperparameter; no controller was found.
    (Snippets and abstract.)
- **Adaptive threshold in STDP networks.** Diehl & Cook 2015, Frontiers
  Comput. Neurosci.
  - Each winner's threshold rises with firing and decays back, so every neuron
    gets to win.
  - Two-sided in effect, but STDP-trained, not backprop. (Abstract.)
- **Intrinsic plasticity.** Triesch 2005; Schrauwen et al. for reservoirs.
  - Each neuron adapts its gain and bias so its output distribution matches an
    exponential with a set mean (a KL target).
  - The nearest match to our `reactivity` setpoint loss. Used in reservoirs
    and echo state networks, not in LLMs. (Snippets.)
- **Activation-threshold homeostasis in standard networks:**
  - MPATH, arXiv 2104.10851: membrane-potential and threshold homeostasis for
    continual learning. (Abstract.)
  - Adaptive threshold to aid grokking in MLPs, arXiv 2608.28184:
    thresholds rise when a unit fires too often and fall when it fires too
    rarely, against a target rate, to stop the population collapsing into a
    few always-on and many always-off units. *Unverified*; that is our
    collapse framing almost word for word.
- **Spiking networks:** Rouser (arXiv 2407.19566) learns thresholds to revive
  dead neurons. AHSAR (arXiv 2512.05015, *unverified*) adjusts each layer's
  excitability from its firing-rate deviation.

### Fatigue and repetition suppression (the overuse side)

- **Neuronal fatigue in a CNN.** Vinken, Boix & Kreiman.
  - An exponentially decaying per-unit fatigue state in a pretrained object
    recognition network reproduces repetition suppression, novelty
    detection and perceptual aftereffects.
  - The closest match to the shutoff half: intrinsic, per unit, in a trained
    deep network. Inference only, with no controller. (Abstract.)
- **Spike-frequency adaptation** in SNNs: a 2024 review, PMC11053160.
- **LLMs.** "Do transformers habituate?" (LessWrong, informal) reports
  activation magnitudes decaying with repeated tokens. Not peer reviewed.

### Repetition in LLMs at the neuron level

- **Repetition neurons.** Hiraoka & Inui, NAACL 2025, arXiv 2410.13497.
  - A few MLP neurons in middle and final layers activate more and more as
    repetition goes on. Deactivating them reduces repetition.
  - A follow-up (arXiv 2507.07810) links them to induction heads and keeps
    in-context learning while ablating them.
  - Relevant twice. It is a mechanism energy may be acting on, since repeated
    use drains energy and shuts these neurons off. And it is a candidate
    "bored" signal for a controller.
- **"Understanding the Repeat Curse"** (arXiv 2504.14218): repetition features
  found with sparse autoencoders; turning them off mitigates repetition.
- **Inference-time noise in the residual stream:**
  - Old precedent: Cho 2016, noisy parallel approximate decoding with an
    annealed noise schedule; arXiv 1911.03373, noise-injection sampling.
  - Recent preprints (arXiv 2604.03380, and a Zenodo record) report that
    hidden-state noise breaks loops. *Unverified.* Our own CPU loop test found
    matched noise did not break loops where energy did.

### Dead and dormant neurons (why the disuse side matters for training)

- **ReDo.** Sokar et al., ICML 2023, arXiv 2302.12902: the share of dormant
  neurons grows during deep RL. Periodically re-initializing them restores
  performance.
- **Continual backprop.** Dohare et al., Nature 2024, arXiv 2306.13812:
  continually re-initialize a small fraction of low-utility units to keep
  plasticity.
- **ReGraMa** (arXiv 2505.24061): pick neurons by gradient magnitude, not
  activation. ReDo misses dormant units in networks with residual paths and
  layer norm, which is the transformer case.
- These rules reset *weights* on a fixed schedule or threshold. Energy acts on
  *activations* every step. Same target (unused units), different lever. None
  has a dose controller beyond the reset fraction and period.

### Energy as an objective, and in biology

- **Ali, Ahmad, de Groot, van Gerven & Kietzmann.** Patterns 2022,
  "Predictive coding is a consequence of energy efficiency in recurrent
  neural networks".
  - Penalizing spikes and synaptic transmission alone makes RNNs predict and
    inhibit their input, and split into error and prediction units.
  - Energy here is a fixed loss weight, not a controlled per-neuron state.
    sili's `energy-proofs.md` makes the same predictive-coding argument.
- **Metabolically driven action potentials** (bioRxiv 2022.10.16.512428).
  - Proposes that some spontaneous firing is a release valve: a neuron that has
    used too little energy fires to protect its mitochondria.
  - The closest biological match to energy's forced firing on disuse.
    (Abstract, *unverified*.)
- **SNEP energy pools** (PMC5326782): each spiking neuron spends from a local
  pool that refills; that dependence alone creates oscillations.
- **Attwell & Laughlin 2001** (cited in sili): the cortical energy budget behind
  `activation_cost`.

### Criticality and attractor escape

- **Zierenberg, Wilting & Priesemann** (arXiv 1807.01479).
  - Homeostatic plasticity plus the external input rate sets the network
    regime: bursting, fluctuating or irregular. Homeostasis keeps a slightly
    sub-critical margin.
  - Relevant to 2c: with no outside data, the input that sets the regime is
    the model's own output. (Snippets.)
- **Noise-driven switching between attractors:** a predictive-coding account of
  chaotic itinerancy (arXiv 2106.08937) and recurrence resonance (arXiv
  2408.05579). Noise moves RNNs between learned attractors.
  - None tests this as exploration during learning.
  - Relevant to energy_rl's original job of breaking stable attractors and
    zero-init deadlocks.

## 1.4 The shape we are looking for: between boredom and confusion

Basu et al. (Mirostat, ICLR 2021) name the two failure ends of text generation:
- **The boredom trap:** cross-entropy falls as the text gets longer, and
  repetition grows.
- **The confusion trap:** cross-entropy rises with length, and the text becomes
  incoherent.

Human text does neither. Its cross-entropy converges to a limit as it gets
longer. These are the same two ends as our results: loops on one side,
gibberish on the other (temperature 1.3, and penalty 1.5's 17k perplexity).

Energy already acts on both ends locally:
- Shutoff stops overused neurons.
- Forced firing wakes unused ones.
- The density target and the hard ceiling bound how much can be active.

What energy cannot know is *where* the two thresholds sit for a given model,
text or moment. A fixed dose put the student in the boredom trap: training
loss 0.008, distinct-3 0.086. So **the controller's job is to place energy
between the two traps, and keep it there as the situation changes.** It does
not replace energy.

The rest of this part looks at each system that controls something across
those two ends. For each one it asks three things:
1. What does it measure?
2. What does it move, and by what rule?
3. Is it two-sided, and does it need an extrinsic reward?

The plan is to train with **no extrinsic reward at all** (user, 2026-11-23).

## 1.5 Computer science systems, in detail

### Mirostat: integral control of surprise

Read in full: arXiv 2007.14966.

- **Measures:** the surprise S(X) = −log p(X) of each sampled token.
- **Moves:** top-k, through a variable µ that starts at 2τ. k comes from µ and
  a Zipf exponent fitted to the top 100 probabilities each step.
- **Rule:** e = S(X) − τ, then µ ← µ − η·e. The error is summed into µ, which
  makes this an **integral controller** on one statistic, with a hand-set
  target τ.
- **Two-sided:** yes. Surprise above τ tightens k; surprise below τ widens it.
- **Extrinsic reward:** none needed.
- **Evidence:**
  - Repetition is close to linear in observed cross-entropy, and once the
    model and the cross-entropy are fixed, it hardly depends on the sampler
    (top-k, top-p, temperature). So *one statistic, surprise, locates both
    traps*.
  - Sentence-level repetition disappears above about 3 nats.
  - 43 human raters preferred τ = 3 for fluency, coherence and quality, and
    more than half took that text for human-written (GPT-2 117M, 300 tokens).
  - It saturates above about 5 nats, the model's untruncated surprise: it can
    only cut the tail, not add surprise.
- **Limits for us:**
  - It acts on the token distribution only, after the model has computed. A
    loop that lives in the hidden state (repetition neurons) is only steered
    from outside.
  - τ is hand-set and model-specific.
  - It cannot push surprise *above* the model's own level. Energy can, because
    it changes the computation itself.

### Soft actor-critic: a dual variable on an entropy floor

Read in full: arXiv 1812.05905, section 5.

- **Measures:** policy entropy, −log π(a|s), averaged over visited states.
- **Moves:** the temperature α, by gradient on J(α) = E[−α log π(a|s) − α·H̄].
  α is the dual variable of the constraint "average entropy ≥ H̄". α rises when
  entropy is below H̄ and falls toward 0 when it is above.
- **Two-sided:** no. It is a **floor only**. The paper drops the upper bound
  because return already pushes the policy toward determinism, so the other
  side comes from the reward.
- **Extrinsic reward:** required for that other side. Without it, nothing pulls
  entropy down; on its own it guards only against boredom.
- **Worth keeping:**
  - The constraint is on the *average*, so entropy may differ by state: explore
    where uncertain, stay deterministic where the answer is clear. A per-region
    or per-token dose should keep that freedom.
  - The target is still a hand heuristic (−dim A), and later work learns it
    (Meta-SAC) or anneals it.

### Plappert et al.: dose set by its measured effect

Read in full: arXiv 1706.01905, section 3 and appendix C.

- **Measures:** the *effect* of the perturbation, d(π, π̃), the distance
  between the clean and the perturbed policy. For DQN it is the KL between
  their softmax policies. For DDPG it is the RMS action difference.
- **Moves:** the weight-noise scale σ. **Rule:** σ ← 1.01·σ if d ≤ δ, else
  σ ← σ/1.01. This is a two-sided bang-bang multiplicative controller.
  - It updates every 50 steps for DQN.
  - The perturbation is held for a whole episode, so exploration is consistent
    within a state rather than per-step jitter.
- **Choosing δ:** δ is not tuned. It is set equal to the effect of a familiar
  dose: the KL of ε-greedy, −log(1 − ε + ε/|A|), or the σ of action noise.
- **Why it is needed:** the right σ depends on the architecture and drifts
  during training, as parameters get more sensitive. Layer norm between
  perturbed layers lets one σ serve every layer.
- **Two-sided:** yes, around the effect.
- **Extrinsic reward:** not needed. The controller only looks at the two
  policies; the reward is used by the learner, not by the controller.
- **Evidence:** only parameter noise learned at all in SparseCartpoleSwingup
  and SparseMountainCar; all action noise never found a reward. The authors
  warn that gains are case by case.
- **The closest structure to our problem:**
  - Our per-site scaling already makes one cost mean the same dose at every
    site, by dividing by each site's mean |h|, the way their layer norm does.
  - Our loop tests already measured energy's effect in nats: energy broke loops
    at about 1–1.5 nats of damage, where matched noise did not.
  - The obvious controlled variable is therefore the KL between next-token
    distributions with energy on and off.

### Curiosity and its trap; counts and learning progress

- **Prediction-error curiosity** (Schmidhuber 1991; Pathak 2017; Burda et al.
  2018, arXiv 1808.04355) rewards what the model fails to predict.
  - That escapes boredom and then heads straight into confusion: the "noisy TV".
    A TV with a channel-change action greatly slows learning.
  - A poor model or partial observability causes the same trap as true noise.
  - One-sided.
- **Count-based exploration** (Bellemare et al. 2016, pseudo-counts) rewards
  what has been *visited* least, not what is least predictable. It reliably
  avoids noisy TVs in minigrid, but is hard to apply to large state spaces.
- **Learning progress** (Oudeyer, Kaplan & Hafner 2007) rewards the *drop* in
  prediction error. That rules out the boring (nothing left to learn) and the
  hopelessly noisy (no progress possible): "neither too predictable nor too
  unpredictable". It is two-sided, because both traps give zero progress.
- **Where energy fits:** energy's forced firing of disused neurons is a
  **count bonus over neurons**. A neuron's energy rises with *how long it has
  gone unused*, not with surprise. So energy is closer to count-based than to
  curiosity-based exploration, the family that resists the noisy TV.

### Directed and random exploration

Wilson et al. 2014 (J Exp Psychol Gen): people use two separable strategies,
and use more of both when the horizon is longer:
- *directed* exploration: an information bonus toward the less-known option;
- *random* exploration: decision noise.

Pupil studies tie norepinephrine to the random kind, not the directed one.
Energy has both:
- **random:** the noise term in energy, `exploration`;
- **directed:** forced firing of the least-used neurons, as above.

A controller can set them separately.

### Agent57: ruled out

- **What it does:** a sliding-window UCB bandit picks the intrinsic-reward
  weight and the discount, *rewarded by extrinsic episode return*.
- **Why it doesn't fit:** with no game score there is nothing to reward the
  bandit.
- **What carries over:** only its structure, a slow selector over a few preset
  regimes. That would need an intrinsic score in place of the game score.

## 1.6 Biology, in detail

### Locus coeruleus and norepinephrine: adaptive gain

Read in full: Aston-Jones & Cohen 2005, Annu. Rev. Neurosci. 28:403–450.

- **What NE does:** it raises the *gain*, the slope of the sigmoid, of target
  neurons. That strengthens both excitatory and inhibitory responses and
  pushes units toward binary responding.
- **Two modes:**
  - *Phasic:* moderate baseline, with bursts locked to task-relevant decisions.
    Performance is accurate. This is exploitation.
  - *Tonic:* high baseline and no bursts, so gain rises everywhere and
    indiscriminately. Behavior is distractible. This is exploration.
  - *Low tonic:* drowsy and disengaged.
- **Performance follows an inverted U of tonic activity**, the Yerkes–Dodson
  curve. The authors call the modes the ends of a continuum.
- **The mode switch, in the model:** weaker electrotonic coupling between LC
  neurons gives the tonic mode, and so does more baseline excitatory drive.
  Stronger coupling and less drive give the phasic mode. One or two physical
  parameters set the regime.
- **What sets the mode: utility, from frontal cortex.**
  - Orbitofrontal cortex (OFC) evaluates reward.
  - Anterior cingulate (ACC) evaluates cost: errors, conflict, difficulty, pain.
  - These are the main cortical inputs to the monkey LC.
  - Utility is averaged over a short (seconds) and a long (minutes) window,
    and combined:

        engagement = [1 − logistic(short-term utility)] · logistic(long-term utility)

  - **When long-term utility is high,** a short-term drop *raises*
    engagement (phasic), repairing a lapse.
  - **When long-term utility stays low,** engagement falls and LC turns tonic:
    disengage and explore.
- **Explicitly annealing.** The authors call exploration "akin to annealing":
  - start with high noise in a new environment;
  - lower it as valuable states are found;
  - **re-melt** (raise noise again) when utility collapses.

  This is our raise-to-explore, lower-to-focus, with the re-melt as the
  escape from a stale attractor.
- **Evidence:**
  - *For:* in pupillometry, Gilzenrat et al. 2010 found that falling long-term
    utility and disengagement went with a larger baseline pupil and smaller
    dilations, which is the tonic signature. Jepma & Nieuwenhuis 2011 found
    that a larger baseline pupil came before exploratory choices.
  - *Against:* raising NE with reboxetine did not increase exploration (Jepma
    et al. 2010).
  - So NE plausibly tracks *random* exploration, and causation is unsettled.

### Amygdala: surprise sets how much to attend and learn

- **The Pearce–Hall link:** amygdala neurons carry the Pearce–Hall
  associability signal, where attention and learning rate follow recent
  *unsigned* surprise.
  - *Basolateral amygdala* (Roesch, Calu, Esber & Schoenbaum 2010, J Neurosci):
    neurons fire more right after a reward changes, and that predicts faster
    orienting on later trials. Inactivating them blocks both the orienting and
    the learning after a change.
  - *Central nucleus* (Calu et al. 2010, Neuron): neurons fire when an expected
    reward is omitted, and decline as learning catches up.
  - *Dopamine:* the basolateral signal needs midbrain dopamine (Esber et al.
    2012).
- **It is a learning-rate controller:** high recent surprise means learn faster
  and attend more, and it decays as prediction recovers. This is the "raise to
  learn new things" half, driven by surprise magnitude, not by reward.
- **Novelty and salience:** the amygdala tags novelty as a separate affective
  dimension. LC–amygdala coupling rises during novelty.
- **Anxiety:** anxiety goes with *less* exploration under uncertainty (2023–2024
  computational reviews). The amygdala's specific part in that is not
  established; one human study found insula, not amygdala, activity for
  sensory uncertainty. This is the brake side: a threat or uncertainty signal
  that narrows exploration.

### Prefrontal cortex: where the targets come from

- **OFC** carries value, and **ACC** carries cost and conflict; together they
  are the utility inputs to LC above.
- **Expected value of control** (Shenhav, Botvinick & Cohen 2013, Neuron):
  dorsal ACC decides *how much* control to allocate. It weighs the expected
  payoff of control against its cost: a controller of a dose, set by value
  minus effort.
- **Frontopolar cortex** (Daw et al. 2006, Nature) is more active on
  exploratory choices. Those choices fit a value-sensitive softmax (random
  exploration scaled by value differences) better than ε-greedy. An
  uncertainty bonus added nothing at the best fit.
- **Doya 2002:** neuromodulators as hyperparameters. Dopamine is the TD error,
  serotonin the discount, noradrenaline the inverse temperature, acetylcholine
  the learning rate.
- **Yu & Dayan 2005:** acetylcholine signals *expected* uncertainty and
  norepinephrine *unexpected* uncertainty.

### sili's own learned "prefrontal" regulator

From `sili__new/energy-params.md`, "The Prefrontal Meta-Controller".

- **What it reads:** each region's energy map, as an image (near-firing, in
  refractory, and spatial structure), through a convolutional or attention
  model.
- **What it sets, per region:** drive, cost, noise, setpoint, reactivity,
  density and KL weight.
- **Its constraints:**
  - outputs stay inside the stable subspace;
  - it is itself an energy-regulated RNN, so it is homeostatic too.
- **The file maps regimes to parameter directions:**
  - *focus/flow:* noise ↓, KL ↑, reactivity ↑;
  - *curiosity:* noise ↑, cost ↓;
  - *sleep:* drive ↓, noise ↓, reactivity ↑;
  - *anxiety:* setpoint ↓, reactivity ↑, drive ↑;
  - *mania:* drive ↑↑, setpoint ↑↑, KL ↓.
- **Compared with biology:** it is the closest to the frontal tier, a learned
  reader of internal state that outputs regime parameters.
- **What it leaves open:** the training signal. It does not say what the
  controller is optimizing when there is no extrinsic reward.

## 1.7 Side by side

| System | Measures | Moves | Rule | Two-sided | Extrinsic reward | Timescale |
|---|---|---|---|---|---|---|
| Mirostat | token surprise | top-k (via µ) | integral: µ ← µ − η(S − τ) | yes | no | per token |
| SAC α | mean policy entropy | temperature α | dual gradient on an entropy floor | **no** (floor) | yes, for the other side | per update |
| Plappert | KL(clean ‖ perturbed) | noise σ | ×1.01 or ÷1.01 around δ | yes | no | every 50 steps |
| Prediction-error curiosity | prediction error | intrinsic reward | maximize | **no** (noisy TV) | no | per step |
| Learning progress | drop in prediction error | where to explore | maximize | yes | no | windowed |
| HTM boosting / k-WTA | duty cycle per unit | per-unit boost | fixed gain | one-sided (disuse) | no | running average |
| Agent57 | episode return | (β, γ) preset | sliding-window UCB | yes | **yes** | episodes |
| LC-NE adaptive gain | utility (OFC reward, ACC cost), 2 windows | gain; tonic vs phasic | engagement = (1 − σ(u_s))·σ(u_l) | yes, inverted U | utility, can be internal | seconds and minutes |
| Amygdala (Pearce–Hall) | unsigned surprise | attention, learning rate | rises with surprise, decays with learning | yes | no | trials |
| ACC (EVC) | payoff − cost of control | amount of control | maximize net value | yes | value, can be internal | trials |
| sili prefrontal regulator | energy maps per region | the 7 energy parameters | learned, within the stable subspace | yes | open | slow |
| **sili energy itself** | per-neuron use | fire or shutoff per neuron | integrate and fire | yes, per neuron | no | per step |

Similarities with our idea:
1. **The two-sided ones agree on the method.** Mirostat, Plappert and LC-NE all
   regulate around a target, and all use a *measured* quantity rather than a
   fixed dose: surprise, the perturbation's effect, utility.
2. **Plappert and our per-site scaling make the same move:** normalize the
   perturbation by its effect, so one setting means the same thing everywhere.
3. **Energy's directed half is count-like,** the family that avoids the noisy
   TV. Its random half is NE-like.
4. **Biology splits the work in three:**
   - a fast gain or noise level (LC);
   - a surprise-driven learning rate (amygdala);
   - slow, two-timescale utility that sets the regime (OFC, ACC).

   sili's regulator corresponds to the slow tier.

Differences:
1. Every computer-science controller sets *one global* knob. Energy is per
   neuron, and the controller sets how those per-neuron rules behave (dose,
   noise, setpoint), maybe per region.
2. Mirostat and SAC act on the output distribution. Energy changes the
   computation, so it can raise surprise above the model's own level, where
   Mirostat saturates.
3. The biological "utility" is reward-based. With no extrinsic reward it has to
   be internal: prediction success, learning progress, or homeostatic error
   (Keramati & Gutkin: reward is the reduction of deviation from a setpoint).

## 1.8 What this implies we should build, and why

**Ruled out:**
- Agent57, because it needs a game score.
- SAC's floor on its own, because without reward nothing pulls the other way.
- Prediction-error curiosity as the signal, because of the noisy TV.

**Candidate 1, the fast tier, a simple program.** Hold the effect of energy at
a target, Plappert style, and move that target with surprise, Mirostat style:
- *Measured effect:* d = KL(p_energy ‖ p_clean) on next-token distributions.
  That is one extra forward pass without energy, or a sampled subset of
  steps.
- *Dose rule:* cost ← cost·1.01 if d < δ, else cost/1.01.
- *Target rule:* δ ← δ + η·(τ − S̄), integral control. S̄ is the running
  surprise of the model's own output.
  - Bored (S̄ < τ) raises δ, so more energy.
  - Confused (S̄ > τ) lowers it.
- *Starting values:* δ from what broke loops in our tests (about 1–1.5 nats
  over a window, to be converted to per-token). τ from Mirostat's human-text
  range (about 3 nats on GPT-2), checked against real text for this model.
- *Why first:* it is two-sided, needs no extrinsic reward, has two parameters
  with published defaults, and runs inside one run. It also gives the direct
  comparison to Mirostat that TODO 2d asks for.

**Candidate 2, two timescales, LC style.** Replace the single target with
Aston-Jones & Cohen's engagement equation, using an internal utility:
- *Utility:* learning progress, the drop in the student's loss on its own data,
  over a short and a long window.
- *Engagement high:* focus. Lower noise, keep the dose.
- *Engagement low:* re-melt. Raise drive and noise.
- *Why:* this is the only model that says when to *re-explore after settling*.
  That is exactly the fixed-dose failure we saw, where the loss hit 0.008 and
  nothing changed it. A loss floor means no learning progress, so the re-melt
  would trigger there.

**Candidate 3, the slow tier, learned: sili's prefrontal regulator.**
- *What it does:* it learns per-region targets for candidates 1 and 2, reading
  the energy maps.
- *Its training signal:* learning progress, or homeostatic reward.
- *When:* after 1 and 2 show which statistics matter. Its training signal is
  the open question, and biology suggests it sets targets slowly rather than
  doses quickly.

**Recommended order for the methodology:**
1. Candidate 1 in the loop test, against fixed doses and Mirostat.
2. Candidate 2 in the collapse loop, where re-melting can be tested.
3. Candidate 3 last.

An amygdala-style term, a learning rate that rises with unsigned surprise, is
a natural extra arm for candidate 2. It is separate from the energy dose, and
the amygdala literature treats it as a separate system.

**Not yet decided:**
- whether the dose is per region or global;
- which energy parameters the fast tier moves (cost only, or noise and drive
  separately, for random vs directed exploration);
- how often the clean forward pass for d is run.

## 1.9 Search again in a later pass

- Mirostat's appendix: the Zipf estimate, and the alternative token-iteration
  version (appendix E).
- The Gilzenrat 2010 and Jepma & Nieuwenhuis 2011 full texts: the exact
  utility windows.
- The *unverified* preprints from 1.3.
- Whether anyone has applied homeostatic excitability inside transformers
  during training. None was found so far.

## Sources

Disuse side (boosting, thresholds, intrinsic plasticity):
- [Ahmad & Scheinkman 2019, How Can We Be So Dense?](https://arxiv.org/pdf/1903.11257)
- [HTM spatial pooler formalization](https://arxiv.org/pdf/1601.06116.pdf)
- [HTM SP formal analysis](https://arxiv.org/pdf/1607.00791)
- [Diehl & Cook 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4522567)
- [Triesch 2005, intrinsic and synaptic plasticity](https://proceedings.neurips.cc/paper_files/paper/2004/file/08f38e0434442128fab5ead6217ca759-Paper.pdf)
- [Intrinsic adaptation in autonomous RNNs](https://arxiv.org/pdf/1110.3161)
- [MPATH](https://arxiv.org/pdf/2104.10851)
- [Adaptive thresholds for grokking (unverified)](https://arxiv.org/pdf/2608.28184)
- [Rouser](https://arxiv.org/pdf/2407.19566v1)
- [AHSAR (unverified)](https://arxiv.org/html/2512.05015v1)

Overuse side (fatigue, adaptation, habituation):
- [Vinken, Boix & Kreiman, neuronal fatigue in DNNs](https://www.biorxiv.org/content/10.1101/642777.full.pdf)
- [Spike frequency adaptation review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11053160/)
- [Do transformers habituate? (informal)](https://www.lesswrong.com/posts/Q5H97tN2rvH7nJuji/do-transformers-habituate-investigating-repetition)

Repetition in LLMs and hidden-state noise:
- [Hiraoka & Inui, Repetition Neurons](https://arxiv.org/abs/2410.13497v2)
- [Repetition neurons and induction heads](https://arxiv.org/pdf/2507.07810.pdf)
- [Understanding the Repeat Curse](https://arxiv.org/html/2504.14218v2)
- [Residual-stream noise at inference (unverified)](https://arxiv.org/pdf/2604.03380)
- [Noise injection sampling](https://arxiv.org/pdf/1911.03373)
- [Noisy parallel approximate decoding](https://arxiv.org/pdf/1605.03835)

Dormant neurons and plasticity loss:
- [Sokar et al., ReDo](https://arxiv.org/abs/2302.12902)
- [Dohare et al., Maintaining plasticity](https://arxiv.org/html/2306.13812v3)
- [Loss of plasticity, Nature](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11338828/)
- [ReGraMa](https://arxiv.org/pdf/2505.24061)

Energy as an objective, and metabolic biology:
- [Ali et al. 2022, predictive coding from energy efficiency](https://pmc.ncbi.nlm.nih.gov/articles/PMC9768680/)
- [Metabolically driven action potentials (unverified)](https://www.biorxiv.org/content/biorxiv/early/2022/10/18/2022.10.16.512428.full.pdf)
- [SNEP energy pools](https://pmc.ncbi.nlm.nih.gov/articles/PMC5326782/)

Criticality and attractor escape:
- [Zierenberg, Wilting & Priesemann](https://arxiv.org/html/1807.01479)
- [Self-organization to sub-criticality](https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-16-S1-O19)
- [Predictive coding account of chaotic itinerancy](https://arxiv.org/pdf/2106.08937)
- [Recurrence resonance](https://arxiv.org/html/2408.05579v1)

Feedback controllers:
- [Mirostat](https://arxiv.org/pdf/2007.14966)
- [EDT](https://arxiv.org/pdf/2403.14541)
- [Entropic online sampling](https://www.cs.purdue.edu/homes/spa/papers/token_online.pdf)
- [SAC algorithms and applications](https://arxiv.org/pdf/1812.05905)
- [Meta-SAC](https://arxiv.org/pdf/2007.01932)
- [Plappert et al., parameter space noise](https://arxiv.org/pdf/1706.01905)
- [Parameter-space noise for RLVR](https://arxiv.org/pdf/2602.02555)
- [AER, adaptive entropy coefficient](https://arxiv.org/html/2510.10959v2)
- [Revisiting entropy in RL for reasoning models](https://arxiv.org/html/2511.05993v2)

Learned and slow selectors:
- [Agent57](https://arxiv.org/pdf/2003.13350)
- [Population based training](https://arxiv.org/pdf/1711.09846v1.pdf)
- [Oudeyer, computational curiosity review](https://arxiv.org/pdf/1802.10546.pdf)
- [Forestier et al., goal exploration](https://arxiv.org/pdf/1708.02190)
- [Backpropamine](https://arxiv.org/pdf/2002.10585)

Neuromodulation and homeostatic RL:
- [Aston-Jones & Cohen 2005 (review PDF)](https://pages.ucsd.edu/~jpineda/COGS260/monoamines/LC-NE%20review.pdf)
- [Doya 2002 (record)](https://www.scilit.net/publications/f6b38cae76f13a56a4319668784cb813)
- [Yu & Dayan 2005](https://princetonuniversity.github.io/NEU-PSY-502/_static/pdf/Class%208/Yu2005.pdf)
- [Keramati & Gutkin 2014](https://elifesciences.org/articles/04811)

Read in full for the second pass:
- [Basu et al., Mirostat](https://arxiv.org/pdf/2007.14966)
- [Haarnoja et al., SAC algorithms and applications](https://arxiv.org/pdf/1812.05905)
- [Plappert et al., parameter space noise](https://arxiv.org/pdf/1706.01905)
- [Aston-Jones & Cohen 2005](https://pages.ucsd.edu/~jpineda/COGS260/monoamines/LC-NE%20review.pdf)

Also used in the second pass:
- [Burda et al., large-scale study of curiosity](https://arxiv.org/pdf/1808.04355)
- [Bellemare et al., count-based exploration](https://arxiv.org/pdf/1606.01868)
- [Noisy-TV avoidance by aleatoric uncertainty](https://arxiv.org/pdf/2102.04399)
- [Wilson et al. 2014, directed and random exploration](https://pmc.ncbi.nlm.nih.gov/articles/PMC5635655)
- [Daw et al. 2006, cortical substrates of exploratory decisions](https://pmc.ncbi.nlm.nih.gov/articles/PMC2635947)
- [Shenhav, Botvinick & Cohen 2013, expected value of control](https://pmc.ncbi.nlm.nih.gov/articles/PMC3767969)
- [Roesch, Calu, Esber & Schoenbaum 2010, basolateral amygdala and Pearce-Hall](https://pmc.ncbi.nlm.nih.gov/articles/PMC2838173)
- [Calu et al. 2010, central amygdala](https://pmc.ncbi.nlm.nih.gov/articles/PMC3033562)
- [Novelty as a dimension in the affective brain](https://www.sciencedirect.com/science/article/abs/pii/S105381190901026X)
- [Influence of anxiety on exploration, review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11213056)
- [Anxiety as uncertainty learning](https://link.springer.com/article/10.3758/s13415-023-01080-w)
- [Gilzenrat et al. 2010, pupil and adaptive gain](https://link.springer.com/content/pdf/10.3758/CABN.10.2.252.pdf)
- [Pupil and exploration (Jepma & Nieuwenhuis line)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3403821)
- [Agrawal, Mattar, Cohen & Daw, opportunity costs of fatigue and boredom](https://www.biorxiv.org/content/10.1101/2020.09.08.287276.full.pdf)
- [Keramati & Gutkin 2014, homeostatic RL](https://elifesciences.org/articles/04811)
