# 1. Introduction and related work (first pass, 2026-11-23)

**Status:** first pass, written before any methodology was chosen. It comes
from web search snippets, abstracts, and papers already cited in sili
(`sili__new/CITATIONS.md`, `energy-params.md`). Where only a snippet or abstract
was read, the entry says so. Entries marked *unverified* are recent preprints
whose results were not checked.

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

## 1.4 Controllers: who sets the dose, and how

No controller was found for a per-neuron homeostatic dose in a backprop
network. What exists controls analogous *global* knobs. The pattern is
consistent across all of them.

### Simple feedback on one measured statistic, toward a target

- **Mirostat.** Basu et al., ICLR 2021, arXiv 2007.14966.
  - Targets a surprise τ per token, by feedback on top-k: μ ← μ − η·(observed
    surprise − τ).
  - Names a "boredom trap" (repetition at low k) and a "confusion trap"
    (incoherence at high k). These are our loop and gibberish ends.
  - Controls token sampling, not neurons. TODO 2d's baseline.
- **EDT.** Zhang, Bao & Huang, arXiv 2403.14541: temperature set per token from
  the model's entropy.
- **EOS** (Purdue): an online inverse temperature that tracks an
  entropy-constrained optimum.
- **SAC automatic temperature.** Haarnoja et al., arXiv 1812.05905.
  - Entropy is a constraint and the temperature is its dual variable, updated
    by gradient to keep policy entropy at a target.
  - The target entropy is still a hand-set number. Meta-SAC (arXiv 2007.01932)
    learns it by metagradient.
- **Adaptive parameter-space noise.** Plappert et al., ICLR 2018, arXiv
  1706.01905.
  - Weight noise σ is rescaled every K steps by a fixed factor, up or down,
    so that the *effect* of the noise (distance between clean and noisy
    policy) stays at a threshold δ.
  - Structurally the closest: it sets a perturbation dose by its measured
    effect, not its size. That is what per-site scaling did for energy.
  - A 2026 RLVR paper reuses it with a KL target (arXiv 2602.02555).
- **Adaptive entropy coefficient for LLM RL.** AER, arXiv 2510.10959. Fixed
  entropy coefficients are hard to tune (too high and entropy explodes, too
  low and nothing happens), so it adapts the coefficient. This is one of
  several 2025 fixes for entropy collapse in RLVR.
- **Per-unit feedback:** HTM boosting (duty cycle → boost) and DeepSeek-V3's
  bias-based MoE load balancing (load → routing bias). Both are per-unit
  feedback controllers of selection, with a fixed gain.

### A slower, learned selector on top

- **Agent57.** Badia et al., ICML 2020, arXiv 2003.13350.
  - A sliding-window UCB bandit per actor chooses among fixed (intrinsic
    weight β, discount γ) pairs, rewarded by *extrinsic* episode return.
  - The closest published controller of an intrinsic-exploration dose. It
    picks a preset arm on episode timescales; it does not track a statistic
    per step.
- **PBT / PB2** (Jaderberg et al., arXiv 1711.09846; a GP-bandit version):
  hyperparameters are copied and perturbed across a population by fitness.
- **Learning progress.** Oudeyer, Kaplan & Hafner 2007, IEEE TEC.
  - Choose where to explore by learning progress, the drop in prediction
    error: situations "neither too predictable nor too unpredictable".
  - It is a signal, not a dose controller, and the nearest literature match to
    "raise the dose to learn".

### Biology: one fast and one slow loop

- **Adaptive gain theory.** Aston-Jones & Cohen 2005, Annu. Rev. Neurosci.
  28:403–450.
  - Locus coeruleus norepinephrine has a phasic mode (gain on the current
    task: exploit) and a tonic mode (disengage and search: explore).
  - The switch is driven by utility estimates in orbitofrontal and anterior
    cingulate cortex.
  - Human tests are mixed: one reboxetine study failed to find the effect, and
    pupillometry studies support it.
- **Metalearning and neuromodulation.** Doya 2002, Neural Networks
  15:495–506.
  - Neuromodulators as the brain's hyperparameters: dopamine for TD error,
    serotonin for discount, noradrenaline for the randomness of action choice
    (inverse temperature), acetylcholine for the learning rate.
- **Uncertainty, neuromodulation and attention.** Yu & Dayan 2005, Neuron
  46:681–692. Acetylcholine tracks *expected* uncertainty, norepinephrine
  *unexpected* uncertainty (surprise or change).
- **Homeostatic RL.** Keramati & Gutkin, eLife 2014: reward is the predicted
  reduction in distance of an internal state from its setpoint. In that frame
  energy's setpoint loss is a reward, which bears on item 6.
- **Backpropamine.** Miconi et al., ICLR 2019, arXiv 2002.10585: a network
  outputs its own neuromodulatory signal that gates plasticity, trained
  end to end. A learned modulator, but of plasticity, not of excitability.
- **sili's own design** (`energy-params.md`, "The Prefrontal Meta-Controller")
  already describes a learned regulator. It reads energy maps per region, sets
  {drive, cost, noise, setpoint, reactivity, density, precision} within a
  stable range, and is itself homeostatic.

## 1.5 What this first pass found

1. **No complete precedent was found.** Nothing found has all of these at once:
   - a per-neuron integrate-and-fire energy that forces firing on disuse *and*
     shuts off on overuse;
   - noise in the energy;
   - running at inference and in training;
   - inside a pretrained, backprop-trained LLM.

   The halves exist separately. k-winners boosting and the adaptive thresholds
   are the disuse side; neuronal fatigue and repetition suppression are the
   overuse side. Repetition neurons show the overuse side is where LLM loops
   live.
2. **No controller for a per-neuron homeostatic dose was found.** Controllers
   exist for global analogues: temperature (Mirostat, EDT), entropy
   coefficient (SAC, AER), noise scale (Plappert), exploration mix (Agent57).
3. **The program-vs-AI split has a consistent answer in the literature.** It
   is two tiers:
   - *fast*: a simple feedback program that holds one measured statistic at a
     target (Mirostat, SAC alpha, Plappert, boosting);
   - *slow*: a learned or bandit selector that sets the target, or picks among
     presets, from task outcome (Agent57 on return, PBT, Meta-SAC, learning
     progress).

   Biology's adaptive-gain account has the same shape: fast phasic/tonic
   switching, set by slower frontal utility estimates. This is a candidate
   for the methodology, not a decision.
4. **Candidate signals, for the methodology stage:**
   - surprise against a target (Mirostat);
   - repetition-neuron activity, or energy's own shut-off fraction, as a
     "bored" signal (2 to 13× more neurons switched off on loops);
   - learning progress (the change in loss);
   - the training-loss floor (0.008 at the fixed dose);
   - extrinsic return where there is one;
   - expected vs unexpected uncertainty;
   - health statistics already in sili: branching ratio, dormant fraction.
5. **Search again in pass two:**
   - exact update rules: the HTM boost function, Plappert's σ update, the
     Agent57 window and arms, the SAC target-entropy defaults;
   - the *unverified* preprints: grokking thresholds, AHSAR, residual-noise
     loop breaking, metabolically driven spikes;
   - whether anyone applies homeostatic excitability inside transformers
     during training, since these searches found none.

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
