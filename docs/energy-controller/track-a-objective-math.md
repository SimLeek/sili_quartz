# Track A objectives: what the math says (2026-11-27)

Before more ablations: what information theory and the literature say about
the pieces on the table, which combination follows, and what each arm
predicts. Notation: S = the model's state for a frame, A = the action chunk,
S' and X' = the next state and the next real input (the next frame), Z = the
trained state that should carry a world model.

## 1. Why the MI arms went to static screens

A1 maximizes I(past; future) with past = (S, A). The chain rule splits it:

    I((S, A); S') = I(S; S') + I(A; S' | S)

- I(S; S') is passive predictability. A static screen maximizes it: the pause
  menu, the death screen, the link dialog. This is the "dark room" of
  free-energy agents (Friston, Thornton and Clark 2012).
- I(A; S' | S) is control: how much the action decides the next state. On a
  static screen it is 0.

Observed:

| arm | past | collapse |
|---|---|---|
| A1 | (S, A) | two alternating no-key actions |
| A1-act | S only, which is pure I(S; S') | held the pause menu for the last ~85 minutes |
| A1pa | (S, A) | paused by minute 15 (format bug as well) |
| A1io | (S, A) | death screen from minute ~25 (format bug as well) |

A1-act is exactly the first term, and it found the darkest room. This matches
the predictive-information robot work (Ay et al. 2008; Martius, Der and Ay
2013). There, maximizing predictive information gave rich behaviour because a
noisy body cannot freeze its own sensor stream. A pausable game breaks that
assumption.

## 2. Why our A2 rewards randomness

A2 uses I(A; S') under the acting policy. Even the conditional version
decomposes as

    I(A; S' | S) = H(A | S) - H(A | S, S')

As a PPO reward, this contains the policy's own entropy H(A | S): it is an
entropy bonus plus "the action can be read off its effect". The best policy
under it picks random actions whose effects differ, and large mouse swings do
that. A2g's actions after minute 15 had median mouse moves of 1,400–2,000.
That run also had the one-sided entropy guard, so it does not separate the
two causes. A2c, with no guard, tests the prediction.

Empowerment as defined (Klyubin, Polani and Nehaniv 2005; Salge, Glackin and
Polani 2014) is different. It is a channel capacity, max over a source
distribution ω of I(A; S' | s), where ω is separate from the acting policy.
The agent is rewarded for reaching states where that capacity is high
(Mohamed and Rezende 2015). Karl et al. 2017 estimate it through a learned
dynamics model. Under that definition:

- a static screen has capacity 0, so there is no dark room;
- the policy's own randomness earns nothing, so there is no mashing.

## 3. Why the input tether is principled

Barber and Agakov (2003): for any predictor q,

    I(Z; X') >= H(X') + E[ log q(X' | Z) ]

The chunk tether trains the state so that its depth average predicts the next
real input, detached. H(X') does not depend on the model, so lowering the
prediction error raises a lower bound on how much the state knows about the
next real frame.

The A1 representation term differs: I(f(S); g(S')) with f and g both learned.
Any slow internal variable can raise it, a clock for example, with nothing to
do with the world. Tschannen et al. (2020) show that MI-maximizing
representations do not work *because of* the MI. The tether's target cannot
be moved by the model, which is what "tethered to reality" means in these
terms.

Two caveats:

- **Deeper chunks predict internal targets.** Only chunk 1's target is the
  raw input. Chunks 2–6 predict their own inputs at the next frame, which
  earlier chunks produce. That is self-predictive learning with a
  stop-gradient target, as in SPR (Schwarzer et al. 2021) and BYOL-Explore
  (Guo et al. 2022). Tang et al. (2023) analyse why the stop-gradient
  prevents collapse.
- **The output tether has no comparable bound.** It distils the model's own
  sampled tokens into the layers below the output. It may sharpen the policy
  toward its own choices; the logged entropy will show that.

Being a representation loss, the tether pays the policy nothing for a static
world through its gradient. **That holds only while the output side is left
alone** (correction, Tin_c, 2026-11-27).

- **What went wrong:** with all six chunks tethered, the features the action
  head reads are trained to predict the next input. The policy becomes a
  by-product of input prediction.
- **Why it reaches a dark room:** actions change the next input, and
  features that predict the next input fit best when the actions do not
  change it. That is a dark room reached through the shared representation,
  not through a reward.
- **Observed:** Tin_c sat on its death screen, confident (log-prob −0.31,
  against −0.52 at the start), no clicks, mouse moves of 2–4.
- **Fix:** tether only the input-side chunks (`--tether-chunks 3`, layers
  1–12) and leave the output side to a policy.

## 4. What the column averaging guarantees

For a chunk of n layers, with average m = (1/n) Σ h_k:

- **Equal share of the error:** ∂m/∂h_k = 1/n. Each layer gets the same share
  of the error and can correct the average.
- **Robust to a few odd neurons:** one neuron forced or exhausted by δ moves
  the average by δ/n.

These are exact and easy. Bend can check identities like these over Nat, but
they are not where the uncertainty is. The open questions are empirical:
does the bound in section 3 rise in practice, and does it carry over to
behaviour?

## 5. Energy, and where it overlaps with the information terms

Energy perturbs neurons, not the world. That matters for any
estimate made through the model:

- **State empowerment through the world model** compares the predicted next
  inputs across probe actions. If each probe action got a different energy
  draw, the noise would spread the predictions and look like capacity. This
  is the "exhaustion overlaps with MI" worry, made concrete.
  - Fix: evaluate every probe action under the same energy draw (common random
    numbers), or with energy off for the probe.
- **The MI estimates** (A1, A2) are taken on states that carry energy noise.
  The noise lowers the measured MI but does not create it, since the
  estimator adds its own unit noise per dimension.

As health: fixed-dose energy held test perplexity on a plateau over 10
self-trained generations, where greedy self-training degraded
([collapse sweep](../results/2026-11-23-collapse-site-sweep.md)). That is the
evidence for energy while awake.

## 6. Health measurements

In peridot, participation ratio did not tell stuck runs from healthy ones; the
freeze of q/k importance did, and it gated sleep. Here, PR, input share and
action entropy stay as logged measurements for switching states (sleep), not
as loss terms. They are especially worth watching when energy changes, since
more energy or exhaustion should move them.

## 7. The combination this points to

- **Representation:** the input chunk tether (section 3). The output tether is
  kept but flagged (section 3).
- **Policy:** state empowerment, estimated through the tether's world model,
  with common energy noise across probe actions, rewarded by PPO. Not the
  policy's own I(A; S').
- **Health:** row-norm renorm toward the starting model, energy at a fixed dose
  while awake, freeze-gated sleep later.

Added one piece at a time, as a ladder rather than a full grid:

| rung | adds | arm |
|---|---|---|
| R1 | input and output tether only | Tchunk_c (queued) |
| R1+E | the same columns and norm band with energy on, from the base model | FULLe_c (queued; FULL_c is its no-energy reference) |
| R2 | state empowerment reward | built (054a1c5, `--reward empower`); offline check first (below) |
| R3 | energy on R2 | |
| R2h | energy curiosity reward (section 8), before empowerment | built (17c6d50); queued after Tin3_e as Tin3_curio |
| R4 | freeze-gated sleep | |

Controls already queued, each with its prediction written down here first:

| arm | prediction |
|---|---|
| A1c | a static screen (pause, death or dialog) within the run |
| A2c | high-entropy actions with large mouse moves, no static screen |
| Tchunk_c | behaviour near the base model; world-model probe gain above base |
| FULLe_c | tether losses fall as in FULL_c with energy on; probe gain no lower than FULL_c |

**Check before R2:** on Tchunk_c's checkpoint, offline, the state-empowerment
estimate should be near 0 on pause, death and dialog frames and higher in the
open world. If it is not, the estimator or the world model is not good enough
yet, and R2 would only measure noise.

## 8. Energy homeostasis as a policy reward (user, 2026-11-27)

Bad energy states in the input columns should reach the output columns. This
is homeostatic RL (Keramati and Gutkin 2014): the reward is the reduction in
drive, where drive is how far an internal state sits from its set point.

    r_t = D(e_t) - D(e_t+1),  D = distance of the input-column neurons' energy from equilibrium

- **Static screen:** neurons that stay active exhaust, and silent ones need
  forced firing, so the energy sits far from equilibrium and D is high. Moving
  into a static screen gives negative reward, against the dark room.
- **Chaotic input:** overactivity, also far from the set point.
- **What gets rewarded:** actions that keep the input side in a healthy band, a
  drive toward a "just right" amount of stimulation. Energy then becomes a
  learning signal for the policy through the output columns, not only health.

**It is curiosity, and it comes before empowerment** (user). A better world
model is what curiosity is for, and empowerment needs a world model to mean
anything (the base and Tin_c checks: static screens scored the same as the
open world). Energy-based curiosity avoids both classic failures:

- **Too unpredictable or dangerous** (the noisy TV): over-driven neurons
  exhaust and shut off, which raises D.
- **Too predictable** (the couch potato): under-driven neurons need forced
  firing, which also raises D.
- **Neuron diversity:** lateral inhibition within a layer, a Gaussian
  max-kernel inhibition, or a variational term would make neurons represent
  different things, so no single static or noisy pattern keeps the whole
  input side engaged. That is a variant to add if the couch-potato failure
  shows up.

**Built** (17c6d50, `--reward energy`):

- D(frame) is the share of the MLP and attention neurons in layers 1–12 that
  energy forced to fire or shut off, averaged over the frame's image tokens.
- The PPO reward for action k is D(frame k) − D(frame k+1).
- Energy at a fixed dose of 1 on the interior and input neurons, none on the
  action choice, with the tether on layers 1–12.
- **Caveat:** energy self-corrects over time. Holding the energy state fixed
  when measuring the next frame would credit only the action's consequence.
  It is not done in v1; each frame is measured on its own tokens as it
  arrives.

## Where the literature runs out

No paper does live RL on a behaviour-cloned vision-language game agent with
any of these objectives. The pieces each have a literature:

- predictive-information robots;
- empowerment;
- curiosity, with its own dark-room and noisy-TV failures (Burda et al. 2018);
- self-predictive representations;
- loss of plasticity in continual learning (Dohare et al. 2024).

The combination, and the depth columns, are ours to test.
