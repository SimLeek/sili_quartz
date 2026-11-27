# Energy vs episodic novelty (track A drives)

Question (2026-11-27): is episodic novelty just input-neuron energy tuned a
certain way, and is RND a subset of energy on an internal model?

**What energy computes.** From `sili_quartz/energy_scan.py` (no noise, before
thresholds): e(t+1) = (1 − d) e(t) + C (μ − |x(t)|). Unrolled, each neuron's
state is C Σ_{s<t} (1 − d)^{t−1−s} (μ − |x(s)|): a leaky sum of its *own*
activity against a fixed set point. The state is per neuron. A reward
computed from energy states and the current input therefore sees the history
only through each neuron's own weighted sum.

**Laws (`LAWS.bend`, checked with `bend PROOF.bend --check-only`).**

- **Setup:** two input features, forgetting ½ (frame weights 4, 2, 1).
  - History A = (1,0), (0,0), (0,4).
  - History B = (0,1), (0,0), (4,0).
- **Laws 1–2:** every feature's energy state is identical after A and B, so
  energy responds identically to any next frame.
- **Laws 3–4:** episodic novelty (L1 distance to the nearest remembered frame)
  of the test frame (4,0) is 3 after A and 0 after B.
- **Check of the check:** a deliberately false variant (novelty after B = 1)
  fails.
- **Against the real kernel** (`energy_scan_reference`, decay ½): histories
  (1,0), (0,0), (0,0.25) and (0,1), (0,0), (0.25,0) leave both features at
  0.075 and give the identical response to the test frame (1,0); episodic
  novelty is 0 and 0.75.

**Conclusions.**

1. **Input-neuron energy is not a special case of episodic novelty.** It is
   factorized, per-feature habituation with forgetting. Episodic novelty is
   joint.
2. **Internal energy sees joint novelty only through neurons that encode the
   conjunction.** Deeper neurons represent concepts, so internal energy
   should seek novel *situations* where input energy seeks novel pixels
   (user). That is better, but bounded by what the representation encodes.
3. **RND is not a strict subset either.** Its predictor generalizes across
   similar inputs (a smooth count over joint inputs); energy habituates each
   neuron only to its own activity, with no generalization across neurons.
   They are different novelty estimators.
4. **What does hold:** energy covers factorized, feature-level novelty over a
   learned representation. Joint episodic memory and generalizing
   (RND-style) novelty are outside it.
