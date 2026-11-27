import Base

# LAWS: per-neuron energy cannot see joint (episodic) novelty.
#
# Energy (sili_quartz energy_scan, no noise, before thresholds) keeps one leaky
# state per neuron:  e(t+1) = (1 - d) e(t) + C (mu - |x(t)|).  Unrolled, a
# neuron's state is a weighted sum of ITS OWN past activity, so any reward
# computed from energy states and the current input sees the history only
# through those per-neuron weighted sums. Here the forgetting is d = 1/2,
# scaled to whole numbers: a 3-frame history weighs frames 4, 2, 1 (oldest
# first). Two input features, so a frame is (x1, x2).
#
# Episodic novelty (nearest neighbour in a memory of whole frames) compares
# the whole current frame with whole past frames: L1 distance to the nearest
# remembered frame.
#
# History A = (1,0), (0,0), (0,4);  history B = (0,1), (0,0), (4,0).
# Both features have weighted sum 4 in both histories, so energy is in the
# same state after A and after B (laws 1, 2) and must respond identically to
# any next frame. For the test frame (4,0), episodic novelty is 3 after A
# and 0 after B (laws 3, 4). So input-neuron energy is not a special case of
# episodic novelty; internal energy sees joint novelty only through neurons
# that encode the conjunction.

# a neuron's energy-relevant history: weighted activity, weights 4, 2, 1
def Nov.state(x0: Nat, x1: Nat, x2: Nat) -> Nat:
  Nat.add(Nat.mul(4n, x0), Nat.add(Nat.mul(2n, x1), x2))

def Nov.absdiff(a: Nat, b: Nat) -> Nat:
  match a:
    case 0n:
      b
    case 1n+p:
      match b:
        case 0n:
          1n+p
        case 1n+q:
          Nov.absdiff(p, q)

def Nov.min(a: Nat, b: Nat) -> Nat:
  match a:
    case 0n:
      0n
    case 1n+p:
      match b:
        case 0n:
          0n
        case 1n+q:
          1n+Nov.min(p, q)

def Nov.dist(a1: Nat, a2: Nat, b1: Nat, b2: Nat) -> Nat:
  Nat.add(Nov.absdiff(a1, b1), Nov.absdiff(a2, b2))

# distance from the test frame (t1, t2) to the nearest of three remembered frames
def Nov.novelty(+t1: Nat, +t2: Nat, f1a: Nat, f1b: Nat, f2a: Nat, f2b: Nat, f3a: Nat, f3b: Nat) -> Nat:
  Nov.min(Nov.dist(t1, t2, f1a, f1b), Nov.min(Nov.dist(t1, t2, f2a, f2b), Nov.dist(t1, t2, f3a, f3b)))

# LAW 1. Feature 1's energy state is the same after A (1, 0, 0) and B (0, 0, 4).
law same_state_feature_1:
  {Nov.state(1n, 0n, 0n) == Nov.state(0n, 0n, 4n) : Nat}

# LAW 2. Feature 2's energy state is the same after A (0, 0, 4) and B (1, 0, 0).
law same_state_feature_2:
  {Nov.state(0n, 0n, 4n) == Nov.state(1n, 0n, 0n) : Nat}

# LAW 3. Episodic novelty of (4,0) after A is 3.
law novelty_after_A:
  {Nov.novelty(4n, 0n, 1n, 0n, 0n, 0n, 0n, 4n) == 3n : Nat}

# LAW 4. Episodic novelty of (4,0) after B is 0: it was seen.
law novelty_after_B:
  {Nov.novelty(4n, 0n, 0n, 1n, 0n, 0n, 4n, 0n) == 0n : Nat}
