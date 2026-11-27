import ./LAWS.bend as Laws
import Base

# All four laws are closed computations: both sides reduce to the same numeral.
def Laws.same_state_feature_1():
  {==}

def Laws.same_state_feature_2():
  {==}

def Laws.novelty_after_A():
  {==}

def Laws.novelty_after_B():
  {==}

def main() -> Nat:
  Laws.Nov.novelty(4n, 0n, 1n, 0n, 0n, 0n, 0n, 4n)
