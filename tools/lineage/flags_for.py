"""Trainer and actor flags of a drive-generation node, from its tag (singles, or combinations joined by '+').
usage: python3 flags_for.py TAG -> "TX|AX" (trainer extra flags | actor extra flags), without --init."""
import sys
BASE = "--health=pr --grammar --tether=chunks --tether-chunks=3 --w-rep=0 --actor-dose=0"
SINGLE = {"Tin3_et_curio": ("energy", "", ""),
          "curio_input": ("energy", "--reward-sites=input --energy-io --energy-io-side=in", "--energy-io --energy-io-side in"),
          "curio_all": ("energy", "--reward-layers=24", ""),
          "episodic_int": ("episodic", "", ""),
          "rnd_int": ("rnd", "", "")}
parts = sys.argv[1].split("+")
reward = ",".join(SINGLE[p][0] for p in parts)
tx = " ".join(x for p in parts for x in [SINGLE[p][1]] if x)
ax = " ".join(x for p in parts for x in [SINGLE[p][2]] if x)
print(f"{BASE} --reward={reward} {tx}".strip() + "|" + f"--grammar {ax}".strip())
