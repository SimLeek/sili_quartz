"""Random-init checks: each trained node against the same random start with no training.
usage: python3 randinit_report.py CONTROL TRAINED [TRAINED...] -> the scorecard trend, coherence, keys, k80 and game
rates of each, so improvement *from the random start* is visible (the base model is behaviour-cloned and near optimal,
so beating it is not the test here)."""
import json, os, sys
E = os.path.expanduser("~/runs/eval")
def last(p):
    return json.loads(open(p).readlines()[-1]) if os.path.exists(p) and os.path.getsize(p) else None
for tag in sys.argv[1:]:
    sc, rp = last(f"{E}/{tag}_scorecard.jsonl"), last(f"{E}/{tag}_repertoire.jsonl")
    if sc is None:
        print(f"RANDINIT {tag}: no run"); continue
    k80 = [w.get("k80") for w in sc.get("per_window", [])]
    print(f"RANDINIT {tag}: deviation {sc.get('median_deviation')} trend {sc.get('trend')} k80 per window {k80} "
          f"| probe coherence {(rp or {}).get('coherence')} keys {(rp or {}).get('distinct_keys')} "
          f"lb {(rp or {}).get('lb_share')} | game {sc.get('game')}")
