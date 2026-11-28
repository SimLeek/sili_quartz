"""Advance or reject a lineage node against its parent (track A evolution rule).
usage: python3 decide_node.py CHILD PARENT  -> prints ADVANCE or REJECT and the reasons.
Scorecard vs Root_g: no smaller share of healthy (no severe flag) windows than the parent, and a median deviation
no worse than the parent plus 0.05. Repertoire: distinct keys >= 0.7 x base, LB share <= 0.6. Active fan-in: no
damage pattern (drive < 0.9 x base together with overlap > base + 0.01)."""
import json, sys, os
E = os.path.expanduser("~/runs/eval")
BASE_KEYS, BASE_DRIVE, BASE_OVERLAP = 13, 18.57, 0.188
def load(tag, kind):
    p = f"{E}/{tag}_{kind}.jsonl"
    return json.loads(open(p).readlines()[-1]) if os.path.exists(p) and os.path.getsize(p) else None
child, parent = sys.argv[1], sys.argv[2]
sc, scp = load(child, "scorecard"), load(parent, "scorecard")
rp, fa = load(child, "repertoire"), load(child, "fanin")
if sc is None:
    print("REJECT no run"); sys.exit(0)
reasons, ok = [], True
h = sc["healthy_windows"] / max(1, sc["windows"])
hp = scp["healthy_windows"] / max(1, scp["windows"]) if scp else 0.0
dv, dvp = sc.get("median_deviation"), (scp or {}).get("median_deviation")
reasons.append(f"healthy {sc['healthy_windows']}/{sc['windows']} (parent {hp:.2f}), deviation {dv} (parent {dvp})")
ok &= h >= hp and (dvp is None or dv is None or dv <= dvp + 0.05)
if rp:
    reasons.append(f"repertoire keys {rp['distinct_keys']} LB {rp['lb_share']}")
    ok &= rp["distinct_keys"] >= 0.7 * BASE_KEYS and rp["lb_share"] <= 0.6
if fa:
    damage = fa["mean_drive"] < 0.9 * BASE_DRIVE and fa["mean_overlap"] > BASE_OVERLAP + 0.01
    reasons.append(f"fanin drive {fa['mean_drive']} overlap {fa['mean_overlap']} damage={damage}")
    ok &= not damage
print(("ADVANCE " if ok else "REJECT ") + "; ".join(reasons))
