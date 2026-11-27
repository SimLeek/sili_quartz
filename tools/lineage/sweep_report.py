"""The winner's knob sweep, reported as the whole spread (learning rate and training-energy dose at 1/4x, 1x, 4x).
usage: python3 sweep_report.py WINNER -> per knob: every point, best, worst, range, whether the best is at an edge.
Worst case = robustness; best = optimality; both matter. If one value is best on every metric, and stays best in
other contexts, it is a setpoint, not a knob. A best at an edge means the sweep should extend past it."""
import json, os, sys
E = os.path.expanduser("~/runs/eval")
BASE_KEYS, BASE_DRIVE, BASE_OVERLAP = 13, 18.57, 0.188
def last(p):
    return json.loads(open(p).readlines()[-1]) if os.path.exists(p) and os.path.getsize(p) else None
def score(tag):
    sc, rp, fa = (last(f"{E}/{tag}_{k}.jsonl") for k in ("scorecard", "repertoire", "fanin"))
    if sc is None:
        return None
    ok = sc["healthy_windows"] / max(1, sc["windows"]) >= 0.5
    if rp:
        ok &= rp["distinct_keys"] >= 0.7 * BASE_KEYS and rp["lb_share"] <= 0.6
    if fa:
        ok &= not (fa["mean_drive"] < 0.9 * BASE_DRIVE and fa["mean_overlap"] > BASE_OVERLAP + 0.01)
    return {"healthy": ok, "coherence": (rp or {}).get("coherence"), "deviation": sc.get("median_deviation"),
            "mined_per_h": sc.get("game", {}).get("mined_per_h"), "distinct_blocks": sc.get("game", {}).get("distinct_blocks")}
# metric -> True if higher is better
METRICS = {"coherence": True, "deviation": False, "mined_per_h": True, "distinct_blocks": True}
w = sys.argv[1]
for knob, tags in (("lr", (f"{w}_lrq", w, f"{w}_lr4")), ("dose", (f"{w}_doseq", w, f"{w}_dose4"))):
    pts = dict(zip(("1/4x", "1x", "4x"), (score(t) for t in tags)))
    print(f"SWEEP {w} {knob}: " + "; ".join(f"{k}={v}" for k, v in pts.items()))
    bests = {}
    for m, up in METRICS.items():
        vals = {k: v[m] for k, v in pts.items() if v and v.get(m) is not None}
        if len(vals) < 2:
            continue
        best = (max if up else min)(vals, key=vals.get)
        worst = (min if up else max)(vals, key=vals.get)
        bests[m] = best
        edge = " (best at an edge: extend the sweep)" if best in ("1/4x", "4x") else ""
        print(f"  {m}: best {best} {vals[best]}, worst {worst} {vals[worst]}, range {abs(vals[best] - vals[worst]):.4g}{edge}")
    agree = len(set(bests.values())) == 1 if bests else False
    healthy_all = all(v and v["healthy"] for v in pts.values())
    print(f"  metrics agree on the best value: {agree} ({bests}); healthy at every value: {healthy_all}")
