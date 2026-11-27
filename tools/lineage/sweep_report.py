"""Robustness of the winner to its main knobs: learning rate and training-energy dose at 1/4x, 1x, 4x.
usage: python3 sweep_report.py WINNER -> one line per knob: coherence and health at each value, and the worst case.
A method that only works at one value is fragile; report the worst case, not the best."""
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
            "mined_per_h": sc.get("game", {}).get("mined_per_h")}
w = sys.argv[1]
for knob, tags in (("lr", (f"{w}_lrq", w, f"{w}_lr4")), ("dose", (f"{w}_doseq", w, f"{w}_dose4"))):
    s = [score(t) for t in tags]
    cohs = [x["coherence"] for x in s if x and x["coherence"] is not None]
    print(f"SWEEP {w} {knob} 1/4x,1x,4x: {s}; worst coherence {min(cohs) if cohs else None}; "
          f"healthy at all values {all(x and x['healthy'] for x in s)}")
