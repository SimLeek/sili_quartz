"""Noise floor for the lineage decisions; writes ~/runs/eval/noise_floor.json.
- play level: the spread (max - min) of each scorecard and game metric across the untouched runs Root_g, Root_g2,
  Root_g3 (same program, different sessions);
- probe level: the standard deviation of the repertoire probe's metrics across sampling seeds on fixed weights
  (eval/base_probe_seeds.jsonl).
A difference smaller than the floor is not evidence."""
import json, os, statistics
E = os.path.expanduser("~/runs/eval")
def last(p):
    return json.loads(open(p).readlines()[-1]) if os.path.exists(p) and os.path.getsize(p) else None
roots = [r for r in (last(f"{E}/{t}_scorecard.jsonl") for t in ("Root_g", "Root_g2", "Root_g3")) if r]
play = {}
if len(roots) >= 2:
    vals = {"median_deviation": [r["median_deviation"] for r in roots],
            "healthy_fraction": [r["healthy_windows"] / max(1, r["windows"]) for r in roots]}
    for k in ("mined_per_h", "picked_per_h", "deaths_per_h", "metres_per_h", "distinct_blocks",
              "mined_per_h_trend", "picked_per_h_trend"):
        v = [r["game"].get(k) for r in roots if r.get("game", {}).get(k) is not None]
        if len(v) >= 2:
            vals[k] = v
    play = {k: {"spread": round(max(v) - min(v), 4), "mean": round(sum(v) / len(v), 4)} for k, v in vals.items()}
probe = {}
p = f"{E}/base_probe_seeds.jsonl"
if os.path.exists(p):
    rows = [json.loads(l) for l in open(p)]
    for k in ("coherence", "distinct_keys", "lb_share", "key_entropy_bits"):
        v = [r[k] for r in rows if r.get(k) is not None]
        if len(v) >= 2:
            probe[k] = {"sd": round(statistics.stdev(v), 4), "mean": round(sum(v) / len(v), 4)}
out = {"roots": len(roots), "play": play, "probe": probe}
json.dump(out, open(f"{E}/noise_floor.json", "w"), indent=1)
print(json.dumps(out))
