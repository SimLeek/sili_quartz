"""Pick the winner among sibling nodes (same parent, one drive each); reruns TAG_s1... are averaged with TAG.
usage: python3 select_sibling.py TAG1 TAG2 ...  -> prints "WINNER <tag>" (or "WINNER none") and one line per sibling.
Health (absolute, against the untouched model): at least half the scorecard windows healthy (no severe flag vs
Root_g), repertoire distinct keys >= 0.7 x 13 and LB share <= 0.6, no active fan-in damage pattern. Among the healthy,
the highest input coherence wins; ties go to the smaller median deviation."""
import json, os, sys
E = os.path.expanduser("~/runs/eval")
BASE_KEYS, BASE_DRIVE, BASE_OVERLAP = 13, 18.57, 0.188
def load(tag, kind):
    p = f"{E}/{tag}_{kind}.jsonl"
    return json.loads(open(p).readlines()[-1]) if os.path.exists(p) and os.path.getsize(p) else None
import re
groups = {}  # a node and its reruns (TAG_s1, TAG_s2, ...) are averaged
for tag in sys.argv[1:]:
    groups.setdefault(re.sub(r"_s\d+$", "", tag), []).append(tag)
rows = []
for base, tags in groups.items():
    vals = []
    for tag in tags:
        sc, rp, fa = load(tag, "scorecard"), load(tag, "repertoire"), load(tag, "fanin")
        if sc is None:
            continue
        healthy = sc["healthy_windows"] / max(1, sc["windows"]) >= 0.5
        if rp:
            healthy &= rp["distinct_keys"] >= 0.7 * BASE_KEYS and rp["lb_share"] <= 0.6
        if fa:
            healthy &= not (fa["mean_drive"] < 0.9 * BASE_DRIVE and fa["mean_overlap"] > BASE_OVERLAP + 0.01)
        vals.append((healthy, (rp or {}).get("coherence", 0.0) or 0.0, sc.get("median_deviation") or 99.0))
        print(f"  {tag}: healthy={healthy} coherence={vals[-1][1]} deviation={vals[-1][2]} "
              f"windows={sc['healthy_windows']}/{sc['windows']} keys={(rp or {}).get('distinct_keys')} "
              f"lb={(rp or {}).get('lb_share')} drive={(fa or {}).get('mean_drive')}")
    if not vals:
        print(f"  {base}: no run"); continue
    rows.append((all(v[0] for v in vals), sum(v[1] for v in vals) / len(vals), -sum(v[2] for v in vals) / len(vals), base))
ok = sorted([r for r in rows if r[0]], reverse=True)
print("WINNER " + (ok[0][3] if ok else "none"))
print("RUNNERUP " + (ok[1][3] if len(ok) > 1 else "none"))
