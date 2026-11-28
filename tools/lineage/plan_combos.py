"""Plan the combination siblings from the single-drive siblings' evaluations.
usage: python3 plan_combos.py TAG...  -> one line per combination: "COMBO <tag>|<reward>|<trainer extra>|<actor extra>".
Only healthy singles are combined (same health gate as select_sibling.py). Energy settings cannot be combined with
each other (one energy module), so the pool is {best healthy energy setting by coherence, episodic_int, rnd_int}:
every pair, plus all three when all are healthy. Rewards are combined by RewardMix (standardized, equal weight)."""
import itertools, json, os, sys
E = os.path.expanduser("~/runs/eval")
BASE_KEYS, BASE_DRIVE, BASE_OVERLAP = 13, 18.57, 0.188
DRIVE = {"Tin3_et_curio": ("energy", "", ""),
         "curio_input": ("energy", "--reward-sites=input --energy-io --energy-io-side=in", "--energy-io --energy-io-side in"),
         "curio_all": ("energy", "--reward-layers=24", ""),
         "episodic_int": ("episodic", "", ""),
         "rnd_int": ("rnd", "", "")}
def load(tag, kind):
    p = f"{E}/{tag}_{kind}.jsonl"
    return json.loads(open(p).readlines()[-1]) if os.path.exists(p) and os.path.getsize(p) else None
def healthy(tag):
    sc, rp, fa = load(tag, "scorecard"), load(tag, "repertoire"), load(tag, "fanin")
    if sc is None:
        return False, 0.0
    ok = sc["healthy_windows"] / max(1, sc["windows"]) >= 0.5
    if rp:
        ok &= rp["distinct_keys"] >= 0.7 * BASE_KEYS and rp["lb_share"] <= 0.6
    if fa:
        ok &= not (fa["mean_drive"] < 0.9 * BASE_DRIVE and fa["mean_overlap"] > BASE_OVERLAP + 0.01)
    return ok, (rp or {}).get("coherence", 0.0) or 0.0
tags = [t for t in sys.argv[1:] if t in DRIVE]
good = {t: c for t in tags for ok, c in [healthy(t)] if ok}
print("HEALTHY " + " ".join(sorted(good)))
energy = sorted((c, t) for t, c in good.items() if DRIVE[t][0] == "energy")
pool = ([energy[-1][1]] if energy else []) + [t for t in ("episodic_int", "rnd_int") if t in good]
for r in (2, 3):
    for combo in itertools.combinations(pool, r):
        reward = ",".join(DRIVE[t][0] for t in combo)
        tx = " ".join(x for t in combo for x in [DRIVE[t][1]] if x)
        ax = " ".join(x for t in combo for x in [DRIVE[t][2]] if x)
        print(f"COMBO {'+'.join(combo)}|{reward}|{tx}|{ax}")
