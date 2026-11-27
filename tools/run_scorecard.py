"""One scorecard per run, the same for every node of the track A lineage
(evaluation only; reads RUN/actor.jsonl and RUN/stats.jsonl).

    python tools/run_scorecard.py --ref RUN_BASE RUN_A RUN_B ... [--out s.jsonl]

Per 10-minute window, from the parsed decisions:
- distinct actions / decisions (low = loops, ~1 = every decision different);
- keys used (of the 74) and key entropy in bits (repertoire);
- distinct keys per decision and median mouse move (mashing shows as many
  different keys and large moves; a key held on every step counts once);
- mean log-prob per sampled token, as the actor sampled it (certainty);
- energy interventions per decision, when logged.
Game, from the stats log: play minutes, deaths, blocks mined, items picked up,
metres travelled, per hour of play.

Health flags against the reference run's own windows (outside the
reference's min..max range): "loops", "mashing", "collapsed keys",
"spread keys". A run with no flags in a window plays like the reference
there.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path


def windows(run: Path, minutes: int = 10) -> list[dict]:
    rows = [json.loads(l) for l in open(run / "actor.jsonl")]
    rows = [r for r in rows if r.get("parsed")]
    if not rows:
        return []
    t0 = rows[0]["t"]
    by = collections.defaultdict(list)
    for r in rows:
        by[int((r["t"] - t0) // (60 * minutes))].append(r)
    out = []
    for k in sorted(by):
        rs = by[k]
        keys, mice, nk, lps, ei = collections.Counter(), [], [], [], []
        uk = []
        for r in rs:
            body = re.sub(r"<\|action_(start|end)\|>", "", r["text"])
            parts = body.split(";")
            try:
                n = parts[0].split()
                mice.append(abs(float(n[0])) + abs(float(n[1])))
            except (ValueError, IndexError):
                pass
            ks = [x for p in parts[1:] for x in p.split()]
            keys.update(ks)
            nk.append(len(ks))
            uk.append(len(set(ks)))
            if r.get("sampled_logp"):
                lps.append(sum(r["sampled_logp"]) / len(r["sampled_logp"]))
            if r.get("energy_intervened") is not None:
                ei.append(r["energy_intervened"])
        n_k = sum(keys.values())
        mice.sort()
        out.append({"minute": k * minutes, "decisions": len(rs),
                    "distinct_ratio": round(len({r["text"] for r in rs}) / len(rs), 3),
                    "keys": len(keys),
                    "key_entropy": round(-sum(v / n_k * math.log2(v / n_k) for v in keys.values()), 2) if n_k else 0.0,
                    "keys_per_decision": round(sum(nk) / len(nk), 1),
                    "distinct_keys_per_decision": round(sum(uk) / len(uk), 2),
                    "mouse_median": mice[len(mice) // 2] if mice else None,
                    "token_logp": round(sum(lps) / len(lps), 3) if lps else None,
                    "energy_per_decision": round(sum(ei) / len(ei)) if ei else None})
    return out


def game(run: Path) -> dict:
    p = run / "stats.jsonl"
    if not p.exists():
        return {}
    last = [json.loads(l) for l in open(p)][-1]["stats"]
    hours = max(last.get("play_ticks", 0) / 72000, 1e-6)
    return {"play_min": round(last.get("play_ticks", 0) / 1200, 1),
            "deaths_per_h": round(last.get("deaths", 0) / hours, 1),
            "mined_per_h": round(last.get("mined", 0) / hours, 1),
            "picked_per_h": round(last.get("picked_up", 0) / hours, 1),
            "metres_per_h": round(last.get("distance_cm", 0) / 100 / hours)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--ref", type=Path, required=True, help="reference run (the untouched model)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--minutes", type=int, default=10, help="window length")
    a = ap.parse_args()
    ref = windows(a.ref, a.minutes)
    rng = {f: (min(w[f] for w in ref), max(w[f] for w in ref))
           for f in ("distinct_ratio", "keys", "key_entropy", "distinct_keys_per_decision")}
    mouse = [w["mouse_median"] for w in ref if w["mouse_median"] is not None]
    rng["mouse_median"] = (min(mouse), max(mouse))
    for run in [a.ref] + list(a.runs):
        ws = windows(run, a.minutes)
        for w in ws:
            flags = []
            if w["distinct_ratio"] < rng["distinct_ratio"][0]:
                flags.append("loops")
            if w["keys"] < rng["keys"][0] or w["key_entropy"] < rng["key_entropy"][0]:
                flags.append("collapsed keys")
            if w["keys"] > rng["keys"][1] or w["key_entropy"] > rng["key_entropy"][1]:
                flags.append("spread keys")
            if (w["mouse_median"] or 0) > rng["mouse_median"][1] or \
                    w["distinct_keys_per_decision"] > rng["distinct_keys_per_decision"][1]:
                flags.append("mashing")
            w["flags"] = flags
        healthy = sum(1 for w in ws if not w["flags"])
        row = {"run": run.name, "windows": len(ws), "healthy_windows": healthy,
               "first_flag_minute": next((w["minute"] for w in ws if w["flags"]), None),
               "game": game(run), "per_window": ws}
        print(f"{run.name:24s} healthy {healthy}/{len(ws)} windows, first flag at minute {row['first_flag_minute']}, "
              f"game {row['game']}")
        for w in ws:
            print(f"    {w['minute']:4d}  distinct {w['distinct_ratio']:.2f}  keys {w['keys']:3d}  H {w['key_entropy']:.2f}  "
                  f"k/dec {w['keys_per_decision']:5.1f} uk/dec {w['distinct_keys_per_decision']:4.2f}  mouse {w['mouse_median']}  logp {w['token_logp']}  {','.join(w['flags'])}")
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
