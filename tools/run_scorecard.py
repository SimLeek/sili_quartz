"""One scorecard per run, the same for every node of the track A lineage
(evaluation only; reads RUN/actor.jsonl and RUN/stats.jsonl).

    python tools/run_scorecard.py --ref RUN_BASE RUN_A RUN_B ... [--out s.jsonl]

Per 10-minute window, from the parsed decisions:
- distinct actions / decisions (low = loops, ~1 = every decision different);
- keys used (of the 74) and key entropy in bits (repertoire);
- k80: how many keys cover 80 % of key usage (the untouched model: about 6 of
  ~30 used in 30 minutes, i.e. 80-20; a collapse goes toward 1, random spread
  past 30);
- distinct keys per decision and median mouse move (mashing shows as many
  different keys and large moves; a key held on every step counts once);
- mean log-prob per sampled token, as the actor sampled it (certainty);
- energy interventions per decision, when logged.
Game, from the stats log: play minutes, deaths, blocks mined, items picked up,
metres travelled (per hour of play), distinct block types, logs, crafting,
how far the tutorial got, and the within-run trend of the mining and pickup
rates (second half minus first half). Trend of the play metrics: last third of
the windows minus the first third (the first minutes are the base model, so
each run is its own control).

Against the reference run (the untouched model with grammar, Root_g):
- deviation: per window, the mean relative distance from the reference's mean
  over distinct ratio, keys, k80, key entropy, distinct keys per decision and
  mouse; the run's median deviation orders nodes;
- severe flags, for a metric at least 50 % off in the failing direction:
  "loops" (distinct ratio), "collapsed keys" (k80 or keys), "spread keys"
  (k80 or keys doubled), "mashing" (distinct keys per decision doubled, or
  mouse tripled). A healthy window has no severe flag.
"""

from __future__ import annotations

import argparse
import itertools
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
                    "k80": next((i + 1 for i, c in enumerate(itertools.accumulate(sorted(keys.values(), reverse=True)))
                                 if c >= 0.8 * n_k), 0) if n_k else 0,
                    "key_entropy": round(-sum(v / n_k * math.log2(v / n_k) for v in keys.values()), 2) if n_k else 0.0,
                    "keys_per_decision": round(sum(nk) / len(nk), 1),
                    "distinct_keys_per_decision": round(sum(uk) / len(uk), 2),
                    "mouse_median": mice[len(mice) // 2] if mice else None,
                    "token_logp": round(sum(lps) / len(lps), 3) if lps else None,
                    "energy_per_decision": round(sum(ei) / len(ei)) if ei else None})
    return out


TUTORIAL = ["movement", "find_tree", "punch_tree", "open_inventory", "craft_planks", "none"]


def game(run: Path) -> dict:
    """Capability, from the stats log: rates per hour of play, distinct block types, crafting, how far the
    tutorial got, and the within-run trend (second half of the run against the first)."""
    p = run / "stats.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(l)["stats"] for l in open(p)]
    last = rows[-1]
    hours = max(last.get("play_ticks", 0) / 72000, 1e-6)
    tut = (last.get("options_changed") or {}).get("tutorialStep")
    step = tut[1] if tut else "punch_tree"
    out = {"play_min": round(last.get("play_ticks", 0) / 1200, 1),
           "deaths_per_h": round(last.get("deaths", 0) / hours, 1),
           "mined_per_h": round(last.get("mined", 0) / hours, 1),
           "picked_per_h": round(last.get("picked_up", 0) / hours, 1),
           "metres_per_h": round(last.get("distance_cm", 0) / 100 / hours),
           "distinct_blocks": last.get("distinct_blocks_mined", 0),
           "logs": last.get("logs_mined", 0), "crafted": last.get("crafted", 0),
           "planks": last.get("planks_crafted", 0), "crafting_tables": last.get("crafting_tables_crafted", 0),
           "tutorial": TUTORIAL.index(step) if step in TUTORIAL else None}
    mid = next((r for r in rows if r.get("play_ticks", 0) >= last.get("play_ticks", 0) / 2), None)
    if mid is not None and 0 < mid.get("play_ticks", 0) < last.get("play_ticks", 0):
        h1, h2 = mid["play_ticks"] / 72000, (last["play_ticks"] - mid["play_ticks"]) / 72000
        out["mined_per_h_trend"] = round((last.get("mined", 0) - mid.get("mined", 0)) / h2 - mid.get("mined", 0) / h1, 1)
        out["picked_per_h_trend"] = round((last.get("picked_up", 0) - mid.get("picked_up", 0)) / h2
                                          - mid.get("picked_up", 0) / h1, 1)
    return out


def pauses(run: Path) -> dict:
    """The dark-room signal. Energy curiosity should visit the dark room and leave on its own (where it settles
    between dark room and chaos depends on its settings and on how predictable the world is); a failing method
    stays. So the dwell matters more than the count: still-screen episodes, how long, and whether the policy left
    on its own. Also how often the policy paused the game (the game log's "Saving and pausing game" lines
    during the run, per hour of the session), how often the watcher found it parked in the pause menu and pressed Esc,
    and whether a hang ended the run (copied logs: game_latest.log, mc_watch.log; the game log covers the session
    since the game's last launch)."""
    out = {}
    try:
        rows = [json.loads(l) for l in open(run / "actor.jsonl")]
        t0, t1 = rows[0]["t"], rows[-1]["t"]
    except (OSError, IndexError, ValueError, KeyError):
        return out
    gl = run / "game_latest.log"
    if gl.exists():
        import time as _time
        n = 0
        for line in gl.read_text(errors="replace").splitlines():
            if "Saving and pausing game" in line and line.startswith("["):
                try:
                    t = _time.mktime(_time.strptime(line[1:19], "%d%b%Y %H:%M:%S"))
                except ValueError:
                    continue
                n += t0 <= t <= t1
        out["pauses_per_h"] = round(n / max((t1 - t0) / 3600, 1e-6), 1)
    wl = run / "mc_watch.log"
    if wl.exists():
        ev = []
        for line in wl.read_text(errors="replace").splitlines():
            try:
                ev.append(json.loads(line))
            except ValueError:
                pass
        out["esc_resumes"] = sum(1 for e in ev if str(e.get("event", "")).startswith("paused: pressed Esc"))
        out["ended_by_hang"] = any(str(e.get("event", "")).startswith("hang:") for e in ev)
        # dark-room dwell: still-screen episodes (>= 30 s) during the run, and how they ended
        still = sorted((e for e in ev if e.get("event") == "still episode" and t0 <= e.get("t", 0) <= t1 + 300),
                       key=lambda e: e["seconds"])
        if still:
            out["still_episodes"] = len(still)
            out["still_median_s"] = still[len(still) // 2]["seconds"]
            out["still_max_s"] = still[-1]["seconds"]
            out["still_self_exit_share"] = round(sum(e["ended_by"] == "self" for e in still) / len(still), 3)
    return out


def trend(ws: list[dict]) -> dict:
    """Within-run trend of the play metrics: mean of the last third of the windows minus the first third."""
    if len(ws) < 3:
        return {}
    n = max(1, len(ws) // 3)
    out = {}
    for f in ("distinct_ratio", "k80", "key_entropy", "distinct_keys_per_decision", "deviation"):
        a = [w[f] for w in ws[:n] if w.get(f) is not None]
        b = [w[f] for w in ws[-n:] if w.get(f) is not None]
        if a and b:
            out[f] = round(sum(b) / len(b) - sum(a) / len(a), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--ref", type=Path, required=True, help="reference run (the untouched model)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--minutes", type=int, default=10, help="window length")
    a = ap.parse_args()
    ref = windows(a.ref, a.minutes)
    METRICS = ("distinct_ratio", "keys", "k80", "key_entropy", "distinct_keys_per_decision", "mouse_median")
    mean = {f: sum((w[f] or 0) for w in ref) / len(ref) for f in METRICS}
    for run in [a.ref] + list(a.runs):
        ws = windows(run, a.minutes)
        for w in ws:
            rel = {f: ((w[f] or 0) - mean[f]) / mean[f] if mean[f] else 0.0 for f in METRICS}
            # graded: mean relative distance from the reference; flags only for metrics >= 50 % off, in the failing direction
            w["deviation"] = round(sum(abs(v) for v in rel.values()) / len(rel), 3)
            flags = []
            if rel["distinct_ratio"] <= -0.5:
                flags.append("loops")
            if rel["k80"] <= -0.5 or rel["keys"] <= -0.5:
                flags.append("collapsed keys")
            if rel["k80"] >= 1.0 or rel["keys"] >= 1.0:
                flags.append("spread keys")
            if rel["distinct_keys_per_decision"] >= 1.0 or rel["mouse_median"] >= 2.0:
                flags.append("mashing")
            w["flags"] = flags
        healthy = sum(1 for w in ws if not w["flags"])
        devs = sorted(w["deviation"] for w in ws)
        row = {"run": run.name, "windows": len(ws), "healthy_windows": healthy,
               "median_deviation": devs[len(devs) // 2] if devs else None,
               "first_flag_minute": next((w["minute"] for w in ws if w["flags"]), None),
               "game": {**game(run), **pauses(run)}, "trend": trend(ws), "per_window": ws}
        print(f"{run.name:24s} healthy {healthy}/{len(ws)} windows, median deviation {row['median_deviation']}, "
              f"first flag at minute {row['first_flag_minute']}, "
              f"game {row['game']}")
        for w in ws:
            print(f"    {w['minute']:4d}  distinct {w['distinct_ratio']:.2f}  keys {w['keys']:3d}  k80 {w['k80']:2d}  H {w['key_entropy']:.2f}  "
                  f"dev {w['deviation']:.2f}  k/dec {w['keys_per_decision']:5.1f} uk/dec {w['distinct_keys_per_decision']:4.2f}  mouse {w['mouse_median']}  logp {w['token_logp']}  {','.join(w['flags'])}")
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
