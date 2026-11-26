"""Repetition in the model's reasoning-probe thoughts (evaluation only).

    python tools/probe_repetition.py RUN/probes.jsonl [--window 1800]

Per run, and per time window:
- thoughts: probes that thought at all (most samples skip thinking);
- distinct3: unique word 3-grams / all 3-grams (low = repetitive);
- reuse4: mean share of a thought's word 4-grams already seen in an earlier
  thought of the run (high = it keeps saying the same things);
- the most repeated sentences and 3-word phrases.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def ngrams(ws: list[str], n: int) -> list[tuple]:
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def sentences(text: str) -> list[str]:
    return [" ".join(words(s)) for s in re.split(r"[.!?]\s+", text) if len(words(s)) >= 4]


def summarize(thoughts: list[str]) -> dict:
    tri, seen4, reuse = [], set(), []
    for t in thoughts:
        ws = words(t)
        tri += ngrams(ws, 3)
        g4 = ngrams(ws, 4)
        if g4:
            reuse.append(sum(g in seen4 for g in g4) / len(g4))
        seen4.update(g4)
    return {"thoughts": len(thoughts),
            "distinct3": round(len(set(tri)) / max(len(tri), 1), 3),
            "reuse4": round(sum(reuse) / max(len(reuse), 1), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("probes", type=Path)
    ap.add_argument("--window", type=float, default=1800.0)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.probes)]
    rows = [r for r in rows if r.get("thought")]
    if not rows:
        raise SystemExit("no thoughts")
    t0 = rows[0]["t"]
    out = {"run": summarize([r["thought"] for r in rows]), "windows": []}
    bins = collections.defaultdict(list)
    for r in rows:
        bins[int((r["t"] - t0) // a.window)].append(r["thought"])
    for b in sorted(bins):
        out["windows"].append({"start_min": round(b * a.window / 60), **summarize(bins[b])})
    sent = collections.Counter(s for r in rows for s in set(sentences(r["thought"])))
    phr = collections.Counter(" ".join(g) for r in rows for g in set(ngrams(words(r["thought"]), 3)))
    out["top_sentences"] = [[c, s] for s, c in sent.most_common(a.top) if c > 1]
    out["top_phrases"] = [[c, p] for p, c in phr.most_common(a.top)]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
