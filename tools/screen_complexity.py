"""Screen complexity of a play session, from a 1 fps frame recording (evaluation only).

    python tools/screen_complexity.py RUN/screen [--window 60] [--history 600] [--out RUN/screen_complexity.csv]

Per window (default one minute of frames):
- spatial: mean JPEG size of a frame, in KB. Detailed screens (JEI item pages,
  daylight terrain) score high; a dark night or a plain menu, low.
- novelty: extra bytes the window's frames add when compressed after the
  previous `history` seconds (lzma, a long-range compressor), per frame.
  Revisiting the same screen (opening the same menu again and again) adds
  almost nothing; new places and new screens add a lot.
- static: share of frames nearly identical to the one before (stuck, paused,
  or staring at a still menu).

Frames are compared as 160x90 grayscale, so compression measures layout and
content, not JPEG noise.
"""

from __future__ import annotations

import argparse
import csv
import lzma
from pathlib import Path

import numpy as np
from PIL import Image


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((160, 90), Image.BILINEAR), dtype=np.uint8)


def compressed(data: bytes) -> int:
    return len(lzma.compress(data, preset=6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--window", type=int, default=60, help="frames per window (1 fps: seconds)")
    ap.add_argument("--history", type=int, default=600, help="frames of history for novelty")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    paths = sorted(a.frames.glob("*.jpg"))
    if len(paths) < a.window:
        raise SystemExit(f"only {len(paths)} frames")
    sizes = np.array([p.stat().st_size for p in paths]) / 1024
    frames = [load(p) for p in paths]
    rows = []
    for start in range(0, len(frames) - a.window + 1, a.window):
        end = start + a.window
        hist = b"".join(f.tobytes() for f in frames[max(0, start - a.history):start])
        win = b"".join(f.tobytes() for f in frames[start:end])
        base = compressed(hist) if hist else 0
        novelty = (compressed(hist + win) - base) / a.window
        diffs = [np.abs(frames[i].astype(int) - frames[i - 1]).mean() for i in range(max(start, 1), end)]
        rows.append({
            "minute": start // 60, "start_frame": start,
            "spatial_kb": round(float(sizes[start:end].mean()), 2),
            "novelty_bytes_per_frame": round(novelty, 1),
            "static_share": round(float(np.mean([d < 1.0 for d in diffs])), 3),
        })
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
