"""Minecraft session control for track A (run on the machine with the game).

    python tools/minecraft.py status
    python tools/minecraft.py stop
    python tools/minecraft.py make-template     # once, from the existing world
    python tools/minecraft.py reset             # world back to its seed, fresh player
    python tools/minecraft.py launch [--windowed] [--wait 240]
    python tools/minecraft.py session           # stop, reset, launch, wait until in-world

Needs: Prism Launcher with one instance, DISPLAY/XAUTHORITY for the game's X
session (defaults below), xdotool, and nbtlib (pip install nbtlib).

Stop is SIGTERM then SIGKILL on the game's java process (the no-quit mod
blocks quitting from inside the game; /nwo_out in chat is the in-game way out).
A reset never keeps anything the game wrote: the template is level.dat with
the player removed and the clock at 0, plus datapacks/ and serverconfig/;
Minecraft regenerates terrain from the seed and spawns a new player at the
world spawn.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PRISM = Path(os.environ.get("PRISM_DIR", Path.home() / ".local/share/PrismLauncher"))
INSTANCE = os.environ.get("MC_INSTANCE", "1.20.1")
WORLD = os.environ.get("MC_WORLD", "World 1")
GAME = PRISM / "instances" / INSTANCE / "minecraft"
TEMPLATE = Path(os.environ.get("MC_TEMPLATE_DIR", Path.home() / "mc_templates")) / WORLD
KEEP = ("level.dat", "datapacks", "serverconfig", "icon.png")
X_ENV = {"DISPLAY": os.environ.get("DISPLAY", ":0"),
         "XAUTHORITY": os.environ.get("XAUTHORITY", f"/run/user/{os.getuid()}/lyxauth")}

# options.txt values for unattended runs: fullscreen, keep running (not
# pause) when the window loses focus, and read the mouse from the pointer
# (re-centered every frame) rather than raw XInput2 device motion, which
# ignores synthetic (XTest) moves -- robonet's mouse would do nothing.
SESSION_OPTIONS = {"fullscreen": "true", "pauseOnLostFocus": "false", "rawMouseInput": "false"}


def game_pids() -> list[int]:
    out = subprocess.run(["pgrep", "-f", f"java.*instances/{re.escape(INSTANCE)}/"],
                         capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def window() -> str | None:
    r = subprocess.run(["xdotool", "search", "--name", "^Minecraft"], capture_output=True,
                       text=True, env={**os.environ, **X_ENV})
    ids = r.stdout.split()
    if not ids:
        return None
    return subprocess.run(["xdotool", "getwindowname", ids[0]], capture_output=True, text=True,
                          env={**os.environ, **X_ENV}).stdout.strip()


LOG = GAME / "logs" / "debug.log"  # Forge: latest.log is sparse, debug.log has the server lines


def in_world() -> bool:
    """The player joined the integrated server since the last launch
    (launch() deletes the log first)."""
    if not LOG.exists():
        return False
    return "joined the game" in LOG.read_text(errors="replace")


def status() -> dict:
    return {"running": bool(game_pids()), "pids": game_pids(), "window": window(),
            "in_world": bool(game_pids()) and in_world(), "world": WORLD,
            "template": TEMPLATE.exists()}


def stop(grace_s: float = 15.0) -> int:
    pids = game_pids()
    for p in pids:
        os.kill(p, signal.SIGTERM)
    end = time.time() + grace_s
    while time.time() < end and game_pids():
        time.sleep(0.5)
    for p in game_pids():
        os.kill(p, signal.SIGKILL)
    while game_pids():
        time.sleep(0.2)
    return len(pids)


def make_template() -> None:
    import nbtlib
    if game_pids():
        sys.exit("stop the game first")
    src = GAME / "saves" / WORLD
    if TEMPLATE.exists():
        sys.exit(f"{TEMPLATE} exists; delete it first to rebuild")
    TEMPLATE.mkdir(parents=True)
    for name in KEEP:
        p = src / name
        if p.is_dir():
            shutil.copytree(p, TEMPLATE / name)
        elif p.exists():
            shutil.copy2(p, TEMPLATE / name)
    f = nbtlib.load(TEMPLATE / "level.dat")
    d = f["Data"]
    d.pop("Player", None)
    d["Time"] = nbtlib.Long(0)
    d["DayTime"] = nbtlib.Long(0)
    f.save()
    print(json.dumps({"template": str(TEMPLATE), "seed": int(d["WorldGenSettings"]["seed"]),
                      "spawn": [int(d["SpawnX"]), int(d["SpawnY"]), int(d["SpawnZ"])]}))


def reset() -> None:
    if game_pids():
        sys.exit("stop the game first")
    if not TEMPLATE.exists():
        sys.exit("no template; run make-template once")
    dst = GAME / "saves" / WORLD
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(TEMPLATE, dst)


def set_options(values: dict) -> None:
    opt = GAME / "options.txt"
    lines = opt.read_text().splitlines()
    seen = set()
    for i, line in enumerate(lines):
        k = line.split(":", 1)[0]
        if k in values:
            lines[i] = f"{k}:{values[k]}"
            seen.add(k)
    lines += [f"{k}:{v}" for k, v in values.items() if k not in seen]
    opt.write_text("\n".join(lines) + "\n")


def launch(windowed: bool = False, wait_s: float = 240.0) -> dict:
    if game_pids():
        sys.exit("already running")
    set_options({**SESSION_OPTIONS, "fullscreen": "false" if windowed else "true"})
    if LOG.exists():
        LOG.unlink()  # so in_world() only sees this launch
    subprocess.Popen(["prismlauncher", "--launch", INSTANCE, "--world", WORLD],
                     env={**os.environ, **X_ENV}, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if game_pids() and in_world():
            return {**status(), "seconds_to_world": round(time.time() - t0, 1)}
        time.sleep(2)
    return {**status(), "timed_out_after_s": wait_s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "stop", "make-template", "reset", "launch", "session"])
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--wait", type=float, default=240.0)
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(status()))
    elif a.cmd == "stop":
        print(json.dumps({"stopped": stop()}))
    elif a.cmd == "make-template":
        make_template()
    elif a.cmd == "reset":
        reset()
        print(json.dumps({"reset": WORLD}))
    elif a.cmd == "launch":
        print(json.dumps(launch(a.windowed, a.wait)))
    elif a.cmd == "session":
        stop()
        reset()
        print(json.dumps(launch(a.windowed, a.wait)))


if __name__ == "__main__":
    main()
