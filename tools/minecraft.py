"""Minecraft session control for track A (run on the machine with the game).

    python tools/minecraft.py status
    python tools/minecraft.py stop
    python tools/minecraft.py make-template     # once, from the existing world
    python tools/minecraft.py reset             # world back to its seed, fresh player
    python tools/minecraft.py launch [--windowed] [--wait 240]
    python tools/minecraft.py session           # stop, reset world, restore settings, launch
    python tools/minecraft.py save-settings     # once: options + mod settings as the template
    python tools/minecraft.py restore-settings  # put them back (game stopped)
    python tools/minecraft.py stats             # evaluation only: per-player statistics
    python tools/minecraft.py watch             # relaunch if the game exits; close folder/browser
                                                # windows and Prism dialogs, keep the game in front

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
SETTINGS_TEMPLATE = Path(os.environ.get("MC_SETTINGS_DIR", Path.home() / "mc_templates" / "settings"))
# Game options and mod settings the agent can change from inside the game.
SETTINGS = ("options.txt", "optionsof.txt", "optionsshaders.txt", "config", "defaultconfigs", "xaero")
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


def _per_world(path: Path) -> bool:
    """Data that belongs to one world (JEI bookmarks, map caches, waypoints):
    left out of the settings template, so a reset world starts without it."""
    return WORLD in path.parts


def save_settings(force: bool = False) -> None:
    if SETTINGS_TEMPLATE.exists() and not force:
        sys.exit(f"{SETTINGS_TEMPLATE} exists; pass --force to replace it")
    if SETTINGS_TEMPLATE.exists():
        shutil.rmtree(SETTINGS_TEMPLATE)
    SETTINGS_TEMPLATE.mkdir(parents=True)
    for name in SETTINGS:
        src = GAME / name
        if src.is_dir():
            shutil.copytree(src, SETTINGS_TEMPLATE / name,
                            ignore=lambda d, names: [n for n in names if n == WORLD])
        elif src.exists():
            shutil.copy2(src, SETTINGS_TEMPLATE / name)
    print(json.dumps({"saved_settings": str(SETTINGS_TEMPLATE)}))


def restore_settings() -> None:
    """Put the settings template back (the game must be stopped). Per-world
    data is removed, not restored."""
    if game_pids():
        sys.exit("stop the game first")
    if not SETTINGS_TEMPLATE.exists():
        sys.exit("no settings template; run save-settings once")
    for name in SETTINGS:
        dst, src = GAME / name, SETTINGS_TEMPLATE / name
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        elif src.exists():
            shutil.copy2(src, dst)


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


def stats() -> dict:
    """Evaluation only (never shown to the agent): the world's per-player
    statistics, summed over players. The game writes them on autosave
    (about every 5 minutes) and when it pauses or stops."""
    out = {"distance_cm": 0, "jumps": 0, "play_ticks": 0, "deaths": 0,
           "mined": 0, "crafted": 0, "used": 0, "picked_up": 0, "killed": 0,
           "distinct_blocks_mined": 0, "distinct_items_crafted": 0}
    for f in (GAME / "saves" / WORLD / "stats").glob("*.json"):
        st = json.loads(f.read_text()).get("stats", {})
        custom = st.get("minecraft:custom", {})
        out["distance_cm"] += sum(v for k, v in custom.items() if k.endswith("_one_cm"))
        out["jumps"] += custom.get("minecraft:jump", 0)
        out["play_ticks"] += custom.get("minecraft:play_time", 0)
        out["deaths"] += custom.get("minecraft:deaths", 0)
        for key, name in (("mined", "mined"), ("crafted", "crafted"), ("used", "used"),
                          ("picked_up", "picked_up"), ("killed", "killed")):
            out[key] += sum(st.get(f"minecraft:{name}", {}).values())
        out["distinct_blocks_mined"] += len(st.get("minecraft:mined", {}))
        out["distinct_items_crafted"] += len(st.get("minecraft:crafted", {}))
        mined, crafted = st.get("minecraft:mined", {}), st.get("minecraft:crafted", {})
        # milestones: the first steps of the tech tree
        out["logs_mined"] = out.get("logs_mined", 0) + sum(v for k, v in mined.items() if k.endswith(("_log", "_stem")))
        out["planks_crafted"] = out.get("planks_crafted", 0) + sum(v for k, v in crafted.items() if k.endswith("_planks"))
        out["crafting_tables_crafted"] = out.get("crafting_tables_crafted", 0) + crafted.get("minecraft:crafting_table", 0)
    out.update(player_state())
    return out


def player_state() -> dict:
    """Inventory and position from the saved player data (written on autosave,
    like the statistics), and horizontal distance from the world spawn."""
    try:
        import nbtlib
    except ImportError:
        return {}
    world = GAME / "saves" / WORLD
    files = sorted((world / "playerdata").glob("*.dat"))
    if not files:
        return {}
    p = nbtlib.load(files[0])
    lv = nbtlib.load(world / "level.dat")["Data"]
    x, y, z = (float(v) for v in p["Pos"])
    inv: dict[str, int] = {}
    for item in p.get("Inventory", []):
        name = str(item["id"]).removeprefix("minecraft:")
        inv[name] = inv.get(name, 0) + int(item["Count"])
    return {"pos": [round(x, 1), round(y, 1), round(z, 1)],
            "from_spawn_m": round(((x - int(lv["SpawnX"])) ** 2 + (z - int(lv["SpawnZ"])) ** 2) ** 0.5, 1),
            "health": round(float(p.get("Health", 0)), 1), "inventory": inv,
            "inventory_items": sum(inv.values())}


# Windows the game (or the agent through it) can open: mods' "open folder"
# and "open link" buttons start a file manager or a browser.
CLOSE_CLASSES = {"pcmanfm", "thunar", "nautilus", "firefox", "navigator", "chromium", "google-chrome"}
DESKTOP_NAMES = {"pcmanfm"}  # pcmanfm's own desktop window: never close
PRISM_KEEP_PREFIXES = ("Console window", "Prism Launcher")  # Prism's main and console windows


def _xdo(*args) -> str:
    r = subprocess.run(["xdotool", *args], capture_output=True, text=True, env={**os.environ, **X_ENV})
    return r.stdout.strip()


def focus_guard() -> list[str]:
    """Close windows the game opened (file managers, browsers, Prism dialogs
    such as "Kill Minecraft?") and put the game window back on top. Returns
    what it did."""
    did = []
    for w in _xdo("search", "--onlyvisible", "--name", ".").split():
        cls, name = _xdo("getwindowclassname", w).lower(), _xdo("getwindowname", w)
        prism_dialog = cls == "prismlauncher" and not name.startswith(PRISM_KEEP_PREFIXES)
        if (cls in CLOSE_CLASSES and name not in DESKTOP_NAMES) or prism_dialog:
            _xdo("windowclose", w)
            did.append(f"closed {cls}: {name[:60]}")
    ids = _xdo("search", "--name", "^Minecraft").split()
    if ids:
        mc = ids[0]
        if _xdo("getactivewindow") != mc:
            _xdo("windowmap", mc)
            _xdo("windowactivate", mc)
            did.append("re-activated the game window")
    return did


def watch(poll_s: float = 0.5, relaunch_every_s: float = 10.0, wait_s: float = 240.0) -> None:
    """Keep the game up and in front: relaunch it into the world (no reset)
    when it is not running, and run focus_guard every poll. One JSON line
    per event."""
    last_check = 0.0
    while True:
        now = time.time()
        if now - last_check >= relaunch_every_s:
            last_check = now
            if not game_pids():
                print(json.dumps({"t": now, "event": "game not running, relaunching"}), flush=True)
                result = launch(False, wait_s)
                print(json.dumps({"t": time.time(), "event": "launched", **result}), flush=True)
        if game_pids():
            try:
                for event in focus_guard():
                    print(json.dumps({"t": time.time(), "event": event}), flush=True)
            except Exception as e:  # the guard must never stop the watchdog
                print(json.dumps({"t": time.time(), "event": f"focus guard error: {e}"}), flush=True)
        time.sleep(poll_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status", "stop", "make-template", "reset", "launch", "session",
                                    "stats", "watch", "save-settings", "restore-settings"])
    ap.add_argument("--force", action="store_true")
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
        if SETTINGS_TEMPLATE.exists():
            restore_settings()
        print(json.dumps(launch(a.windowed, a.wait)))
    elif a.cmd == "save-settings":
        save_settings(a.force)
    elif a.cmd == "restore-settings":
        restore_settings()
        print(json.dumps({"restored_settings": str(SETTINGS_TEMPLATE)}))
    elif a.cmd == "stats":
        print(json.dumps(stats()))
    elif a.cmd == "watch":
        watch(wait_s=a.wait)


if __name__ == "__main__":
    main()
