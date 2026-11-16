"""Run integration-test batches on several GPUs across machines, in parallel.

    python tools/run_batches.py            # default: 3 batches on laptop:0, remote:P40, remote:2060
    python tools/run_batches.py --dry-run

A batch is (host, nvidia-smi GPU index, pytest -k expression). Remote hosts get
the repo rsynced to ~/sili_quartz and use ~/venv_quartz. CUDA_DEVICE_ORDER is
pinned to PCI_BUS_ID so the index means the same thing to CUDA as to nvidia-smi
(on arch-sandbox they otherwise disagree). Logs land in runs/batches/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519_sandbox_link")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lan_hosts import resolve  # noqa: E402

REMOTE = os.environ.get("SQ_REMOTE") or resolve("sandbox")
REMOTE_PY = "~/venv_quartz/bin/python"
LOCAL_PY = sys.executable


@dataclass
class Batch:
    name: str
    host: str  # "local" or ssh target
    gpu: int  # nvidia-smi index on that host
    k: str  # pytest -k expression


DEFAULT = [
    Batch(
        "laptop-1660ti",
        "local",
        0,
        "long_gap_recall and (base or omega) or in_context or no_memory or family_learns",
    ),
    Batch("remote-p40", REMOTE, 0, "poly and not omega_keeps or (family_capacity_probe and titans)"),
    Batch(
        "remote-2060",
        REMOTE,
        1,
        "omega_keeps or (long_gap_recall and both) or (family_capacity_probe and (rmt or combined))",
    ),
]


def ssh(cmd: str) -> list[str]:
    return ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", REMOTE, cmd]


def sync_remote() -> None:
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude",
            ".git",
            "--exclude",
            "runs",
            "--exclude",
            "__pycache__",
            "-e",
            f"ssh -i {SSH_KEY} -o BatchMode=yes",
            f"{ROOT}/",
            f"{REMOTE}:sili_quartz/",
        ],
        check=True,
    )
    subprocess.run(ssh(f"{REMOTE_PY} -m pip install -q -e ~/sili_quartz[dev] 2>&1 | tail -1"), check=True)


def launch(b: Batch, log_dir: Path, max_temp: int) -> tuple[Batch, subprocess.Popen]:
    env = f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={b.gpu} SQ_DEVICE=cuda"
    # -rP keeps the stdout of PASSED tests, which is where the [capacity] numbers are.
    pytest = f'-m pytest -q -rP -m integration -k "{b.k}" -p no:cacheprovider'
    # Every batch runs under the temperature kill-switch. The P40's blower is
    # invisible to the driver, so this is the only thing between a long batch
    # and firmware slowdown at 92C.
    guard = f"tools/gpu_guard.sh --gpu {b.gpu} --max-temp {max_temp} --poll 10 --"
    log = (log_dir / f"{b.name}.log").open("w")
    if b.host == "local":
        cmd = ["bash", "-c", f"cd {ROOT} && {env} {guard} {LOCAL_PY} {pytest}"]
    else:
        cmd = ssh(f"cd ~/sili_quartz && {env} {guard} {REMOTE_PY} {pytest}")
    return b, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--local-only", action="store_true")
    ap.add_argument("--remote-only", action="store_true")
    ap.add_argument("--max-temp", type=int, default=80, help="abort a batch if its GPU reaches this (C)")
    a = ap.parse_args()
    batches = [
        b
        for b in DEFAULT
        if not (a.local_only and b.host != "local") and not (a.remote_only and b.host == "local")
    ]
    for b in batches:
        print(f"  {b.name:<14} {b.host:<16} gpu={b.gpu}  -k '{b.k}'")
    if a.dry_run:
        return
    if any(b.host != "local" for b in batches):
        sync_remote()
    log_dir = ROOT / "runs" / "batches" / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True)
    procs = [launch(b, log_dir, a.max_temp) for b in batches]
    t0 = time.time()
    for b, p in procs:
        rc = p.wait()
        tail = (log_dir / f"{b.name}.log").read_text().strip().splitlines()[-1:]
        print(f"{b.name:<14} rc={rc} {time.time() - t0:6.0f}s  {tail[0] if tail else ''}")
    print(f"logs: {log_dir}")
    sys.exit(max(p.returncode for _, p in procs))


if __name__ == "__main__":
    main()
