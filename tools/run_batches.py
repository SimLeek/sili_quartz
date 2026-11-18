"""Run integration-test batches on several GPUs across machines, in parallel.

    python tools/run_batches.py            # 5 batches: laptop, sandbox P40 + 2060, ai0 1080 + 1660 SUPER
    python tools/run_batches.py --dry-run

A batch is (host, nvidia-smi GPU index, pytest -k expression). Remote hosts are
found by MAC (tools/lan_hosts.py); each gets the repo rsynced to its own repo dir
and runs its own venv. CUDA_DEVICE_ORDER is pinned to PCI_BUS_ID so the index
means the same thing to CUDA as to nvidia-smi (on arch-sandbox they otherwise
disagree). Logs land in runs/batches/.
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

LOCAL_PY = sys.executable


@dataclass(frozen=True)
class Host:
    name: str  # lan_hosts name, or "local"
    repo: str  # repo dir on that host
    py: str  # python on that host

    @property
    def target(self) -> str:
        return resolve(self.name)


LOCAL = Host("local", str(ROOT), LOCAL_PY)
SANDBOX = Host("sandbox", "~/sili_quartz", "~/venv_quartz/bin/python")
AI0 = Host("ai0", "~/claude_code/sili_quartz", "~/claude_code/venv_quartz/bin/python")


@dataclass
class Batch:
    name: str
    host: Host
    gpu: int  # nvidia-smi index on that host
    k: str  # pytest -k expression


DEFAULT = [
    Batch("laptop-1660ti", LOCAL, 0, "in_context or no_memory or memory_learns"),
    Batch("sandbox-p40", SANDBOX, 0, "family_capacity_probe"),
    Batch("sandbox-2060", SANDBOX, 1, "omega_keeps"),
    Batch("ai0-1080", AI0, 0, "family_learns"),
    Batch("ai0-1660super", AI0, 1, "ablation"),
]


def ssh(host: Host, cmd: str) -> list[str]:
    return ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", host.target, cmd]


def sync_remote(host: Host) -> None:
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
            f"{host.target}:{host.repo}/",
        ],
        check=True,
    )
    subprocess.run(ssh(host, f"{host.py} -m pip install -q --no-deps --no-build-isolation -e {host.repo}"), check=True)


def launch(b: Batch, log_dir: Path, max_temp: int) -> tuple[Batch, subprocess.Popen]:
    env = f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={b.gpu} SQ_DEVICE=cuda"
    # -rP keeps the stdout of PASSED tests, which is where the [capacity] numbers are.
    pytest = f'-m pytest -q -rP -m integration -k "{b.k}" -p no:cacheprovider'
    # Every batch runs under the temperature kill-switch. The P40's blower is
    # invisible to the driver, so this is the only thing between a long batch
    # and firmware slowdown at 92C.
    guard = f"tools/gpu_guard.sh --gpu {b.gpu} --max-temp {max_temp} --poll 10 --"
    log = (log_dir / f"{b.name}.log").open("w")
    run = f"cd {b.host.repo} && {env} {guard} {b.host.py} {pytest}"
    cmd = ["bash", "-c", run] if b.host is LOCAL else ssh(b.host, run)
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
        if not (a.local_only and b.host is not LOCAL) and not (a.remote_only and b.host is LOCAL)
    ]
    for b in batches:
        where = "local" if b.host is LOCAL else b.host.target
        print(f"  {b.name:<14} {where:<20} gpu={b.gpu}  -k '{b.k}'")
    if a.dry_run:
        return
    for h in {b.host for b in batches if b.host is not LOCAL}:
        sync_remote(h)
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
