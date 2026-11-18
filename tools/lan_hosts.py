"""Find LAN machines by MAC, so DHCP address changes don't matter.

MACs live outside the (public) repo in ~/.config/sili_quartz/lan_hosts:

    # name   user     mac                 iface(optional)
    sandbox  Bee      2c:f0:5d:xx:xx:xx
    ai0      simleek  60:45:cb:xx:xx:xx

    python tools/lan_hosts.py            # table of name, user@ip
    python tools/lan_hosts.py sandbox    # prints user@ip, exit 1 if not found

Lookup: ARP table first; if a MAC is missing, ping-sweep each wired /24 once
and look again. The result is cached in ~/.cache/sili_quartz/lan_hosts.
"""
import ipaddress
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CONF = os.path.expanduser("~/.config/sili_quartz/lan_hosts")
CACHE = os.path.expanduser("~/.cache/sili_quartz/lan_hosts")


def load_conf():
    hosts = {}
    for line in open(CONF):
        f = line.split("#")[0].split()
        if len(f) >= 3:
            hosts[f[0]] = (f[1], f[2].lower())
    return hosts


def neighbours():
    out = subprocess.run(["ip", "neigh"], capture_output=True, text=True).stdout
    return {m.group(2).lower(): m.group(1)
            for m in re.finditer(r"^(\d+\.\d+\.\d+\.\d+) .*lladdr (\S+)", out, re.M)}


def wired_subnets():
    out = subprocess.run(["ip", "-o", "-4", "addr"], capture_output=True, text=True).stdout
    nets = []
    for m in re.finditer(r"^\d+: (\S+)\s+inet (\S+)", out, re.M):
        if re.match(r"(en|eth)", m.group(1)):
            nets.append((m.group(1), ipaddress.ip_interface(m.group(2)).network))
    return nets


def sweep():
    jobs = [(dev, str(ip)) for dev, net in wired_subnets() if net.num_addresses <= 256
            for ip in net.hosts()]
    with ThreadPoolExecutor(128) as ex:
        list(ex.map(lambda j: subprocess.run(["ping", "-c1", "-W1", "-I", j[0], j[1]],
                                             capture_output=True), jobs))


def ssh_ok(target, timeout=4):
    return subprocess.run(["ssh", "-i", os.path.expanduser("~/.ssh/id_ed25519_sandbox_link"),
                           "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}",
                           "-o", "StrictHostKeyChecking=accept-new", target, "true"],
                          capture_output=True).returncode == 0


def resolve_all():
    hosts = load_conf()
    arp = neighbours()
    if any(mac not in arp for _, mac in hosts.values()):
        sweep()
        arp = neighbours()
    found = {name: f"{user}@{arp[mac]}" for name, (user, mac) in hosts.items() if mac in arp}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        f.writelines(f"{n} {t}\n" for n, t in found.items())
    return hosts, found


def resolve(name):
    """user@ip for `name`: cached address if ssh still answers, else rediscover."""
    if os.path.exists(CACHE):
        for line in open(CACHE):
            n, t = line.split()
            if n == name and ssh_ok(t):
                return t
    _, found = resolve_all()
    if name not in found:
        raise SystemExit(f"{name}: MAC not on any wired subnet ({CONF})")
    return found[name]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(resolve(sys.argv[1]))
    else:
        hosts, found = resolve_all()
        for name in hosts:
            print(f"{name:10s} {found.get(name, 'NOT FOUND')}")
