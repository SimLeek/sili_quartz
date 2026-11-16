#!/usr/bin/env bash
# Bring up the direct GPU-box link on 10.0.0.0/24.
#   sudo tools/lan_up.sh <host-number> [iface]
# laptop = 2 (enp2s0), arch-sandbox = 1 (enp42s0). Safe to re-run.
set -euo pipefail

host="${1:?usage: $0 <host-number 1-254> [iface]}"
[[ "$host" =~ ^[0-9]+$ ]] && (( host >= 1 && host <= 254 )) || { echo "bad host number: $host" >&2; exit 1; }

iface="${2:-}"
if [[ -z "$iface" ]]; then
  # only wired interfaces (en*/eth*); refuse to guess between several
  mapfile -t cands < <(ip -br link | awk '$1 ~ /^(en|eth)/ {print $1}')
  (( ${#cands[@]} == 1 )) || { echo "pick an interface: ${cands[*]:-none found}" >&2; exit 1; }
  iface="${cands[0]}"
fi

ip link set dev "$iface" up
ip addr replace "10.0.0.${host}/24" dev "$iface"
ip -br addr show dev "$iface"
