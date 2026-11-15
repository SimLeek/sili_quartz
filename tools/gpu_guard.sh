#!/usr/bin/env bash
# Run a GPU workload under a temperature kill-switch.
#
# WHY THIS EXISTS. The Tesla P40 is a passively-cooled datacenter card
# being fed by a small external blower, and the driver cannot see that
# blower at all -- `nvidia-smi --query-gpu=fan.speed` returns [N/A], so
# there is no feedback loop and no fan curve. The card's own protection
# is firmware slowdown at 92C and shutdown at 95C (measured), but those
# are last-resort limits: sitting at 90C sustained is still damaging even
# though nothing trips. So experiments get an external guard with real
# margin instead of trusting the card to save itself.
#
# Default abort is 80C, 12C below the P40's slowdown point.
#
# Also refuses to run if the temperature cannot be READ -- running a long
# job blind on this card is the exact thing this script prevents.
#
# Usage:
#   ./gpu_guard.sh [opts] -- <command...>
#   ./gpu_guard.sh [opts] --watch
#
#   --gpu N|NAME       GPU to watch: an nvidia-smi index, or a name
#                      substring like "P40" (PREFERRED -- see note below)
#   --max-temp C       abort the workload at/above this (default 80)
#   --power-limit W    cap board power before running; restored on exit.
#                      Needs root. P40 accepts 125-250W, default 250W.
#   --poll S           seconds between samples (default 5)
#   --log FILE         append CSV: iso_time,temp_c,power_w,util_pct
#   --watch            monitor only, no workload (Ctrl-C to stop)
#
# Exit: the command's own status, or 2 if aborted on temperature,
#       3 if the GPU could not be read.
set -uo pipefail

GPU=0; MAX_TEMP=80; POLL=5; LOG=""; PLIMIT=""; WATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --max-temp) MAX_TEMP="$2"; shift 2;;
    --power-limit) PLIMIT="$2"; shift 2;;
    --poll) POLL="$2"; shift 2;;
    --log) LOG="$2"; shift 2;;
    --watch) WATCH=1; shift;;
    --) shift; break;;
    *) echo "unknown option: $1" >&2; exit 1;;
  esac
done

command -v nvidia-smi >/dev/null || { echo "ERROR: no nvidia-smi" >&2; exit 3; }

# --gpu may be a NAME SUBSTRING, not just an index, and that is the safer
# way to use it. nvidia-smi indexes by PCI bus while CUDA uses its own
# ordering, and on arch-sandbox the two are REVERSED: nvidia-smi 0 is the
# P40, CUDA device 0 is the RTX 2060. So "--gpu 0" can easily guard one
# card while the workload runs on the other. `--gpu P40` cannot.
# A purely numeric argument is ambiguous: "2060" is a card NAME, not an
# index. So treat digits as an index ONLY if such an index actually exists;
# otherwise fall through to name matching.
GPU_IS_INDEX=0
if [[ "$GPU" =~ ^[0-9]+$ ]] \
   && nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
      | tr -d ' ' | grep -qx "$GPU"; then
  GPU_IS_INDEX=1
fi
if [[ -n "$GPU" && $GPU_IS_INDEX -eq 0 ]]; then
  MATCHES="$(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null \
             | grep -i -- "$GPU" || true)"
  COUNT="$(grep -c . <<<"${MATCHES:-}" || true)"
  if [[ -z "$MATCHES" ]]; then
    echo "ERROR: no GPU name matches '$GPU'. Available:" >&2
    nvidia-smi --query-gpu=index,name --format=csv,noheader >&2
    exit 3
  elif [[ "$COUNT" != "1" ]]; then
    echo "ERROR: '$GPU' matches $COUNT GPUs; be more specific:" >&2
    echo "$MATCHES" >&2
    exit 3
  fi
  GPU="$(cut -d, -f1 <<<"$MATCHES" | tr -dc '0-9')"
  echo "resolved --gpu to nvidia-smi index $GPU"
fi

read_temp() { nvidia-smi -i "$GPU" --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | tr -dc '0-9'; }
read_row()  { nvidia-smi -i "$GPU" --query-gpu=temperature.gpu,power.draw,utilization.gpu \
                --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }

NAME="$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null)"
[[ -n "$NAME" ]] || { echo "ERROR: GPU $GPU not readable" >&2; exit 3; }
T0="$(read_temp)"
[[ -n "$T0" ]] || { echo "ERROR: cannot read temperature of GPU $GPU -- refusing to run blind" >&2; exit 3; }

SLOW="$(nvidia-smi -i "$GPU" -q -d TEMPERATURE 2>/dev/null | awk -F: '/Slowdown Temp/{gsub(/[^0-9]/,"",$2); print $2; exit}')"
echo "GPU $GPU: $NAME   now ${T0}C   abort at ${MAX_TEMP}C   firmware slowdown ${SLOW:-?}C"
# Make the workload's CUDA ordering match nvidia-smi's, so "the GPU being
# guarded" and "the GPU being used" are the same card.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [[ -n "$SLOW" && "$MAX_TEMP" -ge "$SLOW" ]]; then
  echo "WARNING: --max-temp ${MAX_TEMP}C is at/above firmware slowdown ${SLOW}C -- no margin." >&2
fi

RESTORE_PL=""
cleanup() {
  if [[ -n "$RESTORE_PL" ]]; then
    nvidia-smi -i "$GPU" -pl "$RESTORE_PL" >/dev/null 2>&1 \
      && echo "restored power limit to ${RESTORE_PL}W"
  fi
}
trap cleanup EXIT

if [[ -n "$PLIMIT" ]]; then
  [[ $EUID -eq 0 ]] || { echo "ERROR: --power-limit needs root" >&2; exit 1; }
  RESTORE_PL="$(nvidia-smi -i "$GPU" --query-gpu=power.limit --format=csv,noheader,nounits | tr -dc '0-9.')"
  nvidia-smi -i "$GPU" -pl "$PLIMIT" >/dev/null 2>&1 \
    && echo "power limit ${RESTORE_PL}W -> ${PLIMIT}W" \
    || { echo "ERROR: could not set power limit to ${PLIMIT}W" >&2; exit 1; }
fi

[[ -n "$LOG" ]] && { [[ -s "$LOG" ]] || echo "iso_time,temp_c,power_w,util_pct" >> "$LOG"; }

PEAK="$T0"
sample() {
  local row t
  row="$(read_row)"; t="${row%%,*}"
  if [[ -z "$t" ]]; then echo "WARN: temperature unreadable" >&2; return 1; fi
  (( t > PEAK )) && PEAK="$t"
  [[ -n "$LOG" ]] && echo "$(date -Is),$row" >> "$LOG"
  if (( t >= MAX_TEMP )); then return 2; fi
  return 0
}

if [[ $WATCH -eq 1 ]]; then
  echo "watching (Ctrl-C to stop)..."
  while :; do
    sample; s=$?
    printf '\r%s  temp=%sC peak=%sC   ' "$(date +%H:%M:%S)" "$(read_temp)" "$PEAK"
    (( s == 2 )) && { echo; echo "ABORT THRESHOLD REACHED (${MAX_TEMP}C)"; exit 2; }
    sleep "$POLL"
  done
fi

[[ $# -gt 0 ]] || { echo "ERROR: no command given (use -- <command>, or --watch)" >&2; exit 1; }

# Own process group, so a kill reaches the whole workload tree.
setsid "$@" & CHILD=$!
PGID="$(ps -o pgid= -p "$CHILD" 2>/dev/null | tr -dc '0-9')"
START=$(date +%s)
echo "launched pid=$CHILD pgid=${PGID:-?}: $*"

ABORTED=0
while kill -0 "$CHILD" 2>/dev/null; do
  sample; s=$?
  if (( s == 2 )); then
    echo
    echo "!! GPU $GPU hit $(read_temp)C >= ${MAX_TEMP}C -- terminating workload"
    [[ -n "$PGID" ]] && kill -TERM -"$PGID" 2>/dev/null || kill -TERM "$CHILD" 2>/dev/null
    for _ in $(seq 1 15); do kill -0 "$CHILD" 2>/dev/null || break; sleep 1; done
    if kill -0 "$CHILD" 2>/dev/null; then
      echo "   did not exit on SIGTERM; sending SIGKILL"
      [[ -n "$PGID" ]] && kill -KILL -"$PGID" 2>/dev/null || kill -KILL "$CHILD" 2>/dev/null
    fi
    ABORTED=1
    break
  fi
  sleep "$POLL"
done

wait "$CHILD" 2>/dev/null; RC=$?
ELAPSED=$(( $(date +%s) - START ))
echo "---"
echo "elapsed ${ELAPSED}s   peak ${PEAK}C   final $(read_temp)C"
if (( ABORTED )); then echo "RESULT: ABORTED ON TEMPERATURE"; exit 2; fi
echo "RESULT: workload exited rc=$RC"
exit $RC
