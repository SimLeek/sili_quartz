"""Is the winner's lead over the runner-up larger than the noise floor?
usage: python3 margin_check.py WINNER RUNNERUP -> prints CLEAR or RERUN with the numbers.
RERUN when the coherence lead is under 2 x the probe's sampling sd, or the median-deviation lead is under the
untouched runs' spread."""
import json, os, sys
E = os.path.expanduser("~/runs/eval")
def last(p):
    return json.loads(open(p).readlines()[-1]) if os.path.exists(p) and os.path.getsize(p) else None
w, r = sys.argv[1], sys.argv[2]
nf = json.load(open(f"{E}/noise_floor.json"))
cw, cr = (last(f"{E}/{t}_repertoire.jsonl") or {} for t in (w, r))
dw, dr = (last(f"{E}/{t}_scorecard.jsonl") or {} for t in (w, r))
coh_lead = (cw.get("coherence") or 0) - (cr.get("coherence") or 0)
dev_lead = (dr.get("median_deviation") or 0) - (dw.get("median_deviation") or 0)
coh_floor = 2 * nf.get("probe", {}).get("coherence", {}).get("sd", 0.0)
dev_floor = nf.get("play", {}).get("median_deviation", {}).get("spread", 0.0)
clear = coh_lead > coh_floor and dev_lead > -dev_floor
print(("CLEAR" if clear else "RERUN") + f" coherence lead {coh_lead:.4f} (floor {coh_floor:.4f}); "
      f"deviation lead {dev_lead:.4f} (floor {dev_floor:.4f})")
