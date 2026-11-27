#!/bin/bash
# Drive generation (same parent: input tether layers 1-12, training-time energy fixed:33, actor clean):
# 0. two more untouched runs (noise floor), 1. single-drive siblings (smoke-gated), 2. combination siblings of the
# healthy singles (RewardMix), 3. select across singles and combinations; if the lead is inside the noise floor, rerun
# winner and runner-up with a second seed and reselect, 4. knob sweep of the winner (lr, dose at 1/4x and 4x),
# 5. the winner's children. After queue 23. QUEUE24_DONE.
C=2656d9b
R=~/runs
until grep -q "QUEUE23_DONE" $R/arms_queue3.log; do sleep 60; done
BASE="--health=pr --grammar --tether=chunks --tether-chunks=3 --w-rep=0 --actor-dose=0"
declare -A TX AX
TX[Tin3_et_curio]="$BASE --reward=energy"; AX[Tin3_et_curio]="--grammar"
TX[curio_input]="$BASE --reward=energy --reward-sites=input --energy-io --energy-io-side=in"; AX[curio_input]="--grammar --energy-io --energy-io-side in"
TX[curio_all]="$BASE --reward=energy --reward-layers=24"; AX[curio_all]="--grammar"
TX[episodic_int]="$BASE --reward=episodic"; AX[episodic_int]="--grammar"
TX[rnd_int]="$BASE --reward=rnd"; AX[rnd_int]="--grammar"
smoke() {  # trainer-only smoke of one configuration; returns 0 if usable
  local T=$1 S=$R/smoke_$1
  rm -rf $S; mkdir -p $S; cp -r $R/trainer_ppo_smoke/actor.jsonl $R/trainer_ppo_smoke/replay $S/; cp $R/t_A1io/energy_means.json $S/
  ( cd ~/sili_quartz_$C && . ~/venv_quartz/bin/activate && export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 && \
    timeout 1500 python examples/gamescaling_trainer.py --run $S --arm A1 --dose fixed:33 --steps 60 --probe-every 0 ${TX[$T]} > $S/trainer.out 2>&1 )
  python3 -c "
import json, math, sys
t=[json.loads(l) for l in open(\"$S/trainer.jsonl\")]; late=[x for x in t if not x[\"warm\"]]
ok = len(late) >= 15 and all(math.isfinite(x[\"grad_norm\"]) for x in t) and sum(x[\"ppo_turns\"] for x in late) > 0 and any(abs(x[\"reward_mean\"]) > 0 for x in late)
print(\"SMOKE $T\", \"ok\" if ok else \"FAIL\", [x[\"reward_mean\"] for x in late][-4:]); sys.exit(0 if ok else 1)"
}
run_node() { bash $R/train_session3.sh $C A1 ${2:-fixed:33} $1 7200 "${TX[$1]}" "${AX[$1]}"; bash $R/eval_node.sh $1; }
# 0. noise floor: two more untouched runs (grammar, no energy, no training, 1 h)
for T in Root_g2 Root_g3; do bash $R/train_session3.sh $C A1 none $T 3600 "--health=pr --grammar --w-rep=0 --w-pg=0" "--grammar"; done
for T in Root_g Root_g2 Root_g3; do bash $R/eval_node.sh $T; done
python3 $R/noise_floor.py
# 1. singles
SINGLES=""
for T in curio_input curio_all episodic_int rnd_int; do smoke $T && SINGLES="$SINGLES $T"; done
echo "SINGLES to run:$SINGLES"
for T in $SINGLES; do run_node $T; done
for T in Tin3_et Tin3_et_curio; do bash $R/eval_node.sh $T; done
# 2. combinations of the healthy singles
PLAN=$(python3 $R/plan_combos.py Tin3_et_curio $SINGLES); echo "$PLAN"
COMBOS=""
while IFS="|" read -r TAG REW XT XA; do
  TAG=${TAG#COMBO }; [ -z "$TAG" ] && continue
  TX[$TAG]="$BASE --reward=$REW $XT"; AX[$TAG]="--grammar $XA"
  smoke $TAG && COMBOS="$COMBOS $TAG"
done <<< "$(echo "$PLAN" | grep '^COMBO ')"
echo "COMBOS to run:$COMBOS"
for T in $COMBOS; do run_node $T; done
# 3. select across singles and combinations
ALL="Tin3_et_curio $SINGLES $COMBOS"
SEL=$(python3 $R/select_sibling.py $ALL); echo "$SEL"
W=$(echo "$SEL" | grep "^WINNER" | awk "{print \$2}"); RU=$(echo "$SEL" | grep "^RUNNERUP" | awk "{print \$2}")
if [ -n "$W" ] && [ "$W" != "none" ] && [ -n "$RU" ] && [ "$RU" != "none" ]; then
  M=$(python3 $R/margin_check.py $W $RU); echo "MARGIN $W vs $RU: $M"
  if [[ "$M" == RERUN* ]]; then
    for T in $W $RU; do TX[${T}_s1]="${TX[$T]} --seed=1"; AX[${T}_s1]="${AX[$T]}"; run_node ${T}_s1; done
    SEL=$(python3 $R/select_sibling.py $ALL ${W}_s1 ${RU}_s1); echo "$SEL"
    W=$(echo "$SEL" | grep "^WINNER" | awk "{print \$2}")
  fi
fi
# 4. knob sweep of the winner: learning rate and training-energy dose at 1/4x and 4x, reported as the worst case
if [ -n "$W" ] && [ "$W" != "none" ]; then
  TX[${W}_lrq]="${TX[$W]} --lr=1.25e-5"; AX[${W}_lrq]="${AX[$W]}"; run_node ${W}_lrq
  TX[${W}_lr4]="${TX[$W]} --lr=2e-4"; AX[${W}_lr4]="${AX[$W]}"; run_node ${W}_lr4
  TX[${W}_doseq]="${TX[$W]}"; AX[${W}_doseq]="${AX[$W]}"; run_node ${W}_doseq fixed:8.25
  TX[${W}_dose4]="${TX[$W]}"; AX[${W}_dose4]="${AX[$W]}"; run_node ${W}_dose4 fixed:132
  python3 $R/sweep_report.py $W
fi
# 5. the winner's children
if [ -n "$W" ] && [ "$W" != "none" ]; then
  for K in "lr1e-5|--lr=1e-5" "tout01|--tether-out --w-tether-out=0.1" "klcov2|--cov=kl --cov-k=0.02"; do
    TAG="${W}_${K%%|*}"; TX[$TAG]="${TX[$W]} ${K#*|}"; AX[$TAG]="${AX[$W]}"
    run_node $TAG
    echo "DECIDE $TAG vs $W: $(python3 $R/decide_node.py $TAG $W)"
  done
else
  echo "NO HEALTHY SIBLING: children not run"
fi
echo QUEUE24_DONE
