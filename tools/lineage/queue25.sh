#!/bin/bash
# Random-init checks after the drive generation (queue 24): the generation's winner and runner-up, trained from
# (a) the last chunk re-initialized (layers 21-24: random policy on pretrained perception) and (b) scrambled keys,
# each against the same random start with no training; then a near-zero start with training energy on vs off.
# Logs to arms_queue3.log; prints QUEUE25_DONE.
C=2656d9b
R=~/runs
until grep -q "QUEUE24_DONE" $R/arms_queue3.log; do sleep 60; done
W=$(grep "^WINNER" $R/arms_queue3.log | tail -n 1 | awk '{print $2}')
RU=$(grep "^RUNNERUP" $R/arms_queue3.log | tail -n 1 | awk '{print $2}')
echo "RANDINIT checks for winner=$W runner-up=$RU"
for START in reinit_l20_23 scrambled_s0; do
  CK=$R/randinit/reinit_l20_23.pt; [ $START = scrambled_s0 ] && CK=$R/tokmem/scrambled_s0.pt
  CTRL=${START}_ctrl
  bash $R/train_session3.sh $C A1 none $CTRL 3600 "--health=pr --grammar --w-rep=0 --w-pg=0 --init=$CK" "--grammar --init $CK"
  bash $R/eval_node.sh $CTRL
  DONE=""
  for N in $W $RU; do
    [ -z "$N" ] || [ "$N" = none ] && continue
    F=$(python3 $R/flags_for.py $N); TX=${F%%|*}; AX=${F#*|}
    TAG=${START}_${N}
    bash $R/train_session3.sh $C A1 fixed:33 $TAG 7200 "$TX --init=$CK" "$AX --init $CK"
    bash $R/eval_node.sh $TAG
    DONE="$DONE $TAG"
  done
  python3 $R/randinit_report.py $CTRL $DONE
done
# near-zero start (user): the last chunk at 1e-6 x its init, connections not grown yet. Gated MLP blocks are
# deadlocked there (every gradient second-order small) and pass the residual stream through; energy's forced firing
# was built to break exactly this. Compare the same drive with training energy on and off. Energy curiosity needs
# energy, so the on/off pair uses the winner if its drive has no energy, else episodic_int.
CK=$R/randinit/tiny_l20_23.pt
bash $R/train_session3.sh $C A1 none tiny_ctrl 3600 "--health=pr --grammar --w-rep=0 --w-pg=0 --init=$CK" "--grammar --init $CK"
bash $R/eval_node.sh tiny_ctrl
DONE=""
if [ -n "$W" ] && [ "$W" != none ]; then
  F=$(python3 $R/flags_for.py $W); TX=${F%%|*}; AX=${F#*|}
  bash $R/train_session3.sh $C A1 fixed:33 tiny_$W 7200 "$TX --init=$CK" "$AX --init $CK"; bash $R/eval_node.sh tiny_$W; DONE="$DONE tiny_$W"
  D=$W; echo "$TX" | grep -q -- "--reward=[a-z,]*energy" && D=episodic_int
  F=$(python3 $R/flags_for.py $D); TX=${F%%|*}; AX=${F#*|}
  if [ "$D" != "$W" ]; then
    bash $R/train_session3.sh $C A1 fixed:33 tiny_${D}_energy 7200 "$TX --init=$CK" "$AX --init $CK"; bash $R/eval_node.sh tiny_${D}_energy; DONE="$DONE tiny_${D}_energy"
  fi
  bash $R/train_session3.sh $C A1 none tiny_${D}_noenergy 7200 "$TX --init=$CK" "$AX --init $CK"; bash $R/eval_node.sh tiny_${D}_noenergy; DONE="$DONE tiny_${D}_noenergy"
fi
python3 $R/randinit_report.py tiny_ctrl $DONE
echo QUEUE25_DONE
