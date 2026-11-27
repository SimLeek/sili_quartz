#!/bin/bash
# Random-init checks after the drive generation (queue 24): the generation's winner and runner-up, trained from
# (a) the last chunk re-initialized (layers 21-24: random policy on pretrained perception) and (b) scrambled keys,
# each against the same random start with no training. Logs to arms_queue3.log; prints QUEUE25_DONE.
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
echo QUEUE25_DONE
