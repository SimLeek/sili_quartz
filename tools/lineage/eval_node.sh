#!/bin/bash
# Evaluate one lineage node after its session: scorecard vs the root (Root_g), repertoire probe and active fan-in on its
# last checkpoint. Writes ~/runs/eval/<TAG>_{scorecard,repertoire,fanin}.jsonl. usage: eval_node.sh TAG
T=$1; R=~/runs; E=$R/eval
[ -s $R/t_$T/actor.jsonl ] || { echo "EVAL_SKIP $T (no run)"; exit 0; }
rm -f $E/${T}_scorecard.jsonl $E/${T}_repertoire.jsonl $E/${T}_fanin.jsonl
python3 ~/sq_dev/tools/run_scorecard.py --ref $R/t_Root_g --minutes 10 --out $E/${T}_scorecard.jsonl $R/t_$T > /dev/null 2>&1
CK=$(ls $R/t_$T/checkpoints/step_*.pt 2>/dev/null | tail -n 1)
if [ -n "$CK" ]; then
  cd ~/sq_dev && . ~/venv_quartz/bin/activate
  export PYTHONPATH=$PWD CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
  timeout 3600 python tools/repertoire_probe.py --set world_A2b:$R/t_A2b:0:10 --set world_A2g:$R/t_A2g:0:10 --states 15 --k 4 --out $E/${T}_repertoire.jsonl $CK > $E/${T}_probe.log 2>&1
  timeout 3600 python tools/fanin_active.py --ref $R/t_A2b --minutes 0:10 --windows 24 --out $E/${T}_fanin.jsonl $CK >> $E/${T}_probe.log 2>&1
fi
echo "EVAL_DONE $T"
