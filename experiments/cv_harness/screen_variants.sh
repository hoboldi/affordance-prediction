#!/bin/bash
cd /home/kraum/Prototype
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src
PY=.venv-affordance/bin/python
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
B=/tmp/slat_screen_board.txt
echo "=== SCREEN START $(date '+%H:%M') | fold-0 baseline mean=0.666 (contain .744 sit .857 pour .697 move .787 display .478 grasp .430) ===" > $B
run() { name=$1; shift
  echo ">>> $name ($*) start $(date '+%H:%M')" >> $B
  nice -n 10 timeout 2400 $PY "$SCR/train_slat_cv.py" --folds 0 --epochs 18 --tag "$name" "$@" > /tmp/slat_screen_$name.log 2>&1; rc=$?
  m=$(grep "TRAINED-MEAN" /tmp/slat_screen_$name.log | grep -oE "SLAT=[0-9.]+" | head -1)
  echo "  $name: $m (vs fold0-base 0.666) rc=$rc" >> $B
  grep -E "^  (contain|sit|pour|move|display|grasp) " /tmp/slat_screen_$name.log | sed 's/^/    /' >> $B
}
run full_ft --full_ft --lr 3e-4
run drop_clip --drop_clip
run bigger --enc_hidden 128 --enc_layers 4
echo "=== SCREEN DONE $(date '+%H:%M') ===" >> $B
