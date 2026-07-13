#!/bin/bash
# 5-fold CV: head-only FT per fold, fixed 40 epochs (patience=40 => no early-stop peeking), eval last.pt later.
cd /home/kraum/Prototype
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src
PY=.venv-affordance/bin/python
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
echo "=== CV START $(date '+%H:%M') ===" > /tmp/cv_status.txt
for k in 0 1 2 3 4; do
  echo ">>> fold $k start $(date '+%H:%M')" >> /tmp/cv_status.txt
  $PY scripts/finetune_human.py --ckpt outputs/ov_concat_finepatch/best.pt --manifest $SCR/manifest.cv.jsonl \
    --split $SCR/cv_fold$k.json --freeze_backbone --epochs 40 --patience 40 --lr 1e-4 --device cuda --seed 0 \
    --output_dir outputs/cv_fold$k > /tmp/cv_fold$k.log 2>&1
  echo "  fold $k done $(date '+%H:%M') last.pt=$([ -f outputs/cv_fold$k/last.pt ] && echo OK || echo MISSING)" >> /tmp/cv_status.txt
done
echo "=== CV TRAIN DONE $(date '+%H:%M') ===" >> /tmp/cv_status.txt
