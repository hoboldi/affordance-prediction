#!/bin/bash
cd /home/kraum/Prototype
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src
PY=.venv-affordance/bin/python
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
echo "=== LOVO START $(date '+%H:%M') ===" > /tmp/lovo_status.txt
for V in contain sit pour move display grasp; do
  echo ">>> exclude $V start $(date '+%H:%M')" >> /tmp/lovo_status.txt
  $PY scripts/finetune_human.py --ckpt outputs/ov_concat_finepatch/best.pt --manifest $SCR/manifest.cv.jsonl \
    --split $SCR/split_all.json --exclude_verb $V --freeze_backbone --max_vertices 30000 --epochs 40 --patience 40 --lr 1e-4 \
    --device cuda --seed 0 --output_dir outputs/lovo_$V > /tmp/lovo_$V.log 2>&1
  echo "  $V done $(date '+%H:%M') last=$([ -f outputs/lovo_$V/last.pt ] && echo OK || echo MISS)" >> /tmp/lovo_status.txt
done
echo "=== LOVO TRAIN DONE $(date '+%H:%M') ===" >> /tmp/lovo_status.txt
