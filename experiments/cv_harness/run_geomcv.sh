#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for k in 0 1 2; do
  for attempt in $(seq 6); do
    rm -rf outputs/geomcv_fold${k}; : > /tmp/geomcv_${k}.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 2400 .venv-affordance/bin/python scripts/finetune_human.py \
      --ckpt outputs/ov_concat_finepatch/best.pt --manifest "$SCR/manifest.cv.jsonl" --split "$SCR/cv_fold${k}.json" \
      --no_freeze_backbone --geom_dim 5 --max_vertices 30000 --epochs 18 --patience 18 --lr 3e-4 \
      --device cuda --seed 0 --output_dir outputs/geomcv_fold${k} >> /tmp/geomcv_${k}.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 45); do
      grep -q "device=cuda" /tmp/geomcv_${k}.log 2>/dev/null && { ok=1; break; }
      grep -q "device=cpu" /tmp/geomcv_${k}.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "fold$k: cuda (attempt $attempt)" >> /tmp/geomcv_launch.log; wait $bg; echo "fold$k: exit $?" >> /tmp/geomcv_launch.log; break; fi
    echo "fold$k attempt $attempt: cpu/retry" >> /tmp/geomcv_launch.log; kill $bg 2>/dev/null; sleep 3
  done
done
echo "ALL GEOMCV DONE" >> /tmp/geomcv_launch.log
