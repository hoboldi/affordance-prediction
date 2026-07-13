#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for attempt in $(seq 8); do
  rm -rf outputs/vdesc_fold0; : > /tmp/vdesc.log
  CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 2400 .venv-affordance/bin/python scripts/finetune_human.py \
    --ckpt outputs/ov_concat_finepatch/best.pt --manifest "$SCR/manifest.cv.jsonl" --split "$SCR/cv_fold0.json" \
    --no_freeze_backbone --verb_desc --max_vertices 30000 --epochs 18 --patience 18 --lr 3e-4 \
    --device cuda --seed 0 --output_dir outputs/vdesc_fold0 >> /tmp/vdesc.log 2>&1 &
  bg=$!; ok=0
  for i in $(seq 40); do
    grep -q "device=cuda" /tmp/vdesc.log 2>/dev/null && { ok=1; break; }
    grep -q "device=cpu" /tmp/vdesc.log 2>/dev/null && break
    kill -0 $bg 2>/dev/null || break; sleep 1
  done
  if [ $ok -eq 1 ]; then echo "attempt $attempt: cuda" >> /tmp/vdesc_launch.log; wait $bg; echo "exit $?" >> /tmp/vdesc_launch.log; exit 0; fi
  echo "attempt $attempt: cpu/retry" >> /tmp/vdesc_launch.log; kill $bg 2>/dev/null; sleep 3
done
echo "GAVE UP" >> /tmp/vdesc_launch.log; exit 1
