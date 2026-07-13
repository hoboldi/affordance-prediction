#!/bin/bash
# Verb-MIXUP training, relaunched until it lands on CUDA (box CUDA init is flaky; torch caches the failed
# init per-process, so only a fresh process fixes it).
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for attempt in $(seq 8); do
  rm -rf outputs/vmix_fold0; : > /tmp/vmix.log
  CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 2400 .venv-affordance/bin/python scripts/finetune_human.py \
    --ckpt outputs/ov_concat_finepatch/best.pt --manifest "$SCR/manifest.cv.jsonl" --split "$SCR/cv_fold0.json" \
    --no_freeze_backbone --verb_mix 0.5 --max_vertices 30000 --epochs 18 --patience 18 --lr 3e-4 \
    --device cuda --seed 0 --output_dir outputs/vmix_fold0 >> /tmp/vmix.log 2>&1 &
  bg=$!
  ok=0
  for i in $(seq 40); do
    if grep -q "device=cuda" /tmp/vmix.log 2>/dev/null; then ok=1; break; fi
    if grep -q "device=cpu" /tmp/vmix.log 2>/dev/null; then break; fi
    kill -0 $bg 2>/dev/null || break
    sleep 1
  done
  if [ $ok -eq 1 ]; then
    echo "attempt $attempt: device=cuda -> training to completion" >> /tmp/vmix_launch.log
    wait $bg; echo "training exit $?" >> /tmp/vmix_launch.log; exit 0
  fi
  echo "attempt $attempt: cpu/fail -> retry" >> /tmp/vmix_launch.log
  kill $bg 2>/dev/null; sleep 3
done
echo "GAVE UP (CUDA never initialized)" >> /tmp/vmix_launch.log; exit 1
