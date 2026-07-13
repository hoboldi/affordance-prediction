#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for attempt in $(seq 6); do
  : > /tmp/dinofusion.log
  CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 3600 .venv-affordance/bin/python "$SCR/build_dino_fusion.py" \
    --manifest "$SCR/manifest.cv.jsonl" --tau 0.1 >> /tmp/dinofusion.log 2>&1 &
  bg=$!; ok=0
  for i in $(seq 90); do
    grep -q "device=cuda" /tmp/dinofusion.log 2>/dev/null && { ok=1; break; }
    grep -q "device=cpu" /tmp/dinofusion.log 2>/dev/null && break
    kill -0 $bg 2>/dev/null || break; sleep 1
  done
  if [ $ok -eq 1 ]; then echo "dinofusion: cuda (att $attempt)" >> /tmp/dinofusion_launch.log; wait $bg; echo "dinofusion: exit $?" >> /tmp/dinofusion_launch.log; exit; fi
  echo "dinofusion att $attempt: cpu/retry" >> /tmp/dinofusion_launch.log; kill $bg 2>/dev/null; sleep 3
done
