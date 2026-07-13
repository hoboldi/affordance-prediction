#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for attempt in $(seq 8); do
  : > /tmp/fewshot.log
  CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 2000 .venv-affordance/bin/python "$SCR/few_shot.py" --device cuda >> /tmp/fewshot.log 2>&1 &
  bg=$!; ok=0
  for i in $(seq 200); do
    grep -q "ZERO-SHOT" /tmp/fewshot.log 2>/dev/null && { ok=1; break; }        # after 1st CUDA forward
    grep -qiE "CUDA driver initialization failed|Traceback|Error" /tmp/fewshot.log 2>/dev/null && break
    kill -0 $bg 2>/dev/null || break; sleep 1
  done
  if [ $ok -eq 1 ]; then echo "fewshot: OK (att $attempt)" >> /tmp/fewshot_launch.log; wait $bg; echo "fewshot: exit $?" >> /tmp/fewshot_launch.log; exit; fi
  if grep -qiE "Traceback|Error:" /tmp/fewshot.log 2>/dev/null; then echo "fewshot: BUG (att $attempt)" >> /tmp/fewshot_launch.log; exit 1; fi
  echo "fewshot att $attempt: cuda-fail/retry" >> /tmp/fewshot_launch.log; kill $bg 2>/dev/null; sleep 5
done
