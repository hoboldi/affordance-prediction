#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for attempt in $(seq 8); do
  : > /tmp/fewshot_lift.log
  CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 2000 .venv-affordance/bin/python "$SCR/few_shot.py" --device cuda --verb lift --ks 1,3,5 >> /tmp/fewshot_lift.log 2>&1 &
  bg=$!; ok=0
  for i in $(seq 200); do
    grep -q "ZERO-SHOT" /tmp/fewshot_lift.log 2>/dev/null && { ok=1; break; }
    grep -qiE "CUDA driver initialization failed|Traceback|Error" /tmp/fewshot_lift.log 2>/dev/null && break
    kill -0 $bg 2>/dev/null || break; sleep 1
  done
  if [ $ok -eq 1 ]; then echo "lift: OK (att $attempt)" >> /tmp/fewshot_lift_launch.log; wait $bg; echo "lift: exit $?" >> /tmp/fewshot_lift_launch.log; exit; fi
  if grep -qiE "Traceback|Error:" /tmp/fewshot_lift.log 2>/dev/null; then echo "lift: BUG" >> /tmp/fewshot_lift_launch.log; exit 1; fi
  echo "lift att $attempt: retry" >> /tmp/fewshot_lift_launch.log; kill $bg 2>/dev/null; sleep 5
done
