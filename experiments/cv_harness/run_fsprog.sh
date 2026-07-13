#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for attempt in $(seq 8); do
  : > /tmp/fsprog.log
  CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 1500 .venv-affordance/bin/python "$SCR/render_fewshot_prog.py" >> /tmp/fsprog.log 2>&1 &
  bg=$!; ok=0
  for i in $(seq 200); do
    grep -q "K=0: AUPRC" /tmp/fsprog.log 2>/dev/null && { ok=1; break; }
    grep -qiE "CUDA driver initialization failed|Traceback|Error" /tmp/fsprog.log 2>/dev/null && break
    kill -0 $bg 2>/dev/null || break; sleep 1
  done
  if [ $ok -eq 1 ]; then wait $bg; echo "fsprog: exit $?" >> /tmp/fsprog_launch.log; exit; fi
  if grep -qiE "Traceback|Error:" /tmp/fsprog.log 2>/dev/null; then echo "fsprog: BUG" >> /tmp/fsprog_launch.log; exit 1; fi
  echo "fsprog att $attempt: retry" >> /tmp/fsprog_launch.log; kill $bg 2>/dev/null; sleep 5
done
