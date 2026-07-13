#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for attempt in $(seq 8); do
  : > /tmp/gc_base.log
  CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 9000 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" \
    --folds 0,1,2,3,4 --epochs 18 --lr 1e-3 --agg max --knn_k 24 --tag k24clean >> /tmp/gc_base.log 2>&1 &
  bg=$!; ok=0
  for i in $(seq 120); do
    grep -q "fold 0:" /tmp/gc_base.log 2>/dev/null && { ok=1; break; }   # printed only AFTER CUDA init succeeds
    grep -qi "CUDA driver initialization failed\|device=cpu" /tmp/gc_base.log 2>/dev/null && break
    kill -0 $bg 2>/dev/null || break; sleep 1
  done
  if [ $ok -eq 1 ]; then echo "gc_base: OK cuda (att $attempt)" >> /tmp/gcbase_launch.log; wait $bg; echo "gc_base: exit $?" >> /tmp/gcbase_launch.log; exit; fi
  echo "gc_base att $attempt: cuda-init-fail/retry" >> /tmp/gcbase_launch.log; kill $bg 2>/dev/null; sleep 5
done
