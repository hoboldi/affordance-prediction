#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for agg in wsum mean; do
  tag=agg$agg
  for attempt in $(seq 6); do
    rm -f outputs/cv_${tag}_fold0.pt; : > /tmp/${tag}.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 2400 .venv-affordance/bin/python "$SCR/train_slat_cv.py" \
      --folds 0 --epochs 18 --lr 3e-4 --full_ft --vertex_agg $agg --tag $tag >> /tmp/${tag}.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 90); do
      grep -q "^fold 0:" /tmp/${tag}.log 2>/dev/null && { ok=1; break; }
      grep -qiE "CUDA (error|driver|initialization)|out of memory|Traceback" /tmp/${tag}.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break
      sleep 1
    done
    if [ $ok -eq 1 ]; then echo "$tag: training (attempt $attempt)" >> /tmp/agg_launch.log; wait $bg; echo "$tag: exit $?" >> /tmp/agg_launch.log; break; fi
    echo "$tag attempt $attempt: fail/retry" >> /tmp/agg_launch.log; kill $bg 2>/dev/null; sleep 3
  done
done
echo "ALL AGG DONE" >> /tmp/agg_launch.log
