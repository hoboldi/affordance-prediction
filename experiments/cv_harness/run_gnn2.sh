#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
run() {  # $1=folds $2=agg $3=tag
  for attempt in $(seq 6); do
    : > /tmp/gnn_${3}.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 4200 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" \
      --folds "$1" --epochs 18 --lr 1e-3 --agg "$2" --tag "$3" >> /tmp/gnn_${3}.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 60); do
      grep -q "device=cuda" /tmp/gnn_${3}.log 2>/dev/null && { ok=1; break; }
      grep -q "device=cpu" /tmp/gnn_${3}.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "gnn_${3}: cuda (att $attempt)" >> /tmp/gnn_launch.log; wait $bg; echo "gnn_${3}: exit $?" >> /tmp/gnn_launch.log; return; fi
    echo "gnn_${3} att $attempt: cpu/retry" >> /tmp/gnn_launch.log; kill $bg 2>/dev/null; sleep 3
  done
}
run "1,2,3,4" max edgeconv_f1234
run "0" attn attn
echo "GNN2 DONE" >> /tmp/gnn_launch.log
