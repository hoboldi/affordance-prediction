#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
run() {  # $1=tag $2=extra_flags
  for attempt in $(seq 6); do
    : > /tmp/gnn_${1}.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 3000 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" \
      --folds 0 --epochs 18 --lr 1e-3 --agg max --tag "$1" $2 >> /tmp/gnn_${1}.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 60); do
      grep -q "device=cuda" /tmp/gnn_${1}.log 2>/dev/null && { ok=1; break; }
      grep -q "device=cpu" /tmp/gnn_${1}.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "gnn_${1}: cuda (att $attempt)" >> /tmp/gnn_launch.log; wait $bg; echo "gnn_${1}: exit $?" >> /tmp/gnn_launch.log; return; fi
    echo "gnn_${1} att $attempt: cpu/retry" >> /tmp/gnn_launch.log; kill $bg 2>/dev/null; sleep 3
  done
}
run L4      "--gnn_layers 4"
run k16     "--knn_k 16"
run big     "--gnn_layers 4 --gnn_hidden 192 --knn_k 16"
echo "GNN3 CAPACITY DONE" >> /tmp/gnn_launch.log
