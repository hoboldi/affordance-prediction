#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
# WAIT for the clean re-run (gc_base=k24clean, mlp_clean=geomclean) to finish before using the GPU
echo "loco: waiting for clean runs..." >> /tmp/loco_launch.log
while ps -u kraum -o cmd | grep -v grep | grep -qE "tag k24clean|geomclean_fold"; do sleep 30; done
echo "loco: GPU free, starting" >> /tmp/loco_launch.log
run() {  # $1=category
  for attempt in $(seq 8); do
    : > /tmp/loco_$1.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 5000 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" \
      --split_file "$SCR/loco_$1.json" --epochs 18 --lr 1e-3 --agg max --knn_k 24 --tag loco_$1 >> /tmp/loco_$1.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 150); do
      grep -q "params" /tmp/loco_$1.log 2>/dev/null && { ok=1; break; }     # printed only AFTER CUDA init
      grep -qi "CUDA driver initialization failed\|device=cpu" /tmp/loco_$1.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "loco_$1: OK (att $attempt)" >> /tmp/loco_launch.log; wait $bg; echo "loco_$1: exit $?" >> /tmp/loco_launch.log; return; fi
    echo "loco_$1 att $attempt: cuda-fail/retry" >> /tmp/loco_launch.log; kill $bg 2>/dev/null; sleep 5
  done
}
run cup
run bottle
run handbag
echo "GNN LOCO DONE" >> /tmp/loco_launch.log
