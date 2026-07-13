#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
retry() {  # $1=logtag ; rest=command
  local tag=$1; shift
  for attempt in $(seq 6); do
    : > /tmp/$tag.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 "$@" >> /tmp/$tag.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 90); do
      grep -q "device=cuda" /tmp/$tag.log 2>/dev/null && { ok=1; break; }
      grep -q "device=cpu" /tmp/$tag.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "$tag: cuda (att $attempt)" >> /tmp/geal_launch.log; wait $bg; echo "$tag: exit $?" >> /tmp/geal_launch.log; return; fi
    echo "$tag att $attempt: cpu/retry" >> /tmp/geal_launch.log; kill $bg 2>/dev/null; sleep 3
  done
}
# 1) GEAL pretrain (k24 config) on 1227 objects x 6 verbs
retry geal_pretrain timeout 14400 .venv-affordance/bin/python "$SCR/train_gnn_pretrain.py" \
  --knn_k 24 --gnn_layers 3 --gnn_hidden 128 --epochs 12 --sub 30000 --out outputs/gnn_pretrain_k24.pt
# 2) from-scratch baseline, k24, folds 1-4 (fold0 already = gnn_k24_fold0)
retry geal_base timeout 7200 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" \
  --folds 1,2,3,4 --epochs 18 --lr 1e-3 --agg max --knn_k 24 --tag k24
# 3) GEAL-pretrained -> human-FT, k24, folds 0-4
retry geal_pre timeout 9000 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" \
  --folds 0,1,2,3,4 --epochs 18 --lr 1e-3 --agg max --knn_k 24 --init_from outputs/gnn_pretrain_k24.pt --tag k24pre
echo "GEAL EXPERIMENT DONE" >> /tmp/geal_launch.log
