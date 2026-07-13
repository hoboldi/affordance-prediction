#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
BIG="--gnn_layers 4 --gnn_hidden 192 --knn_k 24"
retry() { local tag=$1; shift
  for attempt in $(seq 8); do
    : > /tmp/$tag.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 "$@" >> /tmp/$tag.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 250); do
      grep -q "params" /tmp/$tag.log 2>/dev/null && { ok=1; break; }
      grep -qiE "CUDA driver initialization failed|Traceback|Error:" /tmp/$tag.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "$tag: OK (att $attempt)" >> /tmp/final_launch.log; wait $bg; echo "$tag: exit $?" >> /tmp/final_launch.log; return; fi
    if grep -qiE "Traceback|Error:" /tmp/$tag.log 2>/dev/null; then echo "$tag: BUG" >> /tmp/final_launch.log; return; fi
    echo "$tag att $attempt: retry" >> /tmp/final_launch.log; kill $bg 2>/dev/null; sleep 5
  done
}
# scratch folds 3 (was incomplete) + 4
retry final_scratch2 timeout 14400 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" --folds 3,4 --epochs 18 --lr 1e-3 --agg max $BIG --tag bigk24clean
# FINAL: bigk24 + pretrain, folds 0-4
retry final_model timeout 20000 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" --folds 0,1,2,3,4 --epochs 18 --lr 1e-3 --agg max $BIG --init_from outputs/gnn_pretrain_bigk24.pt --tag bigk24preclean
echo "FINAL MODEL DONE" >> /tmp/final_launch.log
