#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
run() { local tag=$1 drop=$2 dropflag=""
  [ -n "$drop" ] && dropflag="--drop $drop"
  for attempt in $(seq 8); do
    : > /tmp/abl_$tag.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 9000 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" \
      --folds 0,1,2 --epochs 18 --lr 1e-3 --agg max --knn_k 24 $dropflag --tag $tag >> /tmp/abl_$tag.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 250); do
      grep -q "params" /tmp/abl_$tag.log 2>/dev/null && { ok=1; break; }
      grep -qiE "CUDA driver initialization failed|Traceback|Error:" /tmp/abl_$tag.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "abl_$tag: OK (att $attempt)" >> /tmp/ablation_launch.log; wait $bg; echo "abl_$tag: exit $?" >> /tmp/ablation_launch.log; return; fi
    if grep -qiE "Traceback|Error:" /tmp/abl_$tag.log 2>/dev/null; then echo "abl_$tag: BUG" >> /tmp/ablation_launch.log; return; fi
    echo "abl_$tag att $attempt: retry" >> /tmp/ablation_launch.log; kill $bg 2>/dev/null; sleep 5
  done
}
run abl_full ""
run abl_noclip clip
run abl_nodino dino
run abl_nogeom geom
run abl_noslat slat
run abl_nonormals normals
echo "ABLATION DONE" >> /tmp/ablation_launch.log
