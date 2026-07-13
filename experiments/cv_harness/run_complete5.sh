#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
# geom folds 3,4  +  no-geom baseline folds 3,4  -> completes 5-fold for both
run() {  # $1=fold $2=outdir $3=extra_flags $4=logtag
  for attempt in $(seq 6); do
    rm -rf "$2"; : > /tmp/$4.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 2400 .venv-affordance/bin/python scripts/finetune_human.py \
      --ckpt outputs/ov_concat_finepatch/best.pt --manifest "$SCR/manifest.cv.jsonl" --split "$SCR/cv_fold$1.json" \
      --no_freeze_backbone $3 --max_vertices 30000 --epochs 18 --patience 18 --lr 3e-4 --device cuda --seed 0 \
      --output_dir "$2" >> /tmp/$4.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 45); do
      grep -q "device=cuda" /tmp/$4.log 2>/dev/null && { ok=1; break; }
      grep -q "device=cpu" /tmp/$4.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "$4: cuda (att $attempt)" >> /tmp/complete5_launch.log; wait $bg; echo "$4: exit $?" >> /tmp/complete5_launch.log; return; fi
    echo "$4 att $attempt: cpu/retry" >> /tmp/complete5_launch.log; kill $bg 2>/dev/null; sleep 3
  done
}
run 3 outputs/geomcv_fold3 "--geom_dim 5" geomcv_3
run 4 outputs/geomcv_fold4 "--geom_dim 5" geomcv_4
run 3 outputs/basefull_fold3 "" basefull_3
run 4 outputs/basefull_fold4 "" basefull_4
echo "ALL COMPLETE5 DONE" >> /tmp/complete5_launch.log
