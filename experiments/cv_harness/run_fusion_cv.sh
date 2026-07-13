#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
run() {  # $1=variant(mean|max|smax) 
  for attempt in $(seq 6); do
    rm -rf "outputs/fus_${1}_fold0"; : > /tmp/fus_${1}.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 2400 .venv-affordance/bin/python scripts/finetune_human.py \
      --ckpt outputs/ov_concat_finepatch/best.pt --manifest "$SCR/manifest.cv.jsonl" --split "$SCR/cv_fold0.json" \
      --no_freeze_backbone --geom_dim 5 --dino_filename "vertex_dino_fine_${1}.pt" \
      --max_vertices 30000 --epochs 18 --patience 18 --lr 3e-4 --device cuda --seed 0 \
      --output_dir "outputs/fus_${1}_fold0" >> /tmp/fus_${1}.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 45); do
      grep -q "device=cuda" /tmp/fus_${1}.log 2>/dev/null && { ok=1; break; }
      grep -q "device=cpu" /tmp/fus_${1}.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "fus_${1}: cuda (att $attempt)" >> /tmp/fus_launch.log; wait $bg; echo "fus_${1}: exit $?" >> /tmp/fus_launch.log; return; fi
    echo "fus_${1} att $attempt: cpu/retry" >> /tmp/fus_launch.log; kill $bg 2>/dev/null; sleep 3
  done
}
run mean
run max
run smax
echo "ALL FUSION CV DONE" >> /tmp/fus_launch.log
