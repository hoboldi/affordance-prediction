#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
for sig in 0.1 0.25 0.5; do
  tag=$(echo $sig | tr -d '.')
  for attempt in $(seq 6); do
    rm -rf outputs/vjit_${tag}_fold0; : > /tmp/vjit_${tag}.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 timeout 2400 .venv-affordance/bin/python scripts/finetune_human.py \
      --ckpt outputs/ov_concat_finepatch/best.pt --manifest "$SCR/manifest.cv.jsonl" --split "$SCR/cv_fold0.json" \
      --no_freeze_backbone --verb_jitter $sig --max_vertices 30000 --epochs 18 --patience 18 --lr 3e-4 \
      --device cuda --seed 0 --output_dir outputs/vjit_${tag}_fold0 >> /tmp/vjit_${tag}.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 40); do
      grep -q "device=cuda" /tmp/vjit_${tag}.log 2>/dev/null && { ok=1; break; }
      grep -q "device=cpu" /tmp/vjit_${tag}.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "sig $sig: cuda (attempt $attempt)" >> /tmp/vjit_launch.log; wait $bg; echo "sig $sig: exit $?" >> /tmp/vjit_launch.log; break; fi
    echo "sig $sig attempt $attempt: cpu/retry" >> /tmp/vjit_launch.log; kill $bg 2>/dev/null; sleep 3
  done
done
echo "ALL VJIT DONE" >> /tmp/vjit_launch.log
