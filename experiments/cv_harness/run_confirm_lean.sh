#!/usr/bin/env bash
# Confirmatory: does the LEAN model (DINO+geom+verb-text; drop clip,slat,normals) match the FULL model?
# Both GEAL-pretrained (gnn_pretrain_k24.pt), k24, 5-fold. Matched so the comparison is self-contained.
set -u
cd /home/kraum/Prototype
export CUDA_VISIBLE_DEVICES=0          # free GPU only; NEVER touch GPU1 (other users)
PY=.venv-affordance/bin/python
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
PRE=outputs/gnn_pretrain_k24.pt

run_cfg () {  # $1=tag  $2=drop-arg
  local tag="$1" drop="$2" log="/tmp/confirm_${1}.log"
  for attempt in 1 2 3 4; do
    echo "### $tag attempt $attempt $(date +%H:%M:%S)" >> "$log"
    nice -n 10 timeout 5400 $PY "$SCR/train_gnn_cv.py" \
        --drop "$drop" --init_from "$PRE" --knn_k 24 --folds 0,1,2,3,4 \
        --tag "$tag" --device cuda >> "$log" 2>&1
    # success = the script reached its final summary line
    if grep -q "GNN CV DONE" "$log"; then echo "### $tag OK" >> "$log"; return 0; fi
    # only retry on a CUDA driver-init failure (the flaky ~1/3 case); otherwise stop
    if grep -qiE "CUDA error|CUDA driver|no CUDA-capable|initialization error|RuntimeError: CUDA" "$log"; then
      echo "### $tag CUDA-init fail, retrying" >> "$log"; sleep 5; continue
    fi
    echo "### $tag FAILED (non-CUDA); stop" >> "$log"; return 1
  done
  echo "### $tag gave up after retries" >> "$log"; return 1
}

run_cfg full_preclean ""                    # all features, pretrain, k24, 5-fold
run_cfg lean_preclean "clip,slat,normals"   # DINO+geom+verb-text only, pretrain, k24, 5-fold
echo "ALL CONFIRM DONE $(date +%H:%M:%S)" >> /tmp/confirm_full_preclean.log
echo "ALL CONFIRM DONE $(date +%H:%M:%S)" >> /tmp/confirm_lean_preclean.log
