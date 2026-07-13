#!/usr/bin/env bash
# View-count ablation TRAINING: lean model (DINO+geom+verb, pretrain, k24) on DINO fused from {4,8,16,24} views.
# 16 = deployed vertex_dino_fine.pt. 3 folds each -> AUPRC vs #views curve. Runs on $CUDA_VISIBLE_DEVICES.
set -u
cd /home/kraum/Prototype
PY=.venv-affordance/bin/python
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
PRE=outputs/gnn_pretrain_k24.pt

train_vc () {  # $1=N (views)  $2=dino_file
  local N="$1"; local df="$2"; local log="/tmp/vt_vc${N}.log"
  for attempt in 1 2 3; do
    echo "### vt_vc$N ($df) attempt $attempt" >> "$log"
    nice -n 10 timeout 5400 $PY "$SCR/train_gnn_cv.py" --drop clip,slat,normals \
      --init_from "$PRE" --knn_k 24 --dino_file "$df" --folds 0,1,2 \
      --tag vt_vc${N} --device cuda >> "$log" 2>&1
    if grep -q "GNN CV DONE" "$log"; then echo "### vt_vc$N OK" >> "$log"; return 0; fi
    if grep -qiE "CUDA error|CUDA driver|initialization error|no CUDA-capable" "$log"; then
      echo "### vt_vc$N CUDA fail, retry" >> "$log"; sleep 5; continue; fi
    echo "### vt_vc$N FAILED (non-CUDA)" >> "$log"; return 1
  done
}

train_vc 4  vertex_dino_vc4.pt
train_vc 8  vertex_dino_vc8.pt
train_vc 16 vertex_dino_fine.pt     # deployed 16-view
train_vc 24 vertex_dino_vc24.pt
echo "VC TRAIN DONE" > /tmp/vt_DONE.marker
