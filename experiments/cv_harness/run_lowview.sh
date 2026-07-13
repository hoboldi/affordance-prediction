#!/usr/bin/env bash
# Extend view-count ablation to the LOW end {1,2,3 views}. Single ring (43 deg, midpoint) since 2 rings
# can't split below 4 views. Extract then train lean (DINO+geom+verb, pretrain, k24, 3-fold). Runs on $CUDA_VISIBLE_DEVICES.
# Self-gates: waits for the {4,8,16,24} view-count training (vt_DONE) to free GPU1 before starting.
set -u
cd /home/kraum/Prototype
PY=.venv-affordance/bin/python
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
DR=/home/datasets/customDatasets/cmr2
PRE=outputs/gnn_pretrain_k24.pt

# wait for the current view-count training batch to finish (frees GPU1)
until [ -f /tmp/vt_DONE.marker ]; do sleep 60; done
echo "vt_DONE seen, starting low-view $(date +%H:%M:%S)" >> /tmp/lv_gate.log

extract () {  # $1=N  $2=az   (single ring 43 deg)
  local N="$1"; local az="$2"; local log="/tmp/vc_extract_${N}.log"
  for a in 1 2 3; do
    echo "### vc$N (1 ring, az=$az) attempt $a" >> "$log"
    nice -n 10 timeout 3600 $PY scripts/generate_vertex_dino.py \
      --manifest "$SCR/manifest.cv.jsonl" --data_root "$DR" \
      --dino_image_size 448 --reduce_dim 128 --elevation_rings "43" --azimuths_per_ring "$az" \
      --dino_filename "vertex_dino_vc${N}.pt" --skip_existing --device cuda >> "$log" 2>&1
    grep -qE "Done: .* ok" "$log" && return 0
    grep -qiE "CUDA error|CUDA driver|initialization error|no CUDA-capable" "$log" && { sleep 5; continue; }
    return 1
  done
}
train_vc () {  # $1=N
  local N="$1"; local log="/tmp/vt_vc${N}.log"
  for a in 1 2 3; do
    echo "### vt_vc$N attempt $a" >> "$log"
    nice -n 10 timeout 5400 $PY "$SCR/train_gnn_cv.py" --drop clip,slat,normals \
      --init_from "$PRE" --knn_k 24 --dino_file "vertex_dino_vc${N}.pt" --folds 0,1,2 \
      --tag vt_vc${N} --device cuda >> "$log" 2>&1
    grep -q "GNN CV DONE" "$log" && return 0
    grep -qiE "CUDA error|CUDA driver|initialization error|no CUDA-capable" "$log" && { sleep 5; continue; }
    return 1
  done
}

extract 1 1
extract 2 2
extract 3 3
train_vc 1
train_vc 2
train_vc 3
echo "VC LOW DONE" > /tmp/vt_low_DONE.marker
