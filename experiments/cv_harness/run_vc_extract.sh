#!/usr/bin/env bash
# View-count ablation: re-extract per-vertex DINO fusing {4,8,24} views (16 = existing vertex_dino_fine.pt).
# Same fine settings as deployed (448px/32x32, reduce_dim128, seed0, rings 33,52) -> view count is the ONLY change.
# Writes vertex_dino_vc<N>.pt into each recon dir (additive; cleaned up after training). Runs on $CUDA_VISIBLE_DEVICES.
set -u
cd /home/kraum/Prototype
PY=.venv-affordance/bin/python
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
DR=/home/datasets/customDatasets/cmr2

extract () {  # $1=N (views)  $2=az_per_ring  (2 rings 33,52)
  local N="$1"; local az="$2"; local log="/tmp/vc_extract_${N}.log"
  for attempt in 1 2 3; do
    echo "### vc$N (az=$az) attempt $attempt" >> "$log"
    nice -n 10 timeout 7200 $PY scripts/generate_vertex_dino.py \
      --manifest "$SCR/manifest.cv.jsonl" --data_root "$DR" \
      --dino_image_size 448 --reduce_dim 128 --elevation_rings "33,52" --azimuths_per_ring "$az" \
      --dino_filename "vertex_dino_vc${N}.pt" --skip_existing --device cuda >> "$log" 2>&1
    if grep -qE "Done: .* ok" "$log"; then echo "### vc$N OK" >> "$log"; return 0; fi
    if grep -qiE "CUDA error|CUDA driver|initialization error|no CUDA-capable" "$log"; then
      echo "### vc$N CUDA fail, retry" >> "$log"; sleep 5; continue; fi
    echo "### vc$N FAILED (non-CUDA)" >> "$log"; return 1
  done
}

extract 4  2      # 2 rings x 2 az
extract 8  4      # 2 rings x 4 az
extract 24 12     # 2 rings x 12 az
echo "VC EXTRACT DONE" > /tmp/vc_extract.marker
