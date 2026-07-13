#!/usr/bin/env bash
# Lean-model ablations (drop DINO / geom / verb from the FINAL model) + label-efficiency curve
# (pretrain vs scratch, fixed val=58). All lean base (drop clip,slat,normals), k24. Runs on $CUDA_VISIBLE_DEVICES.
set -u
cd /home/kraum/Prototype
PY=.venv-affordance/bin/python
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
PRE=outputs/gnn_pretrain_k24.pt
LEAN="clip,slat,normals"          # the lean base: everything the final model doesn't use

run () {  # $1=tag  ; rest = extra args to train_gnn_cv.py
  local tag="$1"; shift; local log="/tmp/le_${tag}.log"
  for attempt in 1 2 3 4; do
    echo "### $tag attempt $attempt" >> "$log"
    nice -n 10 timeout 5400 $PY "$SCR/train_gnn_cv.py" --knn_k 24 --tag "$tag" --device cuda "$@" >> "$log" 2>&1
    if grep -q "GNN CV DONE" "$log"; then echo "### $tag OK" >> "$log"; return 0; fi
    if grep -qiE "CUDA error|CUDA driver|no CUDA-capable|initialization error|RuntimeError: CUDA" "$log"; then
      echo "### $tag CUDA-init fail, retry" >> "$log"; sleep 5; continue; fi
    echo "### $tag FAILED (non-CUDA)" >> "$log"; return 1
  done
}

# ── Group A: ablate the FINAL (lean) model's own components (5-fold, pretrain) ──
run leanabl_nodino  --drop "${LEAN},dino"  --init_from "$PRE" --folds 0,1,2,3,4   # geom + verb only
run leanabl_nogeom  --drop "${LEAN},geom"  --init_from "$PRE" --folds 0,1,2,3,4   # DINO + verb only
run leanabl_noverb  --drop "${LEAN}" --no_verb --init_from "$PRE" --folds 0,1,2,3,4  # DINO+geom, verb-agnostic

# ── Group B: label-efficiency (single split human_split.json, val=58 fixed) ──
for cap in 10 25 50 100 0; do          # 0 = no cap = full 172
  run le_pre_${cap}  --drop "$LEAN" --init_from "$PRE" --split_file human_split.json --train_cap ${cap}  # pretrained
  run le_scr_${cap}  --drop "$LEAN"                    --split_file human_split.json --train_cap ${cap}  # scratch
done

echo "ALL LEANABL DONE" >> /tmp/le_leanabl_nodino.log
echo "ALL LEANABL DONE" >> /tmp/le_DONE.marker
