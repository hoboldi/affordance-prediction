#!/usr/bin/env bash
# View-count x SLAT ablation. Renders N views ONCE per count, encodes CLIP+DINO from them, then trains
# the v11 recipe with SLAT on (sam3d_dim=8) and off (0). N=16 reuses the existing clean features (=v11).
# Usage: run_view_ablation.sh <GPU_ID> <N...>   e.g. run_view_ablation.sh 0 8 1 16
set -u
GPU="$1"; shift
cd /home/kraum/Prototype
ROOT=/home/datasets/customDatasets/cmr2
M=$ROOT/manifest.pseudolabeled.clean.jsonl
PY=.venv-affordance/bin/python
LOG=/tmp/abl_gpu$GPU.log
RES=docs/view_ablation.md
mkdir -p docs

layout() { case "$1" in 1) echo "43 1";; 2) echo "43 2";; 4) echo "33,52 2";; 8) echo "33,52 4";; 16) echo "33,52 8";; esac; }

for N in "$@"; do
  set -- $(layout "$N"); RINGS="$1"; AZ="$2"
  echo "[$(date +%H:%M)] === N=$N (rings=$RINGS az=$AZ) on GPU$GPU ===" >> "$LOG"

  if [ "$N" = 16 ]; then
    MAN="$M"; DINOF="vertex_dino.pt"; SLATS="0"          # 16-slat == v11 (already trained); only do no-SLAT
  else
    MAN="$ROOT/clip_v$N.jsonl"; DINOF="vertex_dino_v$N.pt"; SLATS="8 0"
    CUDA_VISIBLE_DEVICES=$GPU PYOPENGL_PLATFORM=egl PYTHONPATH=src $PY scripts/generate_vertex_semantics.py \
      --manifest "$M" --out_manifest "$MAN" --vsem_filename "vertex_semantics_v$N.pt" \
      --elevation_rings "$RINGS" --azimuths_per_ring "$AZ" --reduce_dim 128 --device cuda --skip_existing >> "$LOG" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU PYOPENGL_PLATFORM=egl PYTHONPATH=src $PY scripts/generate_vertex_dino.py \
      --manifest "$M" --out_manifest "$ROOT/dino_v$N.jsonl" --dino_filename "$DINOF" \
      --elevation_rings "$RINGS" --azimuths_per_ring "$AZ" --reduce_dim 128 --device cuda --skip_existing >> "$LOG" 2>&1
  fi

  for SLAT in $SLATS; do
    TAG=slat; [ "$SLAT" = 0 ] && TAG=noslat
    OUT=outputs/abl_v${N}_$TAG
    echo "[$(date +%H:%M)] train N=$N $TAG -> $OUT" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src $PY scripts/train_affordance.py \
      --manifest "$MAN" --output_dir "$OUT" --epochs 15 --device cuda --grad_accum 8 \
      --pos_dim 0 --vlm_dim 128 --dino_vertex_dim 128 --dino_filename "$DINOF" \
      --sam3d_dim "$SLAT" --contrastive_weight 1.0 >> "$LOG" 2>&1
    {
      echo "### N=$N  SLAT=$TAG  (GPU$GPU, $(date +%H:%M))"
      CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY scripts/eval_verb_conditioning.py \
        --ckpt "$OUT/last.pt" --manifest "$MAN" 2>/dev/null | grep -E "AUPRC=|VERDICT"
      echo ""
    } >> "$RES"
  done
done
echo "[$(date +%H:%M)] GPU$GPU DONE: $*" >> "$LOG"
