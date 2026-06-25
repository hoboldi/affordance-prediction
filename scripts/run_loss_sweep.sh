#!/usr/bin/env bash
# Loss/conditioning sweep on the strong clean+DINO base (training-only, reuses features).
# Usage: run_loss_sweep.sh <GPU> <CONFIG...>   CONFIG = name|dino_filename|dino_dim|extra flags
set -u
GPU="$1"; shift
cd /home/kraum/Prototype
ROOT=/home/datasets/customDatasets/cmr2
M=$ROOT/manifest.pseudolabeled.clean.jsonl
PY=.venv-affordance/bin/python
LOG=/tmp/loss_gpu$GPU.log
RES=docs/overnight_results.md
mkdir -p docs

for CONF in "$@"; do
  IFS='|' read -r NAME DINOF DDIM EXTRA <<< "$CONF"
  OUT=outputs/$NAME
  echo "[$(date +%H:%M)] === $NAME  dino=$DINOF dim=$DDIM  extra=[$EXTRA]  GPU$GPU ===" >> "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src $PY scripts/train_affordance.py \
    --manifest "$M" --output_dir "$OUT" --epochs 30 --device cuda --grad_accum 8 \
    --pos_dim 0 --vlm_dim 128 --dino_vertex_dim "$DDIM" --dino_filename "$DINOF" $EXTRA >> "$LOG" 2>&1 || \
    { echo "[$(date +%H:%M)] $NAME TRAIN FAILED" >> "$LOG"; continue; }
  {
    echo "### $NAME  ($EXTRA)  $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY scripts/eval_verb_conditioning.py \
      --ckpt "$OUT/last.pt" --manifest "$M" 2>/dev/null | grep -E "AUPRC=|VERDICT"
    echo ""
  } >> "$RES"
done
echo "[$(date +%H:%M)] GPU$GPU loss-sweep DONE" >> "$LOG"
