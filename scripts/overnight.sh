#!/usr/bin/env bash
# Fully autonomous overnight run. Self-gates on v13/v15, then both GPUs in parallel.
# Stage 0: eval v13/v15 | Stage1 GPU0: camera(best-effort)+loss-sweep ; GPU1: loss-sweep
# Stage 2: view x SLAT ablation (both) | cross-attn folded into the loss sweep. Continue-on-failure.
set -u
cd /home/kraum/Prototype
ROOT=/home/datasets/customDatasets/cmr2
M=$ROOT/manifest.pseudolabeled.clean.jsonl
PY=.venv-affordance/bin/python
RES=docs/overnight_results.md; mkdir -p docs
echo "# Overnight results — started $(date)" >> "$RES"

# ---- self-gate: wait for v13 + v15 to finish ----
until { grep -q V13_EXIT /tmp/v13-dinolarge.log 2>/dev/null || ! pgrep -f affordance_mlp_v13_dinolarge >/dev/null; } \
   && { grep -q V15_EXIT /tmp/v15-ov-nc.log 2>/dev/null || ! pgrep -f affordance_mlp_v15_ov_nc >/dev/null; }; do sleep 60; done
sleep 30
echo "[$(date +%H:%M)] gates cleared; starting" >> /tmp/overnight.log

# ---- Stage 0: eval v13 / v15 (CPU) ----
for R in v13_dinolarge:affordance_mlp_v13_dinolarge v15_ovnc:affordance_mlp_v15_ov_nc; do
  nm=${R%%:*}; dir=${R##*:}
  { echo "### STAGE0 $nm  $(date +%H:%M)";
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src $PY scripts/eval_verb_conditioning.py --ckpt outputs/$dir/last.pt --manifest "$M" 2>/dev/null | grep -E "AUPRC=|VERDICT"; echo ""; } >> "$RES"
done

# ---- GPU0 sequence ----
(
  # camera validation (best-effort, bounded) — produce camera.pt for morning projection check
  IN=tmp_cam_input; OUTC=tmp_cam_out; rm -rf $IN $OUTC; mkdir -p $IN/images $IN/masks $OUTC
  IMG=$(ls $ROOT/sam3d_inputs/images/* 2>/dev/null | head -1)
  if [ -n "${IMG:-}" ]; then
    STEM=$(basename "$IMG"); STEM=${STEM%.*}
    cp "$IMG" $IN/images/ 2>/dev/null; cp "$ROOT/sam3d_inputs/masks/$STEM."* $IN/masks/ 2>/dev/null
    echo "[$(date +%H:%M)] camera: SAM3D on $STEM" >> /tmp/cam.log
    timeout 2400 env AFFORDANCE_CUDA_DEVICE=0 docker compose run --rm autonomous-pipeline \
      bash -lc "PYTHONPATH=/workspace/src python scripts/generate_sam3d.py --dataset_dir /workspace/$IN --output_dir /workspace/$OUTC" >> /tmp/cam.log 2>&1 || true
    ls $OUTC/*/camera.pt >> /tmp/cam.log 2>&1 && echo "[$(date +%H:%M)] camera.pt OK" >> /tmp/cam.log || echo "[$(date +%H:%M)] camera.pt MISSING (do interactively)" >> /tmp/cam.log
  else echo "no sam3d input image found; skip camera" >> /tmp/cam.log; fi
  bash scripts/run_loss_sweep.sh 0 \
    "abl_large_l0|vertex_dino_large.pt|256|--contrastive_weight 0" \
    "abl_large_l0_noslat|vertex_dino_large.pt|256|--contrastive_weight 0 --sam3d_dim 0" \
    "abl_crossattn_large|vertex_dino_large.pt|256|--contrastive_weight 1.0 --verb_conditioning cross_attn"
  bash scripts/run_view_ablation.sh 0 8 1
) > /tmp/overnight_gpu0.log 2>&1 &

# ---- GPU1 sequence ----
(
  bash scripts/run_loss_sweep.sh 1 \
    "abl_large_l03|vertex_dino_large.pt|256|--contrastive_weight 0.3" \
    "abl_base_l0|vertex_dino.pt|128|--contrastive_weight 0" \
    "abl_crossattn_large_l0|vertex_dino_large.pt|256|--contrastive_weight 0 --verb_conditioning cross_attn"
  bash scripts/run_view_ablation.sh 1 4 2 16
) > /tmp/overnight_gpu1.log 2>&1 &

wait
echo "# Overnight DONE $(date)" >> "$RES"
echo "[$(date +%H:%M)] OVERNIGHT COMPLETE" >> /tmp/overnight.log
