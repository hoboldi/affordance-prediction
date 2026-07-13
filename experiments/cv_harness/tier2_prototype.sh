#!/bin/bash
# Tier-2 prototype: re-extract wideband CLIP+DINO (rings -52,-33,33,52 = upper band + lower mirror, 32 views)
# for ~6 grasp objects, then measure visibility/grasp-coverage gain. Arg $1 = GPU id to use.
GPU="${1:-0}"
cd /home/kraum/Prototype
export AFFORD_ELEV_MIN_DEG=-60 AFFORD_ELEV_MAX_DEG=60 PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src
PY=.venv-affordance/bin/python
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
RES=/tmp/tier2_proto_result.txt
RINGS="-52,-33,33,52"
echo "=== TIER2 PROTO start $(date '+%H:%M') on GPU$GPU (rings $RINGS, 32 views) ===" > $RES
$PY scripts/generate_vertex_semantics.py --manifest /tmp/tier2_proto.jsonl --out_manifest /tmp/wb_clip.jsonl \
  --vsem_filename vertex_semantics_wideband.pt --elevation_rings="$RINGS" --azimuths_per_ring 8 \
  --reduce_dim 128 --device cuda --skip_existing > /tmp/tier2_proto_ext.log 2>&1
echo "  CLIP extract rc=$? $(date '+%H:%M')" >> $RES
$PY scripts/generate_vertex_dino.py --manifest /tmp/tier2_proto.jsonl --out_manifest /tmp/wb_dino.jsonl \
  --dino_filename vertex_dino_wideband.pt --dino_image_size 448 --elevation_rings="$RINGS" --azimuths_per_ring 8 \
  --reduce_dim 128 --device cuda --skip_existing >> /tmp/tier2_proto_ext.log 2>&1
echo "  DINO extract rc=$? $(date '+%H:%M')" >> $RES
CUDA_VISIBLE_DEVICES="" $PY "$SCR/tier2_coverage_check.py" >> $RES 2>&1
echo "=== TIER2 PROTO DONE $(date '+%H:%M') ===" >> $RES
