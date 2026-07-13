#!/bin/bash
# Learning-curve + geometry-human-FT battery — runs sequentially, logs trained-mean per run.
cd /home/kraum/Prototype
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src
PY=.venv-affordance/bin/python
MAN=/home/datasets/customDatasets/cmr2/manifest.finepatch.jsonl
RES=/tmp/lc_results.txt
ev() { timeout 500 $PY scripts/eval_human_gt.py --ckpt "$1" --source model --device cuda --split human_split.json --split_part val 2>&1 | grep -iE "trained-verb mean|overall mean" | tr '\n' ' '; }
echo "=== LC BATTERY START $(date '+%H:%M') | baselines: MLP 0.580 / GEAL 0.593 trained ===" > $RES
# (2) head-only learning curve
for N in 50 100 172; do
  SP=human_split_n$N.json; [ "$N" = 172 ] && SP=human_split.json
  echo ">>> head-only N=$N start $(date '+%H:%M')" >> $RES
  timeout 3000 $PY scripts/finetune_human.py --ckpt outputs/ov_concat_finepatch/best.pt --split $SP --manifest $MAN --freeze_backbone --epochs 40 --patience 10 --lr 1e-4 --device cuda --seed 0 --output_dir outputs/lc_head_n$N >/tmp/lc_head_n$N.log 2>&1
  echo "  N=$N best: $(ev outputs/lc_head_n$N/best.pt)" >> $RES
  echo "  N=$N last: $(ev outputs/lc_head_n$N/last.pt)" >> $RES
done
# (4) geometry + human-FT (does human supervision unlock geometry?)
echo ">>> geom+human-FT start $(date '+%H:%M')" >> $RES
timeout 3000 $PY scripts/finetune_human.py --ckpt outputs/mlp_geom/best.pt --split human_split.json --manifest $MAN --freeze_backbone --epochs 40 --patience 10 --lr 1e-4 --device cuda --seed 0 --output_dir outputs/lc_geomft >/tmp/lc_geomft.log 2>&1
echo "  geomft best: $(ev outputs/lc_geomft/best.pt)" >> $RES
echo "  geomft last: $(ev outputs/lc_geomft/last.pt)" >> $RES
# (3) full-FT check LAST (timeout-guarded so a perf-stall can't block the priorities)
echo ">>> full-FT n172 start $(date '+%H:%M') (timeout-guarded)" >> $RES
timeout 2400 $PY scripts/finetune_human.py --ckpt outputs/ov_concat_finepatch/best.pt --split human_split.json --manifest $MAN --no_freeze_backbone --max_vertices 20000 --lr 3e-5 --epochs 40 --patience 10 --device cuda --seed 0 --output_dir outputs/lc_full_n172 >/tmp/lc_full_n172.log 2>&1
[ -f outputs/lc_full_n172/best.pt ] && echo "  full best: $(ev outputs/lc_full_n172/best.pt)" >> $RES || echo "  full-FT: no checkpoint (perf-blocked/timeout)" >> $RES
echo "=== LC BATTERY DONE $(date '+%H:%M') ===" >> $RES
