#!/usr/bin/env bash
cd /home/kraum/Prototype
LOG=/tmp/crossattn.log
M=/home/datasets/customDatasets/cmr2/manifest.pseudolabeled.clean.jsonl
KL=/tmp/crossattn_kill.log
echo "[$(date +%H:%M)] watcher armed: kill after ep20 completes" > "$KL"
# wait for ep20 val result line
until tr '\r' '\n' < "$LOG" | grep -qE "INFO epoch +20/30"; do sleep 15; done
echo "[$(date +%H:%M)] ep20 complete:" >> "$KL"
tr '\r' '\n' < "$LOG" | grep -E "INFO epoch +20/30|new best" | tail -2 >> "$KL"
sleep 3
# kill ONLY the train python (not the gate bash that needs to hand off to finer-patch)
P=$(ps -eo pid,args | awk '/train_affordance\.py/ && /base_crossattn_nocontrast/ && /python/ && !/bash -c/ {print $1}')
echo "[$(date +%H:%M)] killing train python: $P" >> "$KL"
kill -9 $P 2>/dev/null
sleep 8
# authoritative per-verb eval on both checkpoints
{
  echo "### cross-attn (stopped after ep20) — best.pt PER-VERB  $(date +%H:%M)"
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv-affordance/bin/python scripts/eval_verb_conditioning.py \
    --ckpt outputs/base_crossattn_nocontrast/best.pt --manifest "$M" 2>/dev/null | grep -E "AUPRC=|VERDICT"
  echo "### cross-attn (stopped after ep20) — last.pt PER-VERB"
  CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv-affordance/bin/python scripts/eval_verb_conditioning.py \
    --ckpt outputs/base_crossattn_nocontrast/last.pt --manifest "$M" 2>/dev/null | grep -E "AUPRC=|VERDICT"
  echo ""
} >> docs/overnight_results.md
echo "[$(date +%H:%M)] per-verb evals appended; finer-patch gate will fire next" >> "$KL"
