#!/bin/bash
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
retry() { local tag=$1; shift
  for attempt in $(seq 6); do
    : > /tmp/$tag.log
    CUDA_VISIBLE_DEVICES=0 nice -n 10 "$@" >> /tmp/$tag.log 2>&1 &
    bg=$!; ok=0
    for i in $(seq 90); do
      grep -q "device=cuda" /tmp/$tag.log 2>/dev/null && { ok=1; break; }
      grep -q "device=cpu" /tmp/$tag.log 2>/dev/null && break
      kill -0 $bg 2>/dev/null || break; sleep 1
    done
    if [ $ok -eq 1 ]; then echo "$tag: cuda (att $attempt)" >> /tmp/gealclean_launch.log; wait $bg; echo "$tag: exit $?" >> /tmp/gealclean_launch.log; return; fi
    echo "$tag att $attempt: cpu/retry" >> /tmp/gealclean_launch.log; kill $bg 2>/dev/null; sleep 3
  done
}
# 1) GNN k24 from-scratch, CLEAN labels, folds 0-4
retry gc_base timeout 9000 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" \
  --folds 0,1,2,3,4 --epochs 18 --lr 1e-3 --agg max --knn_k 24 --tag k24clean
# 2) GNN k24 GEAL-pretrained -> human-FT, CLEAN labels, folds 0-4
retry gc_pre timeout 9000 .venv-affordance/bin/python "$SCR/train_gnn_cv.py" \
  --folds 0,1,2,3,4 --epochs 18 --lr 1e-3 --agg max --knn_k 24 --init_from outputs/gnn_pretrain_k24.pt --tag k24preclean
# 3) MLP full-FT+geom, CLEAN labels, folds 0-4 (clean 'previous best' baseline)
for k in 0 1 2 3 4; do
  retry mlp_clean_f$k timeout 3000 .venv-affordance/bin/python scripts/finetune_human.py \
    --ckpt outputs/ov_concat_finepatch/best.pt --manifest "$SCR/manifest.cv.jsonl" --split "$SCR/cv_fold$k.json" \
    --no_freeze_backbone --geom_dim 5 --max_vertices 30000 --epochs 18 --patience 18 --lr 3e-4 --device cuda --seed 0 \
    --output_dir outputs/geomclean_fold$k
done
echo "GEAL-CLEAN EXPERIMENT DONE" >> /tmp/gealclean_launch.log
