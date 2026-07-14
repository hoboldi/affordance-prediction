#!/usr/bin/env bash
# Fresh-server bootstrap for the ReVerb pipeline.
#
# Re-creates everything that does NOT travel with `git clone` (gitignored / external):
#   1. SAM3D submodule          (sam-3d-objects)
#   2. GEAL teacher code        (external/geal)
#   3. GEAL weights             (external/geal/ckpt/*.pt)        [only with --weights]
#   4. prints the remaining manual steps (env, CO3D, SAM3D checkpoints)
#
# Recommended transfer: `git clone` this branch on the new server, then run this script.
# (Do NOT copy the working dir — external/ and data/ are gitignored and are re-created here.)
#
# Usage:
#   bash scripts/bootstrap.sh             # submodule + GEAL code (light; no large downloads)
#   bash scripts/bootstrap.sh --weights   # also download GEAL checkpoints from HuggingFace
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WANT_WEIGHTS=0
for a in "$@"; do [ "$a" = "--weights" ] && WANT_WEIGHTS=1; done
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null 2>&1 || PY=python3

echo "==> [1/4] SAM3D submodule"
git submodule update --init sam-3d-objects || echo "    (submodule init failed — check network/access; not fatal)"

echo "==> [2/4] GEAL code -> external/geal (gitignored)"
if [ -d external/geal/.git ]; then
  echo "    already present"
else
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/DylanOrange/geal external/geal
fi

echo "==> [3/4] GEAL weights -> external/geal/ckpt/"
mkdir -p external/geal/ckpt
if [ "$WANT_WEIGHTS" = "1" ]; then
  "$PY" - <<'PY'
from huggingface_hub import hf_hub_download
import os, shutil
out = "external/geal/ckpt"; os.makedirs(out, exist_ok=True)
# Edit this list to the checkpoints you need (piad_seen is the default teacher).
for f in ["piad_seen.pt"]:  # also: piad_unseen.pt / laso_seen.pt / laso_unseen.pt
    p = hf_hub_download(repo_id="dylanorange/geal", filename=f, repo_type="dataset")
    shutil.copy(p, os.path.join(out, f)); print("  downloaded", f)
PY
else
  echo "    SKIP (pass --weights). Manual: download piad_seen.pt etc. from"
  echo "    https://huggingface.co/datasets/dylanorange/geal into external/geal/ckpt/"
fi

cat <<'NEXT'

==> [4/4] Remaining manual steps (need storage; see README + docs/data_layout.md):
  - Python env:     conda env create -f environment.yml  &&  conda activate reverb
                    pip install -r external/geal/requirements.txt   # GEAL inference deps
  - SAM3D weights:  follow sam-3d-objects/doc/setup.md  (HuggingFace facebook/sam-3d-objects)
  - CO3D:           download CO3D v2 sequences for the target categories

Then run the pipeline (README -> "Pipeline"):
  PYTHONPATH=src python scripts/prepare_co3d.py               --config configs/co3d.yaml
  PYTHONPATH=src python scripts/generate_sam3d.py             --dataset_dir data/co3d/sam3d_inputs --output_dir data/co3d/reconstructions
  PYTHONPATH=src python scripts/generate_vertex_dino.py       --manifest data/co3d/manifest.jsonl   # per-vertex DINOv2
  PYTHONPATH=src python scripts/precompute_geom.py            --manifest data/co3d/manifest.jsonl   # geometry descriptors
  PYTHONPATH=src python scripts/generate_geal_pseudolabels.py --manifest data/co3d/manifest.jsonl --ckpt external/geal/ckpt/piad_seen.pt
  PYTHONPATH=src python experiments/cv_harness/train_gnn_pretrain.py   # Stage 1: GEAL distillation
  PYTHONPATH=src python experiments/cv_harness/train_gnn_cv.py         # Stage 2: 5-fold human finetune
NEXT
echo "bootstrap done."
