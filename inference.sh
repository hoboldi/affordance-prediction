#!/bin/bash
#SBATCH --nodelist=cc-gpu-n04
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out

VENV=~/affordance-prediction/sam-3d-objects/.venv
REQUIREMENTS=~/affordance-prediction/sam-3d-objects/requirements.only_inference.txt

if [ ! -f "$VENV/bin/activate" ] || ! uv pip check --python "$VENV/bin/python" > /dev/null 2>&1; then
    uv venv "$VENV" --python 3.11
    uv pip install --python "$VENV/bin/python" -r "$REQUIREMENTS"
fi

source "$VENV/bin/activate"
python ~/affordance-prediction/dataset_pipeline.py \
  --dataset_dir ~/affordance-prediction/dataset \
  --output_dir ~/affordance-prediction/output \
  --no_masks \
  --skip_existing
