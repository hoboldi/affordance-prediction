#!/bin/bash
#SBATCH --nodelist=cc-gpu-n04
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out

# Apply patches to submodule files we can't push to
cp ~/affordance-prediction/patches/inference.py \
   ~/affordance-prediction/sam-3d-objects/notebook/inference.py

source ~/affordance-prediction/sam-3d-objects/.venv/bin/activate
python ~/affordance-prediction/dataset_pipeline.py \
  --dataset_dir ~/affordance-prediction/dataset_dir \
  --output_dir ~/affordance-prediction/output \
  --no_masks \
  --skip_existing
