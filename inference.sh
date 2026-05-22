#!/bin/bash
#SBATCH --nodelist=cc-gpu-n04
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out

source ~/affordance-prediction/sam-3d-objects/.venv/bin/activate
python ~/affordance-prediction/dataset_pipeline.py \
  --dataset_dir ~/affordance-prediction/dataset \
  --output_dir ~/affordance-prediction/output \
  --no_masks \
  --skip_existing
