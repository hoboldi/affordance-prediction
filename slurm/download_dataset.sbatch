#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --output=slurm/logs/%j_download_dataset.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=all

export HF_TOKEN=$(cat .hf_token)

uv run python scripts/download_dataset.py
