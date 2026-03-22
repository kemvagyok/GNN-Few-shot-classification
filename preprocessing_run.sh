#!/bin/bash
#SBATCH --job-name=preprocessing_chestx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/project/c_gnn42/few_shot_dipterv/GNN-Few-shot-classification/logs/%x_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=0322sagibenedek@gmail.com

set -euo pipefail

PROJECT_DIR=/project/c_gnn42/few_shot_dipterv/GNN-Few-shot-classification
SIF=${PROJECT_DIR}/fewshotgnn.sif

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${PROJECT_DIR}/runs

echo "Project dir: ${PROJECT_DIR}"
echo "Container: ${SIF}"

module load singularity

singularity exec --nv \
  --bind ${PROJECT_DIR}:/workspace \
  ${SIF} \
  bash -lc "cd /workspace/src && python preprocessing.py --train_files_size 4000 --test_files_size 4000"