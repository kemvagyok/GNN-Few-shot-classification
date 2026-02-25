#!/bin/bash
#SBATCH --job-name=surv_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.log

set -euo pipefail

PROJECT_DIR=/project/c_gnn42/few_shot_dipterv/GNN-Few-shot-classification
SIF=${PROJECT_DIR}/fewshotgnn.sif

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${PROJECT_DIR}/runs

echo "Project dir: ${PROJECT_DIR}"
echo "Container: ${SIF}"

module load singularity

srun singularity exec --nv \
  --bind ${PROJECT_DIR}:/workspace \
  ${SIF} \
  bash -lc "cd /workspace/src && python -u training-evaluation.py"
