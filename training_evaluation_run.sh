#!/bin/bash
#SBATCH --job-name=few_shot_gnn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/project/c_gnn42/few_shot_dipterv/GNN-Few-shot-classification/logs/%x_%j.log
#SBATCH --array=0-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=0322sagibenedek@gmail.com

set -euo pipefail

PROJECT_DIR=/project/c_gnn42/few_shot_dipterv/GNN-Few-shot-classification
SIF=${PROJECT_DIR}/fewshotgnn.sif

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export WANDB_API_KEY=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

mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${PROJECT_DIR}/runs

echo "Project dir: ${PROJECT_DIR}"
echo "Container: ${SIF}"
echo "SLURM ARRAY ID: $SLURM_ARRAY_TASK_ID"

module load singularity

PORT=$((29500 + SLURM_ARRAY_TASK_ID))

singularity exec --nv \
  --bind ${PROJECT_DIR}:/workspace \
  ${SIF} \
    bash -lc "cd /workspace/src && python training-evaluation.py --run_id $SLURM_ARRAY_TASK_ID && python aggregate_results.py --dataset ChestX" 
#  bash -lc "cd /workspace/src && torchrun \
#    --nproc_per_node=$SLURM_GPUS_ON_NODE \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint=localhost:${PORT} \
#    training-evaluation.py --run_id $SLURM_ARRAY_TASK_ID"

