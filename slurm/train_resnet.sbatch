#!/bin/bash
#SBATCH --job-name=resnet_cls
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/resnet_%j.out

# Train the ResNet-18 classification baseline for one task. Usage:
#   sbatch slurm/train_resnet.sbatch donation_type
#   sbatch slurm/train_resnet.sbatch defects
#   sbatch slurm/train_resnet.sbatch freshness
# Reminder: don't submit while an interactive GPU Jupyter session is up
# (AssocGrpGRES quota contention).

set -euo pipefail
TASK=${1:?usage: sbatch slurm/train_resnet.sbatch <donation_type|defects|freshness>}

cd /work/mech-ai-scratch/anneka/food/vlm_benchmark
mkdir -p logs

module load python
source /work/mech-ai-scratch/anneka/envs/vlm_bench/bin/activate

export HF_HOME=/work/mech-ai-scratch/anneka/hf_cache
export TORCH_HOME=/work/mech-ai-scratch/anneka/hf_cache/torch

EXTRA=""
if [ "$TASK" = "defects" ]; then
    EXTRA="--epochs 40 --augment"   # tiny positive counts: more epochs + light aug
fi

echo "Training ResNet-18 baseline: $TASK"
python -m src.train_resnet config.yaml --task "$TASK" $EXTRA
