#!/bin/bash
#SBATCH --job-name=vlm_bench
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/vlm_bench_%A_%a.out

# One model per array task -> 5 models run in parallel on 5 GPUs.
# Adjust --array range to match the number of models in config.yaml.

set -euo pipefail
cd /work/mech-ai-scratch/anneka/food/vlm_benchmark
mkdir -p logs

module load ml-gpu 2>/dev/null || true
source activate vlm_bench 2>/dev/null || conda activate vlm_bench

# Keep HF cache on scratch, not home quota
export HF_HOME=/work/mech-ai-scratch/anneka/hf_cache
export TOKENIZERS_PARALLELISM=false

MODELS=(llava15_7b qwen2_vl_7b qwen25_vl_7b internvl3_8b llama32_11b_vision)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL"
python -m src.run_benchmark config.yaml --model "$MODEL"
