#!/bin/bash
#SBATCH --job-name=vlm_one
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/vlm_one_%j.out

# Run ONE model through all tasks/prompts. Usage:
#   sbatch slurm/run_one.sbatch qwen2_vl_7b
# Valid names: llava15_7b qwen2_vl_7b qwen25_vl_7b internvl3_8b llama32_11b_vision

set -euo pipefail
MODEL=${1:?usage: sbatch slurm/run_one.sbatch <model_name>}

cd /work/mech-ai-scratch/anneka/food/vlm_benchmark
mkdir -p logs

module load python
source /work/mech-ai-scratch/anneka/envs/vlm_bench/bin/activate

export HF_HOME=/work/mech-ai-scratch/anneka/hf_cache
export TOKENIZERS_PARALLELISM=false

echo "Running model: $MODEL"
python -m src.run_benchmark config.yaml --model "$MODEL"
