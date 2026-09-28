#!/bin/bash
#SBATCH --job-name=jev-cuda-precision-v4
#SBATCH --account=pi_fl426
#SBATCH --partition=gpu_h100,gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
ulimit -c 0
module purge
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 Transformers/4.55.2-gfbf-2024a
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/yz2324/project_pi_fl426/yz2324/JEV-CoT-Reward/cache
MODEL_PATH="$HF_HOME/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
python scripts/run_guard_qualification.py --config configs/cuda_guard_precision_v4.json --model "$MODEL_PATH" --output outputs --wall-seconds 28200
