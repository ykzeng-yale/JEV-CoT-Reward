#!/bin/bash
#SBATCH --job-name=jev-cuda-qual-v1b
#SBATCH --account=pi_fl426
#SBATCH --partition=gpu_h100,gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
ulimit -c 0
module purge
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 Transformers/4.55.2-gfbf-2024a
export HF_HOME=/home/yz2324/project_pi_fl426/yz2324/JEV-CoT-Reward/cache
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"
# Application footprint cap is 16 GiB; no credentials or gated model are needed.
python qualify_cuda.py
