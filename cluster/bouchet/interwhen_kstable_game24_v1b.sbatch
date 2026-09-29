#!/bin/bash
#SBATCH --job-name=interwhen-kstable-v1b
#SBATCH --account=pi_fl426
#SBATCH --partition=gpu_h100,gpu_b200,gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
test -s "$MODEL_PATH/config.json"
test -s "$MODEL_PATH/model.safetensors.index.json"
for shard in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
  test -s "$MODEL_PATH/$shard"
done
RUN_DIR="runs/interwhen-kstable-game24-v1b-${SLURM_JOB_ID}"
python scripts/run_interwhen_kstable_adaptation.py \
  --model "$MODEL_PATH" --output "$RUN_DIR" --wall-seconds 19800
sha256sum "$RUN_DIR"/* > "$RUN_DIR/SHA256SUMS"
