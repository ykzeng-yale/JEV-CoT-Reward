#!/usr/bin/env bash
# Frozen real-GPU runtime qualification; no production endpoints or Jev credentials.
#SBATCH --job-name=recoma-final-qual-v3f
#SBATCH --account=pi_fl426
#SBATCH --qos=normal
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu:h200|gpu:b200|gpu:rtx_pro_6000_blackwell"
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3/logs/%x-%j.out
#SBATCH --error=/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3/logs/%x-%j.err
#SBATCH --export=NONE
set -euo pipefail
ROOT=/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3
INPUTS="$ROOT/qualification-inputs-v3f"
JOB_DIR="$ROOT/runs/qualification-$SLURM_JOB_ID"
DEADLINE_EPOCH=1790821618
remaining=$((DEADLINE_EPOCH - $(date +%s)))
if (( remaining <= 0 )); then exit 75; fi
mkdir "$JOB_DIR"
mkdir -p "$JOB_DIR/env" "$JOB_DIR/results" "$JOB_DIR/home"
export TASK_WORK_HOME="$JOB_DIR/home"
export SLURM_EXPORT_ENV=ALL
unset TYPESAFE_API_KEY OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 Transformers/4.55.2-gfbf-2024a
cd "$INPUTS"
sha256sum -c SHA256SUMS > "$JOB_DIR/env/input-checksums.txt"
module list 2> "$JOB_DIR/env/modules.txt"
nvidia-smi -q > "$JOB_DIR/env/nvidia-smi.txt"
python -m venv --system-site-packages "$JOB_DIR/venv"
"$JOB_DIR/venv/bin/python" -m pip install --no-index --find-links "$INPUTS/wheelhouse" --require-hashes -r "$INPUTS/recoma_discoveryworld_hf_v1.lock" > "$JOB_DIR/env/pip-install.log" 2>&1
"$JOB_DIR/venv/bin/python" -m pip freeze > "$JOB_DIR/env/requirements-resolved.txt"
export MODEL_PATH=/home/yz2324/project_pi_fl426/yz2324/JEV-CoT-Reward/cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
export MODEL=Qwen/Qwen3-4B-Instruct-2507
export HF_HOME=/nfs/roberts/project/pi_fl426/yz2324/JEV-CoT-Reward/cache
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 MPLBACKEND=Agg OMP_NUM_THREADS=1
export PYTHONPATH="$INPUTS/source/recoma:$INPUTS/source/discoveryworld${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MAX_ENV_CALLS=30 SEED=0 SEEDS='[5]' OUTPUT_DIR="$JOB_DIR/results"
export SCENARIO_PREFIXES='["Combinatorial Chemistry", "It\u0027s (not) Rocket Science!"]'
export RECOMA_USAGE_LOG="$JOB_DIR/results/worker-0-calls.jsonl"
export XDG_CACHE_HOME="$JOB_DIR/home/cache" MPLCONFIGDIR="$JOB_DIR/home/matplotlib"
export TRITON_CACHE_DIR="$JOB_DIR/home/triton" TORCH_HOME="$JOB_DIR/home/torch"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
# File size 128MiB, no core dumps; project phase storage ceiling 2GiB.
ulimit -f 131072
ulimit -c 0
cd "$INPUTS/source/discoveryworld"
status=0
remaining=$((DEADLINE_EPOCH - $(date +%s)))
if (( remaining <= 0 )); then exit 75; fi
timeout --signal=TERM --kill-after=15s "${remaining}s" "$JOB_DIR/venv/bin/python" "$INPUTS/run_bounded_recoma_worker_v3.py" --manifest "$INPUTS/qualification_manifest.json" --config "$INPUTS/recoma_discoveryworld_react_full_v1.jsonnet" --worker-index 0 --output-dir "$JOB_DIR/results/worker-0" &
worker_pid=$!
(
  while kill -0 "$worker_pid" 2>/dev/null; do
    if (( $(du -sk "$JOB_DIR/results" | cut -f1) > 2097152 )); then
      printf 'output cap exceeded\n' > "$JOB_DIR/env/output-cap-stop.txt"
      kill -TERM "$worker_pid"
      break
    fi
    sleep 30
  done
) &
monitor_pid=$!
wait "$worker_pid" || status=$?
kill "$monitor_pid" 2>/dev/null || true
cd "$JOB_DIR"
find env results -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
exit "$status"
