#!/usr/bin/env bash
# TEMPLATE ONLY: markers are rendered from independently audited GPU timing.
#SBATCH --job-name=recoma-final-panel-v3
#SBATCH --account=pi_fl426
#SBATCH --qos=normal
#SBATCH --partition=__PARTITIONS__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=__CPU_COUNT__
#SBATCH --mem=__MEMORY_GIB__G
#SBATCH --gres=__GPU_GRES__
__NODE_PLACEMENT__
#SBATCH --time=__WALL_TIME__
#SBATCH --deadline=2026-09-30T22:26:58
#SBATCH --output=__REMOTE_ROOT__/logs/%x-%j.out
#SBATCH --error=__REMOTE_ROOT__/logs/%x-%j.err
#SBATCH --export=NONE
set -euo pipefail
ROOT=__REMOTE_ROOT__
INPUTS="$ROOT/inputs"
JOB_DIR="$ROOT/runs/panel-$SLURM_JOB_ID"
DEADLINE_EPOCH=1790821618
GPU_COUNT=__GPU_COUNT__
mkdir "$JOB_DIR"
mkdir -p "$JOB_DIR/env" "$JOB_DIR/results" "$JOB_DIR/home"
worker_pids=()
monitor_pid=''
model_verification_started=0
finalize() {
  status=$?
  trap - EXIT INT TERM
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  for pid in "${worker_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${worker_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  if (( model_verification_started )); then
    if /usr/bin/python3 "$INPUTS/verify_model_snapshot_v3.py" --snapshot "$MODEL_PATH" \
      --revision cdbee75f17c01a7cc42f958dc650907174af0554 --checksums "$INPUTS/model_snapshot_SHA256SUMS" \
      --output "$JOB_DIR/env/model-verification-after.json" > "$JOB_DIR/env/model-verification-after.stdout.log" 2>&1; then :; else status=80; fi
  fi
  if [[ -f "$INPUTS/SHA256SUMS" ]]; then
    cd "$INPUTS"
    if sha256sum -c SHA256SUMS > "$JOB_DIR/env/input-checksums-after.txt" 2>&1; then :; else status=79; fi
  fi
  printf '%s\n' "$status" > "$JOB_DIR/env/batch-exit-code.txt"
  cd "$JOB_DIR"
  find env results -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS || true
  exit "$status"
}
trap finalize EXIT
trap 'exit 143' TERM INT
remaining=$((DEADLINE_EPOCH - $(date +%s)))
if (( remaining <= 60 )); then exit 75; fi
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
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$INPUTS/source/recoma:$INPUTS/source/discoveryworld${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export MAX_ENV_CALLS=30 SEED=0 OUTPUT_DIR="$JOB_DIR/results"
export XDG_CACHE_HOME="$JOB_DIR/home/cache" MPLCONFIGDIR="$JOB_DIR/home/matplotlib"
export TRITON_CACHE_DIR="$JOB_DIR/home/triton" TORCH_HOME="$JOB_DIR/home/torch"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
model_verification_started=1
/usr/bin/python3 "$INPUTS/verify_model_snapshot_v3.py" --snapshot "$MODEL_PATH" \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --checksums "$INPUTS/model_snapshot_SHA256SUMS" \
  --output "$JOB_DIR/env/model-verification-before.json" > "$JOB_DIR/env/model-verification-before.stdout.log" 2>&1
ulimit -f 131072
ulimit -c 0
# Parse only frozen assignment/resource/timing data, never endpoint outcomes.
"$JOB_DIR/venv/bin/python" - "$INPUTS/panel_manifest.json" "$GPU_COUNT" <<'PY'
import json, os, sys, time
m=json.load(open(sys.argv[1])); n=int(sys.argv[2]); r=m['resources']
assert r['account']=='pi_fl426' and r['qos']=='normal' and r['gpu_count']==n==len(m['workers'])
assert r['cpus']==4*n and r['memory_gib']==32*n and m['deadline_epoch']==1790821618
assert m['resource_planning']['already_allocated_phase_gpu_hours']+n*r['wall_minutes']/60 <= 16+1e-9
assert int(os.environ['SLURM_CPUS_PER_TASK'])==4*n
need=m['resource_planning']['selected_plan']['predicted_slowest_worker_seconds']+60
assert m['deadline_epoch']-time.time()>=need, 'queue/setup delay leaves insufficient complete-census forecast'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==n
qualified=m['qualified_gpu']
for i in range(n):
    assert torch.cuda.get_device_name(i)==qualified['name'], 'unqualified GPU card: no cross-family timing transfer'
    assert torch.cuda.get_device_properties(i).total_memory==qualified['total_memory_bytes'], 'unqualified physical GPU capacity'
if 'gpu_devel' in r['partitions']:
    assert n<=2, 'gpu_devel per-user allocation ceiling'
PY
IFS=',' read -r -a gpu_devices <<< "${CUDA_VISIBLE_DEVICES:?allocated CUDA device IDs required}"
if (( ${#gpu_devices[@]} != GPU_COUNT )); then exit 76; fi
for (( index=0; index<GPU_COUNT; index++ )); do
  IFS=$'\t' read -r seeds prefixes < <("$JOB_DIR/venv/bin/python" - "$INPUTS/panel_manifest.json" "$index" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); w=m['workers'][int(sys.argv[2])]
print(json.dumps(w['seeds'],separators=(',',':'))+'\t'+json.dumps(w['scenario_prefixes'],separators=(',',':')))
PY
  )
  remaining=$((DEADLINE_EPOCH - $(date +%s) - 60))
  if (( remaining <= 0 )); then exit 75; fi
  CUDA_VISIBLE_DEVICES="${gpu_devices[$index]}" SEEDS="$seeds" SCENARIO_PREFIXES="$prefixes" \
    RECOMA_USAGE_LOG="$JOB_DIR/results/worker-$index-calls.jsonl" \
    timeout --signal=TERM --kill-after=15s "${remaining}s" "$JOB_DIR/venv/bin/python" \
    "$INPUTS/run_bounded_recoma_worker_v3.py" --manifest "$INPUTS/panel_manifest.json" \
    --config "$INPUTS/recoma_discoveryworld_react_full_v1.jsonnet" --worker-index "$index" \
    --output-dir "$JOB_DIR/results/worker-$index" > "$JOB_DIR/env/worker-$index.stdout.log" \
    2> "$JOB_DIR/env/worker-$index.stderr.log" &
  worker_pids+=("$!")
done
(
  while true; do
    primary_kib=$(du -sk "$JOB_DIR/results" | cut -f1)
    total_kib=$(du -sk "$JOB_DIR" | cut -f1)
    if (( primary_kib > 2097152 || total_kib > 4194304 || $(date +%s) >= DEADLINE_EPOCH-60 )); then
      printf 'hard output/deadline stop: primaryKiB=%s totalKiB=%s\n' "$primary_kib" "$total_kib" > "$JOB_DIR/env/output-or-deadline-stop.txt"
      for pid in "${worker_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
      break
    fi
    sleep 15
  done
) &
monitor_pid=$!
status=0
for index in "${!worker_pids[@]}"; do
  pid="${worker_pids[$index]}"
  if wait "$pid"; then :; else
    status=$?
    for other_pid in "${worker_pids[@]}"; do kill -TERM "$other_pid" 2>/dev/null || true; done
    break
  fi
  unset 'worker_pids[index]'
done
exit "$status"
