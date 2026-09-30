#!/usr/bin/env bash
#SBATCH --job-name=recoma-final-qual-reaudit-v3
#SBATCH --account=pi_fl426
#SBATCH --qos=normal
#SBATCH --partition=devel,day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --deadline=2026-09-30T22:26:58
#SBATCH --output=/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3/logs/%x-%j.out
#SBATCH --error=/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3/logs/%x-%j.err
#SBATCH --export=NONE
set -euo pipefail
TASK_PHASE_ROOT=/nfs/roberts/project/pi_fl426/yz2324/recoma-final-six-hour-v3
TASK_AUDIT_INPUTS="$TASK_PHASE_ROOT/audit-inputs-v3f-correction2"
TASK_QUAL_INPUTS="$TASK_PHASE_ROOT/qualification-inputs-v3f"
TASK_QUAL_DIR="$TASK_PHASE_ROOT/runs/qualification-27978394"
TASK_AUDIT_DIR="$TASK_PHASE_ROOT/runs/qualification-audit-$SLURM_JOB_ID"
mkdir "$TASK_AUDIT_DIR"
mkdir "$TASK_AUDIT_DIR/env" "$TASK_AUDIT_DIR/results"
finalize() {
  status=$?
  trap - EXIT
  printf '%s\n' "$status" > "$TASK_AUDIT_DIR/env/batch-exit-code.txt"
  cd "$TASK_AUDIT_DIR"
  find env results -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  exit "$status"
}
trap finalize EXIT
export SLURM_EXPORT_ENV=ALL PYTHONDONTWRITEBYTECODE=1
unset TYPESAFE_API_KEY OPENAI_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN 2>/dev/null || true
module load PyTorch/2.9.1-foss-2024a-CUDA-12.8.0 Transformers/4.55.2-gfbf-2024a
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
module list 2> "$TASK_AUDIT_DIR/env/modules.txt"
cd "$TASK_AUDIT_INPUTS"
sha256sum -c SHA256SUMS > "$TASK_AUDIT_DIR/env/audit-input-checksums.txt"
cd "$TASK_QUAL_DIR"
sha256sum -c SHA256SUMS > "$TASK_AUDIT_DIR/env/qualified-output-checksums.txt"
TASK_SNAPSHOT=/home/yz2324/project_pi_fl426/yz2324/JEV-CoT-Reward/cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
TASK_AUDIT_ARGS=(--manifest "$TASK_QUAL_INPUTS/qualification_manifest.json" --resources "$TASK_AUDIT_INPUTS/resources-27978394.json" --tokenizer-path "$TASK_SNAPSHOT" --source-root "$TASK_QUAL_INPUTS/source/discoveryworld" --output-files "$TASK_QUAL_DIR/results/worker-0/all_data.jsonl" --call-files "$TASK_QUAL_DIR/results/worker-0-calls.jsonl" --task-record-files "$TASK_QUAL_DIR/results/worker-0/task_records.jsonl" --task-event-files "$TASK_QUAL_DIR/results/worker-0/task_events.jsonl" --runtime-report "$TASK_QUAL_DIR/results/worker-0/runtime_report.json" --stress-files "$TASK_QUAL_DIR/results/worker-0/context_stress.json")
timeout --signal=TERM --kill-after=15s 900s "$TASK_QUAL_DIR/venv/bin/python" "$TASK_AUDIT_INPUTS/audit_recoma_discoveryworld_v3.py" "${TASK_AUDIT_ARGS[@]}" --output "$TASK_AUDIT_DIR/results/corrected-independent-audit.json" > "$TASK_AUDIT_DIR/env/corrected-audit.stdout.log" 2> "$TASK_AUDIT_DIR/env/corrected-audit.stderr.log"
printf 'CPU-only independent audit and exactweightcontentcheck; modelgenerations=0,hostedcalls=0,jevCalls=0\n' > "$TASK_AUDIT_DIR/env/no-generation-proof.txt"
