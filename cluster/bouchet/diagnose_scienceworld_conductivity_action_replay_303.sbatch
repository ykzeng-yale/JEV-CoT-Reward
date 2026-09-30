#!/bin/bash -l
#SBATCH --job-name=sw-cond-replay-diag
#SBATCH --account=pi_fl426
#SBATCH --qos=normal
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/roberts/project/pi_fl426/yz2324/scienceworld-conductivity-action-v1/logs/%x-%j.out
#SBATCH --error=/nfs/roberts/project/pi_fl426/yz2324/scienceworld-conductivity-action-v1/logs/%x-%j.err
#SBATCH --export=NONE
set -euo pipefail
ulimit -f 10000
TASK_ROOT=/nfs/roberts/project/pi_fl426/yz2324/scienceworld-conductivity-action-v1
RUN="$TASK_ROOT/runs/27950404"
DIAG="$TASK_ROOT/audits/replay-303-measurement-${SLURM_JOB_ID}"
DIAG_INPUTS="$TASK_ROOT/diagnosis-retry1-inputs"
mkdir -p "$DIAG"
export PYTHONDONTWRITEBYTECODE=1 JAVA_TOOL_OPTIONS='-Xmx4g -XX:ActiveProcessorCount=2' PYTHONPATH="$TASK_ROOT/inputs/src"
module load Java/21.0.2 Python/3.12.3-GCCcore-13.3.0
"$RUN/venv/bin/python" "$DIAG_INPUTS/diagnose_scienceworld_conductivity_action_replay.py" --protocol "$TASK_ROOT/inputs/scienceworld_conductivity_action_v1.json" --result "$RUN/results/result.json" --source-root "$RUN/source" --output "$DIAG/diagnosis.json" --variation 303 --policy measurement > "$DIAG/summary.json"
cd "$DIAG"
sha256sum diagnosis.json summary.json > SHA256SUMS
