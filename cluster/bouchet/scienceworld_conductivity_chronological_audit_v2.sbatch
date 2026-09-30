#!/bin/bash -l
#SBATCH --job-name=sw-cond-chronaudit
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
INPUTS="$TASK_ROOT/chronological-audit-v2-inputs"
OUT="$TASK_ROOT/audits/chronological-v2-${SLURM_JOB_ID}"
mkdir "$OUT"
cd "$INPUTS"
sha256sum -c SHA256SUMS > "$OUT/input-verification.txt"
printf '%s  %s\n' d9c9a5821c68c0852ccae9d36b2b80a8167bd1066e5fe50162d840e09c3a3ebe "$RUN/results/result.json" | sha256sum -c - > "$OUT/result-verification.txt"
export PYTHONDONTWRITEBYTECODE=1 JAVA_TOOL_OPTIONS='-Xmx4g -XX:ActiveProcessorCount=2' PYTHONPATH="$INPUTS/src"
module load Java/21.0.2 Python/3.12.3-GCCcore-13.3.0
module list 2> "$OUT/modules.txt"
java -version > "$OUT/java-version.txt" 2>&1
"$RUN/venv/bin/python" --version > "$OUT/python-version.txt"
finish() {
  TASK_EXIT=$?
  trap - EXIT
  printf '%s\n' "$TASK_EXIT" > "$OUT/exit-code.txt"
  date -Is > "$OUT/finished-at.txt"
  cd "$OUT"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  exit "$TASK_EXIT"
}
trap finish EXIT
"$RUN/venv/bin/python" "$INPUTS/audit_scienceworld_conductivity_action_study.py" --protocol "$TASK_ROOT/inputs/scienceworld_conductivity_action_v1.json" --result "$RUN/results/result.json" --source-root "$RUN/source" --output "$OUT/independent-audit.json" > "$OUT/summary.json" 2> "$OUT/auditor.log"
