#!/bin/bash -l
#SBATCH --job-name=sw-cond-action-v1
#SBATCH --account=pi_fl426
#SBATCH --qos=normal
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/roberts/project/pi_fl426/yz2324/scienceworld-conductivity-action-v1/logs/%x-%j.out
#SBATCH --error=/nfs/roberts/project/pi_fl426/yz2324/scienceworld-conductivity-action-v1/logs/%x-%j.err
#SBATCH --export=NONE
set -euo pipefail
ulimit -f 120000
TASK_ROOT=/nfs/roberts/project/pi_fl426/yz2324/scienceworld-conductivity-action-v1
TASK_INPUTS="$TASK_ROOT/inputs"
TASK_RUN="$TASK_ROOT/runs/$SLURM_JOB_ID"
mkdir -p "$TASK_RUN/env" "$TASK_RUN/logs" "$TASK_RUN/source" "$TASK_RUN/tmp"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$TASK_RUN/tmp"
export JAVA_TOOL_OPTIONS='-Xmx4g -XX:ActiveProcessorCount=2'
cd "$TASK_INPUTS"
sha256sum -c SHA256SUMS
module load Java/21.0.2 Python/3.12.3-GCCcore-13.3.0
java -version > "$TASK_RUN/env/java-version.txt" 2>&1
python --version > "$TASK_RUN/env/python-version.txt"
module list 2> "$TASK_RUN/env/modules.txt"
python -m venv --system-site-packages "$TASK_RUN/venv"
TASK_PY="$TASK_RUN/venv/bin/python"
"$TASK_PY" -m pip install --no-index --find-links "$TASK_INPUTS/wheelhouse" --require-hashes -r "$TASK_INPUTS/py4j.lock" > "$TASK_RUN/env/pip-install.log" 2>&1
"$TASK_PY" -m pip freeze > "$TASK_RUN/env/requirements-resolved.txt"
tar -xzf "$TASK_INPUTS/scienceworld-source.tar.gz" -C "$TASK_RUN/source" --strip-components=1
export PYTHONPATH="$TASK_INPUTS/src"
printf 'job_id=%s\naccount=%s\npartition=%s\nnode=%s\ncpus=%s\nmem=%s\nstart=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_ACCOUNT" "$SLURM_JOB_PARTITION" "$(hostname)" "$SLURM_CPUS_PER_TASK" "${SLURM_MEM_PER_NODE:-unknown}" "$(date -Is)" > "$TASK_RUN/env/job_identity.txt"
cd "$TASK_RUN"
finish() {
  TASK_EXIT=$?
  trap - EXIT
  printf '%s\n' "$TASK_EXIT" > env/exit-code.txt
  date -Is > env/finished-at.txt
  find env logs results source/scienceworld source/simulator/src/main/scala -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp
  mv SHA256SUMS.tmp SHA256SUMS
  exit "$TASK_EXIT"
}
trap finish EXIT
"$TASK_PY" "$TASK_INPUTS/scripts/run_scienceworld_conductivity_action_study.py" --protocol "$TASK_INPUTS/scienceworld_conductivity_action_v1.json" --source-root "$TASK_RUN/source" --output-dir "$TASK_RUN/results" > logs/runner.log 2>&1
"$TASK_PY" "$TASK_INPUTS/scripts/audit_scienceworld_conductivity_action_study.py" --protocol "$TASK_INPUTS/scienceworld_conductivity_action_v1.json" --source-root "$TASK_RUN/source" --result "$TASK_RUN/results/result.json" --output "$TASK_RUN/results/independent-audit.json" > logs/auditor.log 2>&1
TASK_BYTES=$(du -sb env logs results source | awk '{sum += $1} END {print sum + 0}')
if (( TASK_BYTES > 120000000 )); then
  echo 'frozen output bundle cap exceeded' >&2
  exit 70
fi
echo 'Full development action panel and independent replay completed; no test/model/Jev calls.'
