#!/usr/bin/env bash
#SBATCH --job-name=sw-cond-index
#SBATCH --account=pi_fl426
#SBATCH --qos=normal
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/roberts/project/pi_fl426/yz2324/scienceworld-conductivity-visible-index-v1/logs/%x-%j.out
#SBATCH --error=/nfs/roberts/project/pi_fl426/yz2324/scienceworld-conductivity-visible-index-v1/logs/%x-%j.err
#SBATCH --export=ALL
set -euo pipefail

ROOT=/nfs/roberts/project/pi_fl426/yz2324/scienceworld-conductivity-visible-index-v1
INPUTS="$ROOT/inputs"
JOB_DIR="$ROOT/runs/${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR/env" "$JOB_DIR/results" "$JOB_DIR/logs" "$JOB_DIR/home" "$JOB_DIR/source"
export HOME="$JOB_DIR/home" PYTHONDONTWRITEBYTECODE=1
cd "$INPUTS"
sha256sum -c SHA256SUMS
module load Java/21.0.2 Python/3.12.3-GCCcore-13.3.0
java -version > "$JOB_DIR/env/java-version.txt" 2>&1
python --version > "$JOB_DIR/env/python-version.txt"
module list 2> "$JOB_DIR/env/modules.txt"
python -m venv --system-site-packages "$JOB_DIR/venv"
"$JOB_DIR/venv/bin/python" -m pip install --no-index --find-links "$INPUTS/wheelhouse" --require-hashes -r "$INPUTS/py4j.lock" > "$JOB_DIR/env/pip-install.log" 2>&1
"$JOB_DIR/venv/bin/python" -m pip freeze > "$JOB_DIR/env/requirements-resolved.txt"
tar -xzf "$INPUTS/scienceworld-source.tar.gz" -C "$JOB_DIR/source" --strip-components=1
printf 'job_id=%s\naccount=%s\npartition=%s\nnode=%s\ncpus=%s\nmem=%s\nstart=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_ACCOUNT" "$SLURM_JOB_PARTITION" "$(hostname)" "$SLURM_CPUS_PER_TASK" "${SLURM_MEM_PER_NODE:-unknown}" "$(date -Is)" > "$JOB_DIR/env/job_identity.txt"
"$JOB_DIR/venv/bin/python" "$INPUTS/run_scienceworld_conductivity_visible_index.py" --protocol "$INPUTS/scienceworld_conductivity_visible_index_v1.json" --source-root "$JOB_DIR/source" --output "$JOB_DIR/results/result.json" > "$JOB_DIR/logs/summary.json"
"$JOB_DIR/venv/bin/python" "$INPUTS/audit_scienceworld_conductivity_visible_index_result.py" --protocol "$INPUTS/scienceworld_conductivity_visible_index_v1.json" --result "$JOB_DIR/results/result.json" --audit-output "$JOB_DIR/results/independent-audit.json" > "$JOB_DIR/logs/independent-audit.json"
date -Is > "$JOB_DIR/env/finished-at.txt"
cd "$JOB_DIR"
find env results logs source/scienceworld source/simulator/src/main/scala -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp
mv SHA256SUMS.tmp SHA256SUMS
printf 'Train/dev-only conductivity visible-index audit complete; no test, outcome, model, Jev, network, or gold-path calls.\n'
