#!/bin/bash
#SBATCH --job-name=unlearn_all
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-7985%100
set -euo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: submit this script with sbatch, not bash:" >&2
  echo "  sbatch $0" >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PY="${PY:-/home/na658/.conda/envs/mtl_ft/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi

cd "$REPO_ROOT"
mkdir -p results logs

task_id="$SLURM_ARRAY_TASK_ID"

run() {
  echo "[unlearn_all] global_task_id=$task_id"
  echo "[unlearn_all] command: $PY $*"
  "$PY" "$@"
}

# Segment sizes:
#   forgetting                       330    global 0..329
#   sanity omega/lambda variants   6*495   global 330..3299
#   replica sanity                    30    global 3300..3329
#   correlated q sweep               495    global 3330..3824
#   scratch baseline                 165    global 3825..3989
#   readout recovery                 108    global 3990..4097
#   GD recovery                     3888    global 4098..7985

if (( task_id < 330 )); then
  run experiments/diagonal/replica/compute_emp_forgetting_worker.py \
    --task-id "$task_id" \
    --output-dir results/forgetting
elif (( task_id < 825 )); then
  local_id=$((task_id - 330))
  run experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
    --task-id "$local_id" \
    --omega 0.0 \
    --regime-iv-lambda-mult -0.95 \
    --output-dir results/sanity_check_omega00
elif (( task_id < 1320 )); then
  local_id=$((task_id - 825))
  run experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
    --task-id "$local_id" \
    --omega 0.1 \
    --regime-iv-lambda-mult -0.95 \
    --output-dir results/sanity_check_omega01
elif (( task_id < 1815 )); then
  local_id=$((task_id - 1320))
  run experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
    --task-id "$local_id" \
    --omega 0.5 \
    --regime-iv-lambda-mult -0.95 \
    --output-dir results/sanity_check
elif (( task_id < 2310 )); then
  local_id=$((task_id - 1815))
  run experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
    --task-id "$local_id" \
    --omega 0.9 \
    --regime-iv-lambda-mult -0.95 \
    --output-dir results/sanity_check_omega09
elif (( task_id < 2805 )); then
  local_id=$((task_id - 2310))
  run experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
    --task-id "$local_id" \
    --omega 1.0 \
    --regime-iv-lambda-mult -0.95 \
    --output-dir results/sanity_check_omega1
elif (( task_id < 3300 )); then
  local_id=$((task_id - 2805))
  run experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
    --task-id "$local_id" \
    --omega 0.5 \
    --regime-iv-lambda-mult -0.99 \
    --output-dir results/sanity_check_lam099
elif (( task_id < 3330 )); then
  local_id=$((task_id - 3300))
  run experiments/diagonal/replica/compute_replica_sanity_check_worker.py \
    --task-id "$local_id" \
    --omega 0.5 \
    --output-dir results/replica_sanity_check_omega05 \
    --n-alpha-chunks 5 \
    --mc 80000
elif (( task_id < 3825 )); then
  local_id=$((task_id - 3330))
  run experiments/diagonal/replica/compute_emp_correlated_overlap_worker.py \
    --task-id "$local_id" \
    --omega 0.5 \
    --output-dir results/sanity_check_correlated_overlap_q_sweep
elif (( task_id < 3990 )); then
  local_id=$((task_id - 3825))
  run experiments/diagonal/replica/compute_emp_correlated_overlap_scratch_worker.py \
    --task-id "$local_id" \
    --omega 0.5 \
    --output-dir results/sanity_check_correlated_overlap_scratch
elif (( task_id < 4098 )); then
  local_id=$((task_id - 3990))
  run experiments/diagonal/replica/compute_readout_recovery_worker.py \
    --task-id "$local_id" \
    --output-dir results/readout_recovery_correlated_overlap
elif (( task_id < 7986 )); then
  local_id=$((task_id - 4098))
  run experiments/diagonal/replica/compute_gd_recovery_worker.py \
    --task-id "$local_id" \
    --output-dir results/gd_recovery_correlated_overlap \
    --variants all
else
  echo "ERROR: task_id=$task_id outside configured range 0..7985" >&2
  exit 1
fi
