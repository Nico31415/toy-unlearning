#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/na658/multi-task2}"
PY="${PY:-/home/na658/.conda/envs/mtl_ft/bin/python}"
OMEGA="${OMEGA:-0.5}"
REGIME_IV_LAMBDA_MULT="${REGIME_IV_LAMBDA_MULT:--0.95}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/results/sanity_check}"

mkdir -p "$OUT_DIR" "$REPO_ROOT/logs"
cd "$REPO_ROOT"

total_tasks="$("$PY" experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
  --task-id 0 \
  --omega "$OMEGA" \
  --regime-iv-lambda-mult "$REGIME_IV_LAMBDA_MULT" \
  --output-dir "$OUT_DIR" \
  --list | awk -F= '/total_tasks/ {print $2}')"

echo "[local] Running $total_tasks empirical sanity-check tasks"
for task_id in $(seq 0 "$((total_tasks - 1))"); do
  echo "[local] task $task_id/$((total_tasks - 1))"
  "$PY" experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
    --task-id "$task_id" \
    --omega "$OMEGA" \
    --regime-iv-lambda-mult "$REGIME_IV_LAMBDA_MULT" \
    --output-dir "$OUT_DIR"
done
