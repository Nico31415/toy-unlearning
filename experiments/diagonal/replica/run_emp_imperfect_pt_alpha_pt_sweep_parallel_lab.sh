#!/bin/bash

# Runs all tasks from compute_emp_imperfect_pt_alpha_pt_sweep_worker.py locally in parallel
# on the lab machine (no SLURM). Uses the same hardcoded PY/REPO_ROOT style
# as the other scripts in experiments/diagonal/replica/.

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
REPO_ROOT="/home/na658/multi-task2"

JOBS="${JOBS:-8}"

mkdir -p "$REPO_ROOT/logs/emp_imperfect_pt_alpha_pt_sweep_parallel"
mkdir -p "$REPO_ROOT/results/emp_imperfect_pt_alpha_pt_sweep"

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/experiments/diagonal/replica:$PYTHONPATH"

echo "============================================"
echo "Empirical imperfect-PT alpha_pt sweep (local parallel)"
echo "Jobs: $JOBS"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "============================================"

# Uses file locking inside the worker when appending to the shared CSV.
seq 0 2771 | parallel -j "$JOBS" --line-buffer --results "$REPO_ROOT/logs/emp_imperfect_pt_alpha_pt_sweep_parallel" \
  "$PY" experiments/diagonal/replica/compute_emp_imperfect_pt_alpha_pt_sweep_worker.py \
    --task-id {} \
    --output-dir "results/emp_imperfect_pt_alpha_pt_sweep"

echo "============================================"
echo "Finished at: $(date)"
echo "============================================"
