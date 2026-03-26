#!/bin/bash
set -euo pipefail

# Submit imperfect-pretraining replica curves over the oracle PT+FT config grid,
# restricted to omega ∈ {0,1}, for alpha_pt ∈ {0.01, 0.1, 0.5}.
#
# This script computes the required Slurm array size via the worker's --list mode,
# then submits the full array with a single sbatch call.
#
# Usage:
#   bash experiments/diagonal/replica/submit_replica_imperfect_oraclegrid_omega01.sh
#
# Optional environment overrides:
#   N_ALPHA_CHUNKS=5 OUT_DIR=results/... TIME=06:00:00 MEM=8G PARTITION=icelake
#   DRY_RUN=1   (print sbatch command, do not submit)

PY="${PY:-/home/na658/.conda/envs/mtl_ft/bin/python}"
REPO_ROOT="${REPO_ROOT:-/home/na658/multi-task2}"

N_ALPHA_CHUNKS="${N_ALPHA_CHUNKS:-5}"
OUT_DIR="${OUT_DIR:-results/replica_imperfect_pt_oraclegrid}"

TIME="${TIME:-06:00:00}"
MEM="${MEM:-8G}"
PARTITION="${PARTITION:-icelake}"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_ROOT"

# Ensure replica modules are importable (same convention as other submit scripts)
export PYTHONPATH="$REPO_ROOT/experiments/diagonal/replica${PYTHONPATH:+:$PYTHONPATH}"

LIST_OUT="$($PY experiments/diagonal/replica/compute_replica_curves_worker_imperfect_oraclegrid.py \
  --list \
  --n-alpha-chunks "$N_ALPHA_CHUNKS")"

TOTAL_TASKS="$(printf "%s\n" "$LIST_OUT" | sed -n 's/^total_tasks=//p' | head -n 1)"
if [[ -z "${TOTAL_TASKS}" ]]; then
  echo "ERROR: failed to parse total_tasks from --list output"
  echo "$LIST_OUT"
  exit 1
fi

MAX_TASK_ID="$((TOTAL_TASKS - 1))"
echo "Submitting Slurm array 0-${MAX_TASK_ID}"
echo "  n_alpha_chunks=${N_ALPHA_CHUNKS}"
echo "  output_dir=${OUT_DIR}"

SBATCH_CMD=(
  sbatch
  --mem="${MEM}"
  --cpus-per-task=1
  --time="${TIME}"
  --partition="${PARTITION}"
  --job-name=rep_imp_orc_om01
  --output=/dev/null
  --error=/dev/null
  --array="0-${MAX_TASK_ID}"
  --wrap="$PY experiments/diagonal/replica/compute_replica_curves_worker_imperfect_oraclegrid.py \
    --task-id \$SLURM_ARRAY_TASK_ID \
    --n-alpha-chunks ${N_ALPHA_CHUNKS} \
    --output-dir ${OUT_DIR}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf "%q " "${SBATCH_CMD[@]}"
  printf "\n"
  exit 0
fi

("${SBATCH_CMD[@]}")

