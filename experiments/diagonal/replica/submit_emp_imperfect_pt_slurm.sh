#!/usr/bin/env bash
set -euo pipefail

# Helper to submit the imperfect-PT empirical grid as a SLURM array.
# It auto-detects the task count from compute_emp_imperfect_pt_worker.py --info.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKER="$SCRIPT_DIR/compute_emp_imperfect_pt_worker.py"
SBATCH_SCRIPT="$SCRIPT_DIR/run_emp_imperfect_pt_slurm_array.sh"

PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-results/emp_imperfect_pt}"
WORKER_ARGS=()
EXTRA_SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  experiments/diagonal/replica/submit_emp_imperfect_pt_slurm.sh [options] [-- <worker args>]

Options:
  -p, --python PATH       Python executable (default: PYTHON env or python3)
  -o, --output-dir DIR    Output dir for shared CSV (default: results/emp_imperfect_pt)
  -s, --sbatch "ARGS"     Extra args passed to sbatch (repeatable)
  -n, --dry-run           Print sbatch command, don't submit
  -h, --help              Show this help

Examples:
  # Submit full grid as array, 0..N-1 auto-detected
  ./experiments/diagonal/replica/submit_emp_imperfect_pt_slurm.sh

  # Submit with custom partition/account (passed through to sbatch)
  ./experiments/diagonal/replica/submit_emp_imperfect_pt_slurm.sh -s "--partition=cpu" -s "--account=mygrp"

  # Override worker knobs
  ./experiments/diagonal/replica/submit_emp_imperfect_pt_slurm.sh -- --epochs 2000000 --lr 0.25
EOF
}

DRY_RUN="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--python)     PYTHON="${2:-}"; shift 2;;
    -o|--output-dir) OUTPUT_DIR="${2:-}"; shift 2;;
    -s|--sbatch)     EXTRA_SBATCH_ARGS+=("${2:-}"); shift 2;;
    -n|--dry-run)    DRY_RUN="1"; shift;;
    -h|--help)       usage; exit 0;;
    --)              shift; WORKER_ARGS+=("$@"); break;;
    *)               # treat unknowns as worker args
                     WORKER_ARGS+=("$1"); shift;;
  esac
done

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR" "logs/emp_imperfect_pt_slurm"

TOTAL_TASKS="$("$PYTHON" "$WORKER" --info | "$PYTHON" -c 'import sys,re; s=sys.stdin.read(); m=re.search(r"Total tasks:\\s*(\\d+)", s); print(m.group(1) if m else "")' || true)"
if [[ -z "${TOTAL_TASKS}" ]]; then
  echo "Warning: could not detect task count; defaulting to 200." >&2
  TOTAL_TASKS="200"
fi
MAX_ID="$((TOTAL_TASKS - 1))"

# Pack worker args into WORKER_ARGS env var for the array script.
WORKER_ARGS_ENV=""
if [[ ${#WORKER_ARGS[@]} -gt 0 ]]; then
  # Join with spaces; if you need more complex quoting, set WORKER_ARGS env yourself.
  WORKER_ARGS_ENV="${WORKER_ARGS[*]}"
fi

cmd=(sbatch)
cmd+=(--array="0-${MAX_ID}")
cmd+=(--export="ALL,PYTHON=${PYTHON},OUTPUT_DIR=${OUTPUT_DIR},WORKER_ARGS=${WORKER_ARGS_ENV}")
for a in "${EXTRA_SBATCH_ARGS[@]}"; do
  # Allow user to pass e.g. "--partition=cpu" or "--cpus-per-task=4"
  # shellcheck disable=SC2206
  cmd+=($a)
done
cmd+=("$SBATCH_SCRIPT")

echo "Submitting array: 0..$MAX_ID (total $TOTAL_TASKS)"
echo "Output dir: $OUTPUT_DIR"
if [[ -n "$WORKER_ARGS_ENV" ]]; then
  echo "Worker args: $WORKER_ARGS_ENV"
fi
if [[ ${#EXTRA_SBATCH_ARGS[@]} -gt 0 ]]; then
  echo "Extra sbatch args: ${EXTRA_SBATCH_ARGS[*]}"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'Dry run. Command:\n'
  printf '  %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"

