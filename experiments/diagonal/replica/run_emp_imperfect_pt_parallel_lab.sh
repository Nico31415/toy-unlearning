#!/usr/bin/env bash
set -euo pipefail

# Runs all tasks from compute_emp_imperfect_pt_worker.py locally in parallel.
# Safe to run in parallel: the worker uses file locking when appending to the shared CSV.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKER="$SCRIPT_DIR/compute_emp_imperfect_pt_worker.py"

PYTHON="${PYTHON:-python3}"
N_JOBS="${N_JOBS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/emp_imperfect_pt}"
LOG_DIR="${LOG_DIR:-logs/emp_imperfect_pt}"
DRY_RUN="0"

usage() {
  cat <<'EOF'
Usage:
  experiments/diagonal/replica/run_emp_imperfect_pt_parallel_lab.sh [options] [-- <extra args forwarded to worker>]

Options:
  -j, --jobs N          Max parallel processes (default: N_JOBS env or auto-detect CPUs)
  -p, --python PATH     Python executable (default: PYTHON env or python3)
  -o, --output-dir DIR  Output dir for shared CSV (default: results/emp_imperfect_pt)
  -l, --log-dir DIR     Directory for per-task logs (default: logs/emp_imperfect_pt)
  -n, --dry-run         Print what would run, then exit
  -h, --help            Show this help

Examples:
  # Run with 16 parallel processes
  ./experiments/diagonal/replica/run_emp_imperfect_pt_parallel_lab.sh -j 16

  # Override worker knobs (forwarded to compute_emp_imperfect_pt_worker.py)
  ./experiments/diagonal/replica/run_emp_imperfect_pt_parallel_lab.sh -j 8 -- --epochs 2000000 --lr 0.25
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)      N_JOBS="${2:-}"; shift 2;;
    -p|--python)    PYTHON="${2:-}"; shift 2;;
    -o|--output-dir) OUTPUT_DIR="${2:-}"; shift 2;;
    -l|--log-dir)   LOG_DIR="${2:-}"; shift 2;;
    -n|--dry-run)   DRY_RUN="1"; shift;;
    -h|--help)      usage; exit 0;;
    --)             shift; EXTRA_ARGS+=("$@"); break;;
    *)              EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "${N_JOBS}" ]]; then
  # macOS: sysctl, Linux: nproc. Fallback to 4.
  N_JOBS="$( (sysctl -n hw.ncpu 2>/dev/null || true) )"
  if [[ -z "${N_JOBS}" ]]; then
    N_JOBS="$( (nproc 2>/dev/null || true) )"
  fi
  N_JOBS="${N_JOBS:-4}"
fi

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ ! -f "$WORKER" ]]; then
  echo "Error: worker not found at: $WORKER" >&2
  exit 1
fi

# Discover total tasks from --info output.
TOTAL_TASKS="$("$PYTHON" "$WORKER" --info | "$PYTHON" -c 'import sys,re; s=sys.stdin.read(); m=re.search(r"Total tasks:\s*(\d+)", s); print(m.group(1) if m else "")' || true)"
if [[ -z "${TOTAL_TASKS}" ]]; then
  echo "Warning: could not detect task count; defaulting to 200." >&2
  TOTAL_TASKS="200"
fi
MAX_ID="$((TOTAL_TASKS - 1))"

echo "Repo root:    $REPO_ROOT"
echo "Python:       $PYTHON"
echo "Worker:       $WORKER"
echo "Tasks:        0..$MAX_ID  (total $TOTAL_TASKS)"
echo "Parallelism:  $N_JOBS"
echo "Output dir:   $OUTPUT_DIR"
echo "Log dir:      $LOG_DIR"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Extra args:   ${EXTRA_ARGS[*]}"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run: exiting without launching tasks."
  exit 0
fi

run_one_task() {
  local id="$1"
  "$PYTHON" "$WORKER" --task-id "$id" --output-dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}" \
    >"$LOG_DIR/task_${id}.out" 2>"$LOG_DIR/task_${id}.err"
}

export PYTHON WORKER OUTPUT_DIR LOG_DIR
export -f run_one_task

echo "Launching tasks..."

if command -v parallel >/dev/null 2>&1; then
  # GNU parallel installed.
  seq 0 "$MAX_ID" | parallel -j "$N_JOBS" --line-buffer bash -lc 'run_one_task "$1"' _ {}
else
  # Portable fallback.
  seq 0 "$MAX_ID" | xargs -n 1 -P "$N_JOBS" -I {} bash -lc 'run_one_task "$1"' _ {}
fi

echo "Done."
echo "Shared CSV should be at: $OUTPUT_DIR/emp_imperfect_pt.csv"
