#!/bin/bash
#SBATCH --partition=icelake
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ptft_list_%A_%a.out
#SBATCH --error=logs/ptft_list_%A_%a.err

set -euo pipefail

cd /home/na658/multi-task2
mkdir -p logs results/ptft_replica_qk

PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

COMBO_FILE="experiments/ptft_combo_list.jsonl"

# Optional safety check: count lines and ensure array id is valid
NLINES=$(wc -l < "$COMBO_FILE")
TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -ge "$NLINES" ]; then
  echo "[SLURM] Task id $TID out of range for $COMBO_FILE (0..$((NLINES-1)))" >&2
  exit 2
fi

echo "[SLURM] Running combo index $TID from $COMBO_FILE (NLINES=$NLINES)"

$PY ptft_replica_qk.py \
  --combo_file "$COMBO_FILE" \
  --combo_index "$TID" \
  --outdir "results/ptft_replica_qk" \
  --tag "job${SLURM_JOB_ID}_task${TID}"

echo "[SLURM] Done"