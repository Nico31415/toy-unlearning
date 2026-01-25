#!/bin/bash
#
# Empirical single-task sweep over (alpha, seed, c_pt, lambda_pt).
# One SLURM array task = one run, appended to a master CSV under a file lock.
#
# Setups (4):
#   0: c_pt = 1.0     , lambda_pt = 0
#   1: c_pt = 1e-9    , lambda_pt = 0
#   2: c_pt = 1e-3    , lambda_pt = +0.99*c = +0.00099
#   3: c_pt = 1e-3    , lambda_pt = -0.99*c = -0.00099
#
# Total tasks = 5 alphas * 4 seeds * 4 setups = 80  => array=0-79
#
# Submit:
#   sbatch experiments/diagonal/slurm/run_empirical_single_task_big_sweep.sh
#
# Optional overrides:
#   sbatch --export=RUN_NAME=mytag,PYTHON_BIN=/path/to/python experiments/diagonal/slurm/run_empirical_single_task_big_sweep.sh
#

#SBATCH --job-name=emp_st_big
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --array=0-79
#SBATCH --output=logs/slurm/emp_st_big_%A_%a.out
#SBATCH --error=logs/slurm/emp_st_big_%A_%a.err
#
# If your cluster requires a partition, uncomment and set appropriately:
#SBATCH --partition=icelake

set -euo pipefail

echo "[SLURM] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on $(hostname)"

# Run from repo root (important for relative paths used by the script)
cd /home/na658/multi-task2
mkdir -p logs/slurm

# Avoid oversubscription when launching many tasks in parallel.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Pick Python. Prefer $PYTHON_BIN if provided; otherwise use the repo's common conda env if present.
PY="${PYTHON_BIN:-/home/na658/.conda/envs/mtl_ft/bin/python}"
if [ ! -x "$PY" ]; then
  PY="${PYTHON_BIN:-python}"
fi
echo "[SLURM] Using Python: ${PY}"

# 5 values from 0.01 to 0.4 (inclusive)
ALPHAS=(0.01 0.1075 0.205 0.3025 0.4)
SEEDS=(0 2 3 4)

# Map array_id -> (alpha_idx, seed_idx, setup_idx)
tid="${SLURM_ARRAY_TASK_ID}"
setup_idx=$(( tid % 4 ))
tmp=$(( tid / 4 ))
seed_idx=$(( tmp % 4 ))
alpha_idx=$(( tmp / 4 ))

alpha="${ALPHAS[$alpha_idx]}"
seed="${SEEDS[$seed_idx]}"

case "$setup_idx" in
  0) c_pt="1.0";    lam_pt="0.0" ;;
  1) c_pt="1e-9";   lam_pt="0.0" ;;
  2) c_pt="0.001";  lam_pt="0.00099" ;;
  3) c_pt="0.001";  lam_pt="-0.00099" ;;
  *) echo "bad setup_idx=$setup_idx"; exit 2 ;;
esac

RUN_NAME="${RUN_NAME:-emp_single_task_big_sweep}"
SAVE_ROOT="results/diagonal/replica_like_empirical/${RUN_NAME}"
MASTER_CSV="results/${RUN_NAME}_master.csv"

echo "[task] alpha=${alpha} seed=${seed} c_pt=${c_pt} lambda_pt=${lam_pt}"

exec "$PY" experiments/diagonal/replica/ptft_empirical_finetune_df.py \
  --setting single_task \
  --array_id 0 \
  --save_root "${SAVE_ROOT}" \
  --master_csv "${MASTER_CSV}" \
  --alphas_json "[${alpha}]" \
  --seeds_json "[${seed}]" \
  --inp_dim 5000 \
  --n_test 10000 \
  --rho_pt_json '[0.10]' \
  --a_pt 1.0 \
  --c_pt_json "[${c_pt}]" \
  --lambda_pt_json "[${lam_pt}]" \
  --lr 0.5 \
  --epochs 5000000 \
  --test_every_n_epochs 5000 \
  --log_every_n_epochs 50000 \
  --no_tuning \
  --threshold 1e-5

