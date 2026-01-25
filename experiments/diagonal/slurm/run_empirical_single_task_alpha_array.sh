#!/bin/bash
#
# Empirical single-task sweep (ptft_empirical_finetune_df.py)
# One SLURM array task = one (alpha, seed) run, appended to a master CSV under a file lock.
#
# Submit:
#   sbatch experiments/diagonal/slurm/run_empirical_single_task_alpha_array.sh
#
# Optional overrides:
#   sbatch --export=RUN_NAME=mytag,PYTHON_BIN=/path/to/python experiments/diagonal/slurm/run_empirical_single_task_alpha_array.sh
#

#SBATCH --job-name=emp_st_alpha
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --array=0-43
#SBATCH --output=logs/slurm/emp_st_alpha_%A_%a.out
#SBATCH --error=logs/slurm/emp_st_alpha_%A_%a.err
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

# np.linspace(0.01, 0.5, 11)
ALPHAS_JSON='[0.01,0.059,0.108,0.157,0.206,0.255,0.304,0.353,0.402,0.451,0.5]'
SEEDS_JSON='[0,2,3,4]'

RUN_NAME="${RUN_NAME:-emp_single_task_alpha_sweep}"
SAVE_ROOT="results/diagonal/replica_like_empirical/${RUN_NAME}"
MASTER_CSV="results/${RUN_NAME}_master.csv"

cmd=(
  "$PY" experiments/diagonal/replica/ptft_empirical_finetune_df.py
  --setting single_task
  --array_id "${SLURM_ARRAY_TASK_ID}"
  --save_root "${SAVE_ROOT}"
  --master_csv "${MASTER_CSV}"
  --alphas_json "${ALPHAS_JSON}"
  --seeds_json "${SEEDS_JSON}"
  --inp_dim 5000
  --n_test 10000
  --rho_pt_json '[0.10]'
  --a_pt 1.0
  --c_pt_json '[0.001]'
  --lambda_pt_json '[0.0]'
  --lr 0.5
  --epochs 5000000
  --test_every_n_epochs 5000
  --log_every_n_epochs 50000
  --no_tuning
  --threshold 1e-5
  --stop_beta_rate 0.0
  --stop_grad_norm 0.0
  --lr_decay 1.0
  --lr_decay_interval 2000
)

echo "[SLURM] Running: ${cmd[*]}"
exec "${cmd[@]}"

