#!/bin/bash
#
# Empirical PT+FT finetune sweep (ptft_empirical_finetune_df.py)
# Matches replica_ptft.ipynb: omega=0, c_pt=1e-3, lambda_pt=0, gamma_reinit=0.
# One SLURM array task = one (alpha, seed, config) run, appended to a master CSV under a file lock.
#
# Submit:
#   sbatch experiments/diagonal/slurm/run_empirical_ptft_omega0_c1e3_lam0_array.sh
#

#SBATCH --job-name=emp_ptft_o0
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --array=0-10
#SBATCH --output=logs/slurm/emp_ptft_o0_%A_%a.out
#SBATCH --error=logs/slurm/emp_ptft_o0_%A_%a.err
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
SEEDS_JSON='[0]'

RUN_NAME="${RUN_NAME:-emp_ptft_omega0_c1e3_lam0}"
SAVE_ROOT="results/diagonal/replica_like_empirical/${RUN_NAME}"
MASTER_CSV="results/${RUN_NAME}_master.csv"

exec "$PY" experiments/diagonal/replica/ptft_empirical_finetune_df.py \
  --setting ptft \
  --array_id "${SLURM_ARRAY_TASK_ID}" \
  --save_root "${SAVE_ROOT}" \
  --master_csv "${MASTER_CSV}" \
  --alphas_json "${ALPHAS_JSON}" \
  --seeds_json "${SEEDS_JSON}" \
  --inp_dim 5000 \
  --n_test 10000 \
  --rho_pt_json '[0.10]' \
  --rho_ft_json '[0.10]' \
  --omega_json '[0.0]' \
  --a_pt 1.0 \
  --c_pt_json '[0.001]' \
  --lambda_pt_json '[0.0]' \
  --gamma_reinit_json '[0.0]' \
  --lr 0.5 \
  --epochs 5000000 \
  --test_every_n_epochs 5000 \
  --log_every_n_epochs 50000 \
  --no_tuning \
  --threshold 1e-5

