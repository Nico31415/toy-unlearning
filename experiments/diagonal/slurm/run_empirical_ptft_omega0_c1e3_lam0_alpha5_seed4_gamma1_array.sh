#!/bin/bash
#
# Empirical PT+FT finetune sweep (ptft_empirical_finetune_df.py)
# Fixed params: rho_pt=rho_ft=0.10, c_pt=1e-3, gamma_reinit=1.0
# Sweeps:
#   - omega in {0.0, 1.0}
#   - lambda_pt in {-0.00099, 0.0, +0.00099}  (i.e. ±0.99*c_pt and 0)
#   - alphas=[0.01, 0.0576, 0.105, 0.1524, 0.2]
#   - seeds=[0,1,2,3]
# Total tasks = 2*3*5*4 = 120  => array=0-119
#
# (Optional) overwrite previous results:
#   rm -f results/emp_ptft_sweep_c1e3_gamma1_alpha5_seed4_master.csv \
#         results/emp_ptft_sweep_c1e3_gamma1_alpha5_seed4_master.csv.lock
#
# Submit:
#   sbatch experiments/diagonal/slurm/run_empirical_ptft_omega0_c1e3_lam0_alpha5_seed4_gamma1_array.sh
#

#SBATCH --job-name=emp_ptft_sweep_g1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --array=0-119
#SBATCH --output=logs/slurm/emp_ptft_sweep_g1_%A_%a.out
#SBATCH --error=logs/slurm/emp_ptft_sweep_g1_%A_%a.err
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

ALPHAS_JSON='[0.01,0.0576,0.105,0.1524,0.2]'
SEEDS_JSON='[0,1,2,3]'

RUN_NAME="${RUN_NAME:-emp_ptft_sweep_c1e3_gamma1_alpha5_seed4}"
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
  --omega_json '[0.0,1.0]' \
  --a_pt 1.0 \
  --c_pt_json '[0.001]' \
  --lambda_pt_json '[-0.00099,0.0,0.00099]' \
  --gamma_reinit_json '[1.0]' \
  --lr 0.5 \
  --epochs 5000000 \
  --test_every_n_epochs 5000 \
  --log_every_n_epochs 50000 \
  --no_tuning \
  --threshold 1e-5

