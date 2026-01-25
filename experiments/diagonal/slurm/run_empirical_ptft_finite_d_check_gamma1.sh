#!/bin/bash
#
# Finite-dimension check for empirical PTFT vs replica (gamma=1).
#
# Goal:
#   Test whether the systematic bias (empirical slightly below replica) shrinks as d increases.
#   We hold alpha fixed so n_train scales with d (n_train = round(alpha * d)).
#
# Design:
#   - Fix: rho_pt=rho_ft=0.10, omega=1.0, c_pt=1e-3, lambda_pt=0.0, gamma_reinit=1.0
#   - Test: inp_dim in {2000, 5000, 10000}
#   - Seeds: {0, 1, 2}
#   - Alpha: {0.105} (single point, chosen because you’ve been inspecting it)
#   - Stop: require BOTH train_pred_mse < 1e-5 AND beta_update_rate < 1e-6 (near fixed point)
#
# Total tasks = 3 dims * 3 seeds = 9  => array=0-8
#
# Submit:
#   sbatch experiments/diagonal/slurm/run_empirical_ptft_finite_d_check_gamma1.sh
#

#SBATCH --job-name=emp_ptft_finite_d_g1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --array=0-8
#SBATCH --output=logs/slurm/emp_ptft_finite_d_g1_%A_%a.out
#SBATCH --error=logs/slurm/emp_ptft_finite_d_g1_%A_%a.err
#
# If your cluster requires a partition, uncomment and set appropriately:
#SBATCH --partition=icelake

set -euo pipefail

echo "[SLURM] Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} starting on $(hostname)"

cd /home/na658/multi-task2
mkdir -p logs/slurm

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY="${PYTHON_BIN:-/home/na658/.conda/envs/mtl_ft/bin/python}"
if [ ! -x "$PY" ]; then
  PY="${PYTHON_BIN:-python}"
fi
echo "[SLURM] Using Python: ${PY}"

# Task grid (size 9)
DIMS=(2000 5000 10000)
SEEDS=(0 1 2)

nd=${#DIMS[@]}
ns=${#SEEDS[@]}
TOTAL=$((nd * ns))

tid=${SLURM_ARRAY_TASK_ID}
if [ "$tid" -lt 0 ] || [ "$tid" -ge "$TOTAL" ]; then
  echo "[SLURM] array_id out of range: ${tid} (expected 0..$((TOTAL-1)))" >&2
  exit 2
fi

dim_idx=$((tid / ns))
seed_idx=$((tid % ns))

INP_DIM="${DIMS[$dim_idx]}"
SEED="${SEEDS[$seed_idx]}"

ALPHAS_JSON='[0.105]'
SEEDS_JSON="[$SEED]"

RUN_NAME="${RUN_NAME:-emp_ptft_finite_d_check_gamma1_alpha0105}"
SAVE_ROOT="results/diagonal/replica_like_empirical/${RUN_NAME}"
MASTER_CSV="results/${RUN_NAME}_master.csv"

exec "$PY" experiments/diagonal/replica/ptft_empirical_finetune_df.py \
  --setting ptft \
  --array_id 0 \
  --save_root "${SAVE_ROOT}" \
  --master_csv "${MASTER_CSV}" \
  --alphas_json "${ALPHAS_JSON}" \
  --seeds_json "${SEEDS_JSON}" \
  --inp_dim "${INP_DIM}" \
  --n_test 10000 \
  --rho_pt_json '[0.10]' \
  --rho_ft_json '[0.10]' \
  --omega_json '[1.0]' \
  --a_pt 1.0 \
  --c_pt_json '[0.001]' \
  --lambda_pt_json '[0.0]' \
  --gamma_reinit_json '[1.0]' \
  --lr 0.5 \
  --epochs 5000000 \
  --test_every_n_epochs 5000 \
  --log_every_n_epochs 50000 \
  --no_tuning \
  --threshold 1e-5 \
  --stop_beta_rate 1e-6

