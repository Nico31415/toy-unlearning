#!/bin/bash
#
# Empirical PT+FT oracle training sweep (diagonal_ptft_oracle.py)
# Matches the UNIQUE set of PTFT configs used for the replica curves (experiments 2–8),
# crossed with:
#   - 10 alpha values in [0.01, 0.5]  (implemented via n_train = round(alpha * inp_dim))
#   - 3 seeds per (config, alpha)
#
# One SLURM task = one (config, alpha, seed) run.
#
# Submit:
#   sbatch experiments/diagonal/slurm/run_ptft_oracle_empirical_alpha10_lr0p1.sh
#
# Optional:
#   sbatch --export=RUN_NAME=my_run_tag experiments/diagonal/slurm/run_ptft_oracle_empirical_alpha10_lr0p1.sh
#

#SBATCH --partition=icelake
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-599
#SBATCH --output=logs/ptft_oracle_empirical_%A_%a.out
#SBATCH --error=logs/ptft_oracle_empirical_%A_%a.err

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID starting on $(hostname)"

# Run from repo root
cd /home/na658/multi-task2
mkdir -p logs

# Use absolute Python from conda env (match repo conventions)
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi
echo "[SLURM] Using Python: $PY"

# Avoid oversubscription when launching many tasks in parallel
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# -----------------------
# Shared run hyperparams
# -----------------------
INP_DIM=1000
N_TEST=10000

# Requested: lr=0.1, double max epochs vs previous 5e6
LR=0.1
EPOCHS=10000000

# "Everything else same" as previous oracle sweeps
THRESH=1e-12
TEST_EVERY=200
NO_TUNING=1

# PTFT oracle parameters fixed across all configs
A_PT=1.0
C_PT=0.001

# -----------------------
# Alpha grid (10 values on [0.01, 0.5])
# We implement alpha via n_train = round(alpha * inp_dim) with inp_dim=1000.
# Effective alpha used in folder names is alpha_eff = n_train / 1000.
# -----------------------
NTRAIN_LIST=(10 64 119 173 228 282 337 391 446 500)

# 3 seeds per (config, alpha)
SEEDS=(0 1 2)

# -----------------------
# UNIQUE PTFT config list (20 total)
# Each line: rho_pt  rho_ft  omega  lambda_pt  gamma_reinit
# Fixed across all lines: a_pt=1.0, c_pt=0.001
#
# This list is the union of experiments (2)-(8) from the replica sweep,
# with the final constraint that all overlap-varying families use rho_pt=0.1.
# -----------------------
CFG_LINES=(
  # --- High rho_pt, full overlap, rho_ft=0.1 ---
  "0.9999  0.1  1.0  -0.00099  0.0"   # (2) lambda sweep
  "0.9999  0.1  1.0   0.0      0.0"   # (2) lambda sweep (center)
  "0.9999  0.1  1.0   0.00099  0.0"   # (2) lambda sweep
  "0.9999  0.1  1.0   0.0      1.0"   # (3) gamma sweep (gamma in {0,1}) adds gamma=1

  # --- Overlap-varying families at rho_pt=0.1, rho_ft=0.1 ---
  # omega=0, lambda=0, gamma in {0,0.1,1}
  "0.1     0.1  0.0   0.0      0.0"   # (4)/(6)
  "0.1     0.1  0.0   0.0      0.1"   # (6)
  "0.1     0.1  0.0   0.0      1.0"   # (6)
  # omega=0, lambda in {-0.00099,+0.00099}, gamma=0
  "0.1     0.1  0.0  -0.00099  0.0"   # (5)
  "0.1     0.1  0.0   0.00099  0.0"   # (5)
  # omega=0.5, lambda=0, gamma=0
  "0.1     0.1  0.5   0.0      0.0"   # (4)
  # omega=1, lambda=0, gamma in {0,0.1,1}
  "0.1     0.1  1.0   0.0      0.0"   # (4)/(6)
  "0.1     0.1  1.0   0.0      0.1"   # (6)
  "0.1     0.1  1.0   0.0      1.0"   # (6)
  # omega=1, lambda in {-0.00099,+0.00099}, gamma=0
  "0.1     0.1  1.0  -0.00099  0.0"   # (5)
  "0.1     0.1  1.0   0.00099  0.0"   # (5)

  # --- High rho_pt, full overlap, rho_ft=0.9 (rho_ft × lambda sweep) ---
  "0.9999  0.9  1.0  -0.00099  0.0"   # (7)
  "0.9999  0.9  1.0   0.0      0.0"   # (7)
  "0.9999  0.9  1.0   0.00099  0.0"   # (7)

  # --- Overlap-varying family at rho_pt=0.1, rho_ft=0.9, omega=0 (rho_ft × gamma sweep) ---
  "0.1     0.9  0.0   0.0      0.0"   # (8)
  "0.1     0.9  0.0   0.0      1.0"   # (8)
)

N_CFG=${#CFG_LINES[@]}
N_ALPHA=${#NTRAIN_LIST[@]}
N_SEED=${#SEEDS[@]}

if [ "$N_CFG" -ne 20 ]; then
  echo "[SLURM] Expected 20 unique PTFT configs, got $N_CFG" >&2
  exit 2
fi
if [ "$N_ALPHA" -ne 10 ]; then
  echo "[SLURM] Expected 10 alpha points, got $N_ALPHA" >&2
  exit 2
fi
if [ "$N_SEED" -ne 3 ]; then
  echo "[SLURM] Expected 3 seeds, got $N_SEED" >&2
  exit 2
fi

TOTAL=$((N_CFG * N_ALPHA * N_SEED))
if [ "$TOTAL" -ne 600 ]; then
  echo "[SLURM] Expected TOTAL=600 tasks, got $TOTAL" >&2
  exit 2
fi

TID=${SLURM_ARRAY_TASK_ID}
if [ "$TID" -lt 0 ] || [ "$TID" -ge "$TOTAL" ]; then
  echo "[SLURM] Task id $TID out of range (0..$((TOTAL-1)))" >&2
  exit 2
fi

PER_CFG=$((N_ALPHA * N_SEED))   # 30 tasks per config
CFG_ID=$((TID / PER_CFG))       # 0..19
REM=$((TID % PER_CFG))          # 0..29
AIDX=$((REM / N_SEED))          # 0..9
SIDX=$((REM % N_SEED))          # 0..2

N_TRAIN=${NTRAIN_LIST[$AIDX]}
SEED=${SEEDS[$SIDX]}

read -r rho_pt rho_ft omega lambda_pt gamma_reinit <<< "${CFG_LINES[$CFG_ID]}"

# alpha_eff for folder naming (string, fixed precision)
alpha_eff="$($PY - <<PY
print(f"{int($N_TRAIN)/int($INP_DIM):.6f}")
PY
)"

RUN_NAME="${RUN_NAME:-ptft_oracle_empirical_alpha10_lr0p1}"
OUT_BASE="results/diagonal/ptft_oracle_empirical_alpha10_lr0p1/${RUN_NAME}"

CFG_TAG="rpt=${rho_pt}__rft=${rho_ft}__om=${omega}__cpt=${C_PT}__lpt=${lambda_pt}__gam=${gamma_reinit}"
SAVE_FOLDER="${OUT_BASE}/${CFG_TAG}/alpha=${alpha_eff}__seed=${SEED}"

# Idempotency: if df.feather exists, assume run completed.
if [ -f "${SAVE_FOLDER}/df.feather" ]; then
  echo "[SKIP] exists: ${SAVE_FOLDER}/df.feather"
  exit 0
fi

mkdir -p "${SAVE_FOLDER}"

echo "[task ${TID}] cfg=${CFG_ID}/${N_CFG} alpha_idx=${AIDX}/${N_ALPHA} seed_idx=${SIDX}/${N_SEED}"
echo "  n_train=${N_TRAIN} alpha_eff=${alpha_eff} seed=${SEED}"
echo "  rho_pt=${rho_pt} rho_ft=${rho_ft} omega=${omega} lambda_pt=${lambda_pt} gamma_reinit=${gamma_reinit}"
echo "  save_folder=${SAVE_FOLDER}"
echo "  lr=${LR} epochs=${EPOCHS} stop_pred_mse=${THRESH}"

cmd=(
  "$PY" experiments/diagonal/diagonal_ptft_oracle.py
  --seed "${SEED}"
  --save_folder "${SAVE_FOLDER}"
  --inp_dim "${INP_DIM}"
  --n_train "${N_TRAIN}"
  --n_test "${N_TEST}"
  --rho_pt "${rho_pt}"
  --rho_ft "${rho_ft}"
  --omega "${omega}"
  --a_pt "${A_PT}"
  --c_pt "${C_PT}"
  --lambda_pt "${lambda_pt}"
  --gamma_reinit "${gamma_reinit}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --threshold "${THRESH}"
  --stop_pred_mse "${THRESH}"
  --test_every_n_epochs "${TEST_EVERY}"
)
if [ "${NO_TUNING}" -eq 1 ]; then
  cmd+=(--no_tuning)
fi

echo "[SLURM] Running: ${cmd[*]}"
exec "${cmd[@]}"

