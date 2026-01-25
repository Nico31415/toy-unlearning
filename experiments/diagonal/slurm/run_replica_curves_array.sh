#!/bin/bash
#
# Slurm array: one replica curve per task (32 total).
#
# Writes caches to:
#   figures/diagonal/bg_generalization/replica_cache/
#
# Notes:
# - We set --empirical_dir to a non-existent path to avoid overlay-plot races
#   (plot_replica_q_bg.py only overlays if empirical CSVs are found).
# - For PTFT runs, set --rho equal to --rho_ft (solver config uses --rho).
#
# Submit:
#   sbatch experiments/diagonal/slurm/run_replica_curves_array.sh
#

#SBATCH --job-name=replica_curves
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --array=0-31
#SBATCH --output=slurm/replica_curves_%A_%a.out
#SBATCH --error=slurm/replica_curves_%A_%a.err

set -euo pipefail

# Avoid oversubscription when running many tasks at once.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="scripts/diagonal/plot_replica_q_bg.py"

OUTDIR="figures/diagonal/bg_generalization"
EMPIRICAL_DIR="__no_empirical__"   # intentionally missing

# Shared knobs (per your spec)
ALPHA_MIN="0.01"
ALPHA_MAX="0.5"
ALPHA_POINTS="30"
MC_SAMPLES="30000"
SEED="12345"
FTREG="1e-6"

# PTFT shared knobs
APT="1.0"
CPT="0.001"
LNEG="-0.00099"   # -0.99 * c_pt with c_pt=0.001
L0="0.0"
LPOS="0.00099"

common_flags=(
  --ft_regulariser_scale "${FTREG}"
  --alpha_min "${ALPHA_MIN}"
  --alpha_max "${ALPHA_MAX}"
  --alpha_points "${ALPHA_POINTS}"
  --mc_samples "${MC_SAMPLES}"
  --seed "${SEED}"
  --output_dir "${OUTDIR}"
  --empirical_dir "${EMPIRICAL_DIR}"
)

cmds=(
  # --------------------------------------------------------------------------
  # (1) Single-task BG: rho=0.1, c in {1, 1e-3}
  # --------------------------------------------------------------------------
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode bg --rho 0.1 --c_values 1.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode bg --rho 0.1 --c_values 0.001 ${common_flags[*]}"

  # --------------------------------------------------------------------------
  # (2) PT+FT oracle (full overlap), lambda_pt sweep, high rho_pt
  #   rho_pt=0.9999, rho_ft=0.1, omega=1, c_pt=1e-3, gamma_reinit=0
  #   lambda_pt in {-0.00099, 0, +0.00099}
  # --------------------------------------------------------------------------
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LNEG} --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${L0}   --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LPOS} --gamma_reinit 0.0 ${common_flags[*]}"

  # --------------------------------------------------------------------------
  # (3) PT+FT oracle (full overlap), gamma_reinit sweep, high rho_pt
  #   rho_pt=0.9999, rho_ft=0.1, omega=1, c_pt=1e-3, lambda_pt=0
  #   gamma_reinit in {0, 1}
  # --------------------------------------------------------------------------
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 1.0 ${common_flags[*]}"

  # --------------------------------------------------------------------------
  # (4) PT+FT oracle, omega sweep (overlap varying)  [rho_pt=0.1]
  #   rho_pt=0.1, rho_ft=0.1, c_pt=1e-3, lambda_pt=0, gamma_reinit=0
  #   omega in {0, 0.5, 1}
  # --------------------------------------------------------------------------
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.5 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.0 ${common_flags[*]}"

  # --------------------------------------------------------------------------
  # (5) PT+FT oracle, joint lambda_pt × omega sweep  [rho_pt=0.1]
  #   rho_pt=0.1, rho_ft=0.1, c_pt=1e-3, gamma_reinit=0
  #   omega in {0,1} and lambda in {-0.00099,0,+0.00099}
  # --------------------------------------------------------------------------
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LNEG} --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${L0}   --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LPOS} --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LNEG} --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${L0}   --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LPOS} --gamma_reinit 0.0 ${common_flags[*]}"

  # --------------------------------------------------------------------------
  # (6) PT+FT oracle, joint gamma_reinit × omega sweep  [rho_pt=0.1]
  #   rho_pt=0.1, rho_ft=0.1, c_pt=1e-3, lambda_pt=0
  #   omega in {0,1} and gamma_reinit in {0,0.1,1}
  # --------------------------------------------------------------------------
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.1 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 1.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.1 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 1.0 ${common_flags[*]}"

  # --------------------------------------------------------------------------
  # (7) PT+FT oracle, joint rho_ft × lambda_pt sweep, high rho_pt, full overlap
  #   rho_pt=0.9999, omega=1, c_pt=1e-3, gamma_reinit=0
  #   rho_ft in {0.1,0.9} and lambda in {-0.00099,0,+0.00099}
  #   IMPORTANT: set --rho to match rho_ft.
  # --------------------------------------------------------------------------
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LNEG} --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${L0}   --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.1 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LPOS} --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.9 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.9 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LNEG} --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.9 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.9 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${L0}   --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.9 --c_values 0.001 --rho_pt 0.9999 --rho_ft 0.9 --omega 1.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt ${LPOS} --gamma_reinit 0.0 ${common_flags[*]}"

  # --------------------------------------------------------------------------
  # (8) PT+FT oracle, joint rho_ft × gamma_reinit sweep, overlap varying (omega=0)  [rho_pt=0.1]
  #   rho_pt=0.1, omega=0, c_pt=1e-3, lambda_pt=0
  #   rho_ft in {0.1,0.9} and gamma_reinit in {0,1}
  #   IMPORTANT: set --rho to match rho_ft.
  # --------------------------------------------------------------------------
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.1 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.1 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 1.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.9 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.9 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 0.0 ${common_flags[*]}"
  "${PYTHON_BIN} ${SCRIPT} --teacher_mode ptft_oracle --rho 0.9 --c_values 0.001 --rho_pt 0.1 --rho_ft 0.9 --omega 0.0 --a_pt ${APT} --c_pt ${CPT} --lambda_pt 0.0 --gamma_reinit 1.0 ${common_flags[*]}"
)

task_id="${SLURM_ARRAY_TASK_ID}"
if [[ "${task_id}" -lt 0 || "${task_id}" -ge "${#cmds[@]}" ]]; then
  echo "Invalid task id ${task_id}; have ${#cmds[@]} commands." >&2
  exit 2
fi

echo "Task ${task_id}/${#cmds[@]} running:"
echo "${cmds[${task_id}]}"
eval "${cmds[${task_id}]}"
