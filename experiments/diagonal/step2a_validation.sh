#!/bin/bash
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/step2a_val_%A_%a.out
#SBATCH --error=logs/step2a_val_%A_%a.err

set -euo pipefail

echo "[SLURM] Job $SLURM_JOB_ID, Array task $SLURM_ARRAY_TASK_ID starting on $(hostname)"

# Ensure we run from project root and logs exist
cd /home/na658/multi-task2
mkdir -p logs
mkdir -p figures/step2a_validation

# Use absolute Python from conda env to avoid activation issues on compute nodes
PY="/home/na658/.conda/envs/mtl_ft/bin/python"
if [ ! -x "$PY" ]; then
  echo "[SLURM] Expected Python not found at $PY" >&2
  exit 1
fi

echo "[SLURM] Using Python: $PY"

OUTPUT_DIR="figures/step2a_validation"

# Array task mapping:
# 0-4: Plot A (omega sweep): omega in {0.0, 0.25, 0.5, 0.75, 1.0}
# 5-6: Plot B (PT irrelevance, omega=0): B1=baseline, B2=extreme PT params
# 7-8: Plot C (BG baseline vs ptft_oracle): C1=bg, C2=ptft_oracle omega=0
# 9-12: Plot D (a_pt sweep): a_pt in {0.0, 0.5, 1.0, 2.0}
# 13-14: Plot E (c_values irrelevance): c=0.001, c=0.5

case $SLURM_ARRAY_TASK_ID in
  # ===== Plot A: Omega sweep =====
  0)
    echo "[Plot A] omega=0.0"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.0 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  1)
    echo "[Plot A] omega=0.25"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.25 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  2)
    echo "[Plot A] omega=0.5"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.5 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  3)
    echo "[Plot A] omega=0.75"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.75 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  4)
    echo "[Plot A] omega=1.0"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 1.0 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;

  # ===== Plot B: PT irrelevance when omega=0 =====
  5)
    echo "[Plot B1] omega=0, baseline PT params"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.0 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  6)
    echo "[Plot B2] omega=0, extreme PT params"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.0 \
      --a_pt 10.0 --c_pt 0.001 --lambda_pt 1.0 --gamma_reinit 5.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;

  # ===== Plot C: BG baseline vs ptft_oracle (omega=0) =====
  7)
    echo "[Plot C1] BG baseline (rho=0.04)"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode bg \
      --rho 0.04 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  8)
    echo "[Plot C2] ptft_oracle omega=0"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.0 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;

  # ===== Plot D: a_pt sweep (omega=1.0) =====
  9)
    echo "[Plot D] a_pt=0.0"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 1.0 \
      --a_pt 0.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  10)
    echo "[Plot D] a_pt=0.5"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 1.0 \
      --a_pt 0.5 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  11)
    echo "[Plot D] a_pt=1.0"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 1.0 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;
  12)
    echo "[Plot D] a_pt=2.0"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 1.0 \
      --a_pt 2.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --output_dir $OUTPUT_DIR
    ;;

  # ===== Plot E: c_values irrelevance =====
  13)
    echo "[Plot E1] c_values=0.001"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.5 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --c_values 0.001 \
      --output_dir $OUTPUT_DIR
    ;;
  14)
    echo "[Plot E2] c_values=0.5"
    $PY scripts/diagonal/plot_replica_q_bg.py \
      --teacher_mode ptft_oracle \
      --rho_pt 0.10 --rho_ft 0.04 --omega 0.5 \
      --a_pt 1.0 --c_pt 0.001 --lambda_pt 0.0 --gamma_reinit 0.0 \
      --ft_regulariser_scale 1e-6 --mc_samples 50000 --seed 12345 \
      --c_values 0.5 \
      --output_dir $OUTPUT_DIR
    ;;

  *)
    echo "[SLURM] Unknown array task ID: $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo "[SLURM] Job $SLURM_JOB_ID, Array task $SLURM_ARRAY_TASK_ID finished"



