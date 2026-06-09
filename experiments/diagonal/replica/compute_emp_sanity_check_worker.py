"""
Empirical sanity-check worker for the forgetting framework.

Runs three FT teacher conventions across three regimes to verify
that the replica theory forgetting curves match empirical measurements.

Teacher conventions:
  aligned_overlap  — β*_FT[g=0] = β*_PT[g=0]   (same sign, same magnitude)
  zero_overlap     — β*_FT[g=0] = 0              (FT ignores shared features)
  opposite_overlap — β*_FT[g=0] = -β*_PT[g=0]   (flipped sign on shared features)

Analytic predictions at α→∞:
  F^overlap:  aligned→0,  zero→1,  opposite→4
  F^ptonly:   all→1

Regimes:
  regime_II  — lazy:           c=1e-3, λ=0,    γ=0
  regime_III — PT-independent: c=1e-3, λ=0,    γ=10
  regime_IV  — rich/selective: c=1e-3, λ=-0.95c by default, γ=0

Fixed unless passed as CLI flags: ρ_PT=ρ_FT=0.1, ω=0.5, D=5000, a_PT=1

Run individual tasks:
  python experiments/diagonal/replica/compute_emp_sanity_check_worker.py \
    --task-id 0 --omega 0.5 --regime-iv-lambda-mult -0.95 \
    --output-dir results/sanity_check

Total tasks: 3 regimes × 3 norms × 11 alphas × 5 seeds = 495
"""
import argparse
import itertools
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path

from unlearning_experiment_utils import SANITY_ALPHAS, SANITY_TEACHERS, append_csv_locked, regimes_with_iv_lambda


TEACHER_NORMS = SANITY_TEACHERS
ALPHAS = SANITY_ALPHAS
SEEDS  = list(range(5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id",    type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="results/sanity_check")
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--regime-iv-lambda-mult", type=float, default=-0.95)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    regimes = regimes_with_iv_lambda(args.regime_iv_lambda_mult)
    param_combinations = [
        (regime_name, c_pt, lambda_pt, gamma_reinit, teacher_norm, seed, alpha)
        for (regime_name, c_pt, lambda_pt, gamma_reinit), teacher_norm, seed, alpha
        in itertools.product(regimes, TEACHER_NORMS, SEEDS, ALPHAS)
    ]

    total_tasks = len(param_combinations)
    if args.list:
        print(f"total_tasks={total_tasks}")
        print(f"array_range=0-{total_tasks - 1}")
        return

    if args.task_id < 0 or args.task_id >= total_tasks:
        print(f"Error: task_id {args.task_id} out of range (0..{total_tasks-1})")
        return

    import ptft_empirical_finetune_df as emp

    regime_name, c_pt, lambda_pt, gamma_reinit, teacher_norm, seed, alpha = param_combinations[args.task_id]

    print(f"Total tasks: {total_tasks}")
    print(f"Running task {args.task_id}/{total_tasks-1}")
    print(
        f"regime={regime_name} | teacher={teacher_norm} | seed={seed} | "
        f"alpha={alpha:.4f} | omega={args.omega:g} | lambda_pt={lambda_pt:g}"
    )

    save_folder = str(
        Path(args.output_dir) / regime_name / teacher_norm / f"seed{seed}_alpha{alpha:.4f}"
    )

    row = emp.run_one(
        setting="ptft",
        seed=seed,
        inp_dim=5000,
        alpha=alpha,
        n_test=10_000,
        c_pt=c_pt,
        lambda_pt=lambda_pt,
        gamma_reinit=gamma_reinit,
        rho_pt=0.1,
        a_pt=1.0,
        rho_ft=0.1,
        omega=float(args.omega),
        ft_teacher_norm=teacher_norm,
        lr=0.5,
        epochs=5_000_000,
        test_every_n_epochs=5_000,
        log_every_n_epochs=50_000,
        no_tuning=True,
        threshold=1e-4,
        stop_pred_mse=None,
        stop_beta_rate=0.0,
        stop_grad_norm=0.0,
        lr_decay=1.0,
        lr_decay_interval=2000,
        save_folder=save_folder,
    )
    row["regime"]       = regime_name
    row["teacher_norm"] = teacher_norm
    row["regime_iv_lambda_mult"] = float(args.regime_iv_lambda_mult)
    row["experiment"]   = "sanity_check"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "sanity_check_results.csv"
    append_csv_locked(master_csv, row)

    print(f"Saved weights to: {save_folder}")
    print(f"Appended row to:  {master_csv}")


if __name__ == "__main__":
    main()
