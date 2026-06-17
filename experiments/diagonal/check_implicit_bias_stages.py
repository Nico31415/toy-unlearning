#!/usr/bin/env python3
"""Finite-dimensional checks of the DLN implicit-bias formula.

This script does not use replica/state-evolution approximations.  For each
stage it:

  1. trains the two-pathway diagonal network with small-step full-batch
     gradient descent, approximating parameter gradient flow;
  2. computes the exact stage-start Bregman centre and geometry from the
     actual weights;
  3. solves the finite-data constrained Bregman projection from the KKT
     equations; and
  4. compares the trained predictor with the constrained optimum.

Run from the repo root, preferably in the project conda environment:

    python experiments/diagonal/check_implicit_bias_stages.py
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EPS_K = 1e-30


@dataclass(frozen=True)
class Weights:
    w_pos: np.ndarray
    w_neg: np.ndarray
    v_pos: np.ndarray
    v_neg: np.ndarray

    def copy(self) -> "Weights":
        return Weights(
            self.w_pos.copy(),
            self.w_neg.copy(),
            self.v_pos.copy(),
            self.v_neg.copy(),
        )


@dataclass(frozen=True)
class Combo:
    name: str
    c_pt: float
    lambda_frac: float
    alpha_in: float
    alpha_out: float
    reset_mode: str = "mode_b"
    gamma_reinit: float = 0.0
    stage3_reset_mode: str = "continue"
    stage3_gamma_reinit: float = 0.0
    stage3_alpha_in: float = 1.0

    @property
    def lambda_pt(self) -> float:
        return self.lambda_frac * self.c_pt


def default_combos() -> list[Combo]:
    return [
        Combo("balanced_c0.10", c_pt=0.10, lambda_frac=0.0, alpha_in=1.0, alpha_out=1.0),
        Combo("balanced_c0.35", c_pt=0.35, lambda_frac=0.0, alpha_in=1.0, alpha_out=1.0),
        Combo("input_heavy", c_pt=0.15, lambda_frac=0.0, alpha_in=2.0, alpha_out=0.5),
        Combo("output_heavy", c_pt=0.15, lambda_frac=0.0, alpha_in=0.5, alpha_out=2.0),
        Combo("lambda_input_heavy", c_pt=0.20, lambda_frac=0.7, alpha_in=2.0, alpha_out=0.5),
        Combo("modeA_gamma0.00", c_pt=0.10, lambda_frac=0.0, alpha_in=1.0, alpha_out=1.0,
              reset_mode="mode_a", gamma_reinit=0.0),
        Combo("modeA_gamma0.05", c_pt=0.10, lambda_frac=0.0, alpha_in=1.0, alpha_out=1.0,
              reset_mode="mode_a", gamma_reinit=0.05),
        Combo("modeA_lambda_gamma", c_pt=0.20, lambda_frac=0.5, alpha_in=1.0, alpha_out=1.0,
              reset_mode="mode_a", gamma_reinit=0.05),
        Combo("stage3_resetA_g0", c_pt=0.10, lambda_frac=0.0, alpha_in=1.0, alpha_out=1.0,
              stage3_reset_mode="mode_a", stage3_gamma_reinit=0.0),
        Combo("stage3_resetA_g005", c_pt=0.10, lambda_frac=0.0, alpha_in=1.0, alpha_out=1.0,
              stage3_reset_mode="mode_a", stage3_gamma_reinit=0.05),
        Combo("modeA_then_stage3_resetA", c_pt=0.20, lambda_frac=0.5, alpha_in=1.0, alpha_out=1.0,
              reset_mode="mode_a", gamma_reinit=0.05,
              stage3_reset_mode="mode_a", stage3_gamma_reinit=0.05),
    ]


def complex_init(dim: int, c: float, lmda: float) -> Weights:
    """Complex init in the convention used by the LaTeX notes."""
    if c * c < lmda * lmda:
        raise ValueError(f"Require c^2 >= lambda^2, got c={c}, lambda={lmda}.")
    w = math.sqrt((c - lmda) / 2.0)
    v = math.sqrt((c + lmda) / 2.0)
    return Weights(
        w_pos=np.full(dim, w, dtype=np.float64),
        w_neg=np.full(dim, w, dtype=np.float64),
        v_pos=np.full(dim, v, dtype=np.float64),
        v_neg=np.full(dim, v, dtype=np.float64),
    )


def beta_from_weights(weights: Weights) -> np.ndarray:
    return weights.w_pos * weights.v_pos - weights.w_neg * weights.v_neg


def k_from_weights(weights: Weights) -> np.ndarray:
    c_i = weights.w_pos * weights.w_neg + weights.v_pos * weights.v_neg
    lambda_plus = weights.w_pos**2 - weights.v_pos**2
    lambda_minus = weights.w_neg**2 - weights.v_neg**2
    return (lambda_plus - lambda_minus) ** 2 + 4.0 * c_i**2


def apply_mode_b(weights: Weights, alpha_in: float, alpha_out: float) -> Weights:
    return Weights(
        w_pos=alpha_in * weights.w_pos,
        w_neg=alpha_in * weights.w_neg,
        v_pos=alpha_out * weights.v_pos,
        v_neg=alpha_out * weights.v_neg,
    )


def apply_mode_a(weights: Weights, gamma_reinit: float, input_scale: float = 1.0) -> Weights:
    """Readout reinit + symmetric input reset, matching functions.unlearning.

    This uses the actual reset weights for the finite-dimensional check.  The
    resulting Bregman centre and geometry are then computed directly from those
    weights, so the test is independent of any closed-form approximation.
    """
    symmetric_input = input_scale * (weights.w_pos + weights.w_neg)
    readout = np.full_like(symmetric_input, float(gamma_reinit))
    return Weights(
        w_pos=symmetric_input.copy(),
        w_neg=symmetric_input.copy(),
        v_pos=readout.copy(),
        v_neg=readout.copy(),
    )


def apply_stage_reset(weights: Weights, combo: Combo) -> Weights:
    if combo.reset_mode == "mode_b":
        return apply_mode_b(weights, combo.alpha_in, combo.alpha_out)
    if combo.reset_mode == "mode_a":
        return apply_mode_a(weights, combo.gamma_reinit, input_scale=combo.alpha_in)
    raise ValueError(f"Unknown reset mode: {combo.reset_mode}")


def apply_stage3_reset(weights: Weights, combo: Combo) -> Weights:
    if combo.stage3_reset_mode == "continue":
        return weights
    if combo.stage3_reset_mode == "mode_a":
        return apply_mode_a(
            weights,
            combo.stage3_gamma_reinit,
            input_scale=combo.stage3_alpha_in,
        )
    raise ValueError(f"Unknown Stage 3 reset mode: {combo.stage3_reset_mode}")


def q_k(z: np.ndarray, k: np.ndarray) -> np.ndarray:
    k = np.maximum(np.asarray(k, dtype=np.float64), EPS_K)
    z = np.asarray(z, dtype=np.float64)
    sqrt_k = np.sqrt(k)
    t = 2.0 * z / sqrt_k
    return (sqrt_k / 4.0) * (1.0 - np.sqrt(1.0 + t * t) + t * np.arcsinh(t))


def q_prime(z: np.ndarray, k: np.ndarray) -> np.ndarray:
    k = np.maximum(np.asarray(k, dtype=np.float64), EPS_K)
    return 0.5 * np.arcsinh(2.0 * np.asarray(z, dtype=np.float64) / np.sqrt(k))


def inv_q_prime(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    k = np.maximum(np.asarray(k, dtype=np.float64), EPS_K)
    # The check cases are deliberately moderate; clipping only protects failed
    # Newton iterates during line search from numerical overflow.
    return 0.5 * np.sqrt(k) * np.sinh(np.clip(2.0 * p, -60.0, 60.0))


def bregman_objective(beta: np.ndarray, beta0: np.ndarray, k: np.ndarray) -> float:
    div = q_k(beta, k) - q_k(beta0, k) - q_prime(beta0, k) * (beta - beta0)
    return float(np.sum(div))


def solve_bregman_projection(
    design: np.ndarray,
    target_y: np.ndarray,
    beta0: np.ndarray,
    k: np.ndarray,
    *,
    tol: float = 1e-11,
    max_newton: int = 100,
) -> tuple[np.ndarray, dict]:
    """Solve min sum_i D_qk(beta_i,beta0_i) s.t. design @ beta = target_y.

    The KKT equations reduce to a dual problem in one variable per constraint:

        q'_k(beta_i) = q'_k(beta0_i) - (design.T @ nu)_i.

    We solve the resulting nonlinear constraint equation by damped Newton.
    """
    design = np.asarray(design, dtype=np.float64)
    target_y = np.asarray(target_y, dtype=np.float64)
    beta0 = np.asarray(beta0, dtype=np.float64)
    k = np.maximum(np.asarray(k, dtype=np.float64), EPS_K)

    if np.linalg.matrix_rank(design) < design.shape[0]:
        raise ValueError("Design matrix must have full row rank for this check.")

    dual = np.zeros(design.shape[0], dtype=np.float64)
    q0 = q_prime(beta0, k)
    residual_norm = float("inf")

    for it in range(max_newton):
        beta = inv_q_prime(q0 - design.T @ dual, k)
        residual = design @ beta - target_y
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm < tol:
            return beta, {
                "dual_iters": it,
                "dual_residual": residual_norm,
                "dual_converged": True,
            }

        local_speed = np.sqrt(k + 4.0 * beta**2)
        jac = -(design * local_speed) @ design.T
        try:
            step_dir = np.linalg.solve(jac, -residual)
        except np.linalg.LinAlgError:
            step_dir = np.linalg.lstsq(jac, -residual, rcond=None)[0]

        step = 1.0
        accepted = False
        for _ in range(35):
            candidate_dual = dual + step * step_dir
            candidate_beta = inv_q_prime(q0 - design.T @ candidate_dual, k)
            candidate_norm = float(np.linalg.norm(design @ candidate_beta - target_y))
            if candidate_norm < residual_norm:
                dual = candidate_dual
                accepted = True
                break
            step *= 0.5

        if not accepted:
            dual = dual + step_dir

    beta = inv_q_prime(q0 - design.T @ dual, k)
    residual_norm = float(np.linalg.norm(design @ beta - target_y))
    return beta, {
        "dual_iters": max_newton,
        "dual_residual": residual_norm,
        "dual_converged": False,
    }


def train_dln(
    initial: Weights,
    design: np.ndarray,
    target_y: np.ndarray,
    *,
    lr: float,
    tol: float,
    max_steps: int,
) -> tuple[Weights, dict]:
    """Small-step full-batch GD on parameters, approximating gradient flow."""
    weights = initial.copy()
    n_samples = design.shape[0]
    best_loss = float("inf")
    lr_current = float(lr)
    steps_done = 0
    restarts = 0

    # If a step overflows or dramatically increases loss, restart from the
    # stage initialisation with a smaller step size.
    while restarts <= 8:
        weights = initial.copy()
        prev_loss = float("inf")
        for step in range(max_steps):
            beta = beta_from_weights(weights)
            residual = target_y - design @ beta
            loss = 0.5 * float(np.mean(residual**2))
            best_loss = min(best_loss, loss)
            steps_done = step + 1
            if loss < tol:
                return weights, {
                    "loss": loss,
                    "steps": steps_done,
                    "lr": lr_current,
                    "restarts": restarts,
                    "converged": True,
                }
            if (not np.isfinite(loss)) or (loss > 20.0 * prev_loss and step > 50):
                lr_current *= 0.5
                restarts += 1
                break

            grad_signal = design.T @ residual / n_samples
            w_pos, w_neg = weights.w_pos, weights.w_neg
            v_pos, v_neg = weights.v_pos, weights.v_neg
            weights = Weights(
                w_pos=w_pos + lr_current * grad_signal * v_pos,
                w_neg=w_neg - lr_current * grad_signal * v_neg,
                v_pos=v_pos + lr_current * grad_signal * w_pos,
                v_neg=v_neg - lr_current * grad_signal * w_neg,
            )
            prev_loss = loss
        else:
            return weights, {
                "loss": best_loss,
                "steps": max_steps,
                "lr": lr_current,
                "restarts": restarts,
                "converged": False,
            }

    return weights, {
        "loss": best_loss,
        "steps": steps_done,
        "lr": lr_current,
        "restarts": restarts,
        "converged": False,
    }


def make_sparse_teacher(dim: int, active_dim: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(dim, dtype=np.float64)
    active = rng.choice(dim, size=active_dim, replace=False)
    beta[active] = rng.choice([-1.0, 1.0], size=active_dim) / math.sqrt(active_dim)
    return beta, active


def make_design(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    # Rows have expected norm 1, matching the scaling in the LaTeX document.
    design = rng.normal(0.0, 1.0 / math.sqrt(dim), size=(n_samples, dim))
    if np.linalg.matrix_rank(design) < n_samples:
        return make_design(n_samples, dim, rng)
    return design


def stage_result(
    *,
    combo: Combo,
    stage: str,
    initial: Weights,
    design: np.ndarray,
    target_beta: np.ndarray,
    lr: float,
    train_tol: float,
    max_steps: int,
    dual_tol: float,
) -> tuple[Weights, dict]:
    target_y = design @ target_beta
    beta0 = beta_from_weights(initial)
    k0 = k_from_weights(initial)

    final_weights, train_info = train_dln(
        initial,
        design,
        target_y,
        lr=lr,
        tol=train_tol,
        max_steps=max_steps,
    )
    beta_trained = beta_from_weights(final_weights)
    beta_implicit, dual_info = solve_bregman_projection(
        design,
        target_y,
        beta0,
        k0,
        tol=dual_tol,
    )

    diff = beta_trained - beta_implicit
    implicit_norm = max(float(np.linalg.norm(beta_implicit)), 1e-12)
    train_constraint = float(np.linalg.norm(design @ beta_trained - target_y))
    implicit_constraint = float(np.linalg.norm(design @ beta_implicit - target_y))
    k_drift = float(np.max(np.abs(k_from_weights(final_weights) - k0)))

    result = {
        "combo": combo.name,
        "stage": stage,
        "c_pt": combo.c_pt,
        "lambda_pt": combo.lambda_pt,
        "lambda_frac": combo.lambda_frac,
        "alpha_in": combo.alpha_in,
        "alpha_out": combo.alpha_out,
        "reset_mode": combo.reset_mode,
        "gamma_reinit": combo.gamma_reinit,
        "stage3_reset_mode": combo.stage3_reset_mode,
        "stage3_gamma_reinit": combo.stage3_gamma_reinit,
        "stage3_alpha_in": combo.stage3_alpha_in,
        "train_converged": train_info["converged"],
        "train_loss": train_info["loss"],
        "train_steps": train_info["steps"],
        "train_lr": train_info["lr"],
        "train_restarts": train_info["restarts"],
        "constraint_residual_train": train_constraint,
        "constraint_residual_implicit": implicit_constraint,
        "dual_converged": dual_info["dual_converged"],
        "dual_iters": dual_info["dual_iters"],
        "dual_residual": dual_info["dual_residual"],
        "beta_l2_abs": float(np.linalg.norm(diff)),
        "beta_l2_rel": float(np.linalg.norm(diff) / implicit_norm),
        "beta_linf": float(np.max(np.abs(diff))),
        "bregman_obj_train": bregman_objective(beta_trained, beta0, k0),
        "bregman_obj_implicit": bregman_objective(beta_implicit, beta0, k0),
        "k_start_min": float(np.min(k0)),
        "k_start_max": float(np.max(k0)),
        "k_drift_linf": k_drift,
    }
    result["bregman_obj_gap"] = result["bregman_obj_train"] - result["bregman_obj_implicit"]
    return final_weights, result


def run_checks(args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    combos = default_combos()
    beta_pt, active = make_sparse_teacher(args.dim, args.active_dim, rng)

    shuffled_active = np.array(active, copy=True)
    rng.shuffle(shuffled_active)
    split = max(1, len(shuffled_active) // 2)
    forget_idx = shuffled_active[:split]
    retain_idx = shuffled_active[split:]

    beta_forget = np.zeros(args.dim, dtype=np.float64)
    beta_retain = np.zeros(args.dim, dtype=np.float64)
    beta_forget[forget_idx] = beta_pt[forget_idx]
    beta_retain[retain_idx] = beta_pt[retain_idx]

    rows: list[dict] = []
    for combo_id, combo in enumerate(combos):
        combo_rng = np.random.default_rng(args.seed + 1000 * (combo_id + 1))
        design_pt = make_design(args.n_per_stage, args.dim, combo_rng)
        design_ul = make_design(args.n_per_stage, args.dim, combo_rng)
        design_rl = make_design(args.n_per_stage, args.dim, combo_rng)

        init_stage1 = complex_init(args.dim, combo.c_pt, combo.lambda_pt)
        weights_stage1, row = stage_result(
            combo=combo,
            stage="stage1_pretrain",
            initial=init_stage1,
            design=design_pt,
            target_beta=beta_pt,
            lr=args.lr,
            train_tol=args.train_tol,
            max_steps=args.max_steps,
            dual_tol=args.dual_tol,
        )
        rows.append(row)

        init_stage2 = apply_stage_reset(weights_stage1, combo)
        weights_stage2, row = stage_result(
            combo=combo,
            stage="stage2_unlearn",
            initial=init_stage2,
            design=design_ul,
            target_beta=beta_retain,
            lr=args.lr,
            train_tol=args.train_tol,
            max_steps=args.max_steps,
            dual_tol=args.dual_tol,
        )
        rows.append(row)

        # A forget-only relearning target isolates the adversary's recovered
        # capability.  The theorem only needs a squared effective target, so this
        # finite-dimensional check is valid for either forget-only or full PT.
        init_stage3 = apply_stage3_reset(weights_stage2, combo)
        stage3_name = (
            "stage3_relearn"
            if combo.stage3_reset_mode == "continue"
            else f"stage3_relearn_{combo.stage3_reset_mode}"
        )
        _, row = stage_result(
            combo=combo,
            stage=stage3_name,
            initial=init_stage3,
            design=design_rl,
            target_beta=beta_forget,
            lr=args.lr,
            train_tol=args.train_tol,
            max_steps=args.max_steps,
            dual_tol=args.dual_tol,
        )
        rows.append(row)

    return rows


def write_csv(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print("\nFinite-dimensional implicit-bias check")
    print("combo                  stage              rel_l2      linf       train_res  dual_res   k_drift")
    print("-" * 104)
    for row in rows:
        print(
            f"{row['combo']:<22} {row['stage']:<18} "
            f"{row['beta_l2_rel']:<10.3e} {row['beta_linf']:<10.3e} "
            f"{row['constraint_residual_train']:<10.3e} "
            f"{row['dual_residual']:<10.3e} {row['k_drift_linf']:<10.3e}"
        )
    worst = max(rows, key=lambda r: r["beta_l2_rel"])
    print(
        "\nWorst relative beta mismatch: "
        f"{worst['beta_l2_rel']:.3e} "
        f"({worst['combo']} / {worst['stage']})."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=12)
    parser.add_argument("--active_dim", type=int, default=4)
    parser.add_argument("--n_per_stage", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--train_tol", type=float, default=1e-12)
    parser.add_argument("--dual_tol", type=float, default=1e-11)
    parser.add_argument("--max_steps", type=int, default=400_000)
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("results/implicit_bias_checks/stage_implicit_bias_check.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_per_stage >= args.dim:
        raise ValueError("--n_per_stage must be smaller than --dim for an underdetermined check.")
    if args.active_dim < 2:
        raise ValueError("--active_dim must be at least 2 so forget/retain can both be nonempty.")

    rows = run_checks(args)
    write_csv(rows, args.out_csv)
    print_summary(rows)
    print(f"\nWrote detailed results to {args.out_csv}")


if __name__ == "__main__":
    main()
