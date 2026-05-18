"""Oracle-PT implicit-bias checks for effective-teacher FT losses.

This script tests the composed derivation:

  oracle pretraining + FT init hyperparams -> Q_k
  FT loss type + P/A transforms          -> beta_eff

Then, in an underdetermined FT problem, diagonal-network fine-tuning should
converge to the minimum-Q_k interpolant subject to X_ft @ beta = X_ft @ beta_eff.

The pretraining stage is "oracle" in the sense that beta_hat_PT is set directly.
We then compute the predicted k_d from the PT hyperparameters and initialize the
FT diagonal net so that its known zero-beta geometry has that same k_d.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F

from check_effective_teacher_losses import (
    DiagonalNet,
    make_deterministic,
    make_masks,
    parameter_mse,
    prediction_mse,
    sample_teacher,
    solve_min_q_interpolant,
)


@dataclass(frozen=True)
class OracleFtCase:
    name: str
    beta_eff: torch.Tensor
    loss_fn: Callable[[torch.Tensor], torch.Tensor]


def oracle_sqrt_k(
    beta_hat_pt: torch.Tensor,
    *,
    c_pt: float,
    lambda_pt: float,
    gamma_ft: float,
) -> torch.Tensor:
    if c_pt <= 0.0:
        raise ValueError("c_pt must be positive.")
    return (
        2.0
        * c_pt
        * (1.0 + lambda_pt)
        * (1.0 + torch.sqrt(1.0 + (beta_hat_pt / c_pt) ** 2))
        + gamma_ft**2
    )


def set_zero_beta_init_from_sqrt_k(model: DiagonalNet, sqrt_k: torch.Tensor) -> None:
    """Set balanced zero-beta init with sqrt(k_d)=4*s_d^2."""
    s = 0.5 * torch.sqrt(sqrt_k)
    with torch.no_grad():
        model.w_pos.copy_(s)
        model.v_pos.copy_(s)
        model.v_neg.copy_(s)
        model.w_neg.copy_(s)


def build_ft_teacher(
    beta_hat_pt: torch.Tensor,
    retain: torch.Tensor,
    forget: torch.Tensor,
    *,
    retain_scale: float,
    forget_scale: float,
    other_scale: float,
    generator: torch.Generator,
    other_noise_scale: float,
) -> torch.Tensor:
    other = 1.0 - retain - forget
    beta_ft = retain_scale * retain * beta_hat_pt + forget_scale * forget * beta_hat_pt + other_scale * other * beta_hat_pt
    if other_noise_scale != 0.0:
        beta_ft = beta_ft + other_noise_scale * other * torch.randn(
            beta_hat_pt.shape[0], generator=generator, dtype=beta_hat_pt.dtype
        )
    return beta_ft


def build_cases(
    x_ft: torch.Tensor,
    beta_ft: torch.Tensor,
    retain: torch.Tensor,
    forget: torch.Tensor,
    *,
    signed_c_r: float,
    signed_c_f: float,
    positive_c_r: float,
    positive_c_f: float,
    generator: torch.Generator,
) -> list[OracleFtCase]:
    if signed_c_r <= signed_c_f:
        raise ValueError("Signed case requires signed_c_r > signed_c_f.")
    if positive_c_r <= 0.0 or positive_c_f <= 0.0:
        raise ValueError("Positive cases require positive c_r and c_f.")

    beta_r = retain * beta_ft
    beta_f = forget * beta_ft
    y_r = x_ft @ beta_r
    y_f = x_ft @ beta_f
    y_zero = torch.zeros(x_ft.shape[0], dtype=x_ft.dtype)

    a_r = torch.ones_like(beta_ft)
    a_f = torch.zeros_like(beta_ft)
    a_r[retain.bool()] = 0.75
    a_r[forget.bool()] = 0.10
    a_f[retain.bool()] = 0.15
    a_f[forget.bool()] = -1.25
    beta_ar = a_r * beta_ft
    beta_af = a_f * beta_ft
    y_ar = x_ft @ beta_ar
    y_af = x_ft @ beta_af

    beta_noise = torch.randn(beta_ft.shape[0], generator=generator, dtype=beta_ft.dtype)
    y_noise = x_ft @ beta_noise

    beta_eff_signed = (signed_c_r * beta_r - signed_c_f * beta_f) / (signed_c_r - signed_c_f)
    beta_eff_transform = (signed_c_r * beta_ar - signed_c_f * beta_af) / (signed_c_r - signed_c_f)
    beta_eff_zero = (positive_c_r / (positive_c_r + positive_c_f)) * beta_r
    beta_eff_noise = (positive_c_r * beta_r + positive_c_f * beta_noise) / (positive_c_r + positive_c_f)

    return [
        OracleFtCase(
            name="ordinary_ft",
            beta_eff=beta_ft,
            loss_fn=lambda pred: 0.5 * F.mse_loss(pred, x_ft @ beta_ft),
        ),
        OracleFtCase(
            name="signed_binary_mask",
            beta_eff=beta_eff_signed,
            loss_fn=lambda pred: 0.5 * signed_c_r * F.mse_loss(pred, y_r) - 0.5 * signed_c_f * F.mse_loss(pred, y_f),
        ),
        OracleFtCase(
            name="signed_coordinate_transform",
            beta_eff=beta_eff_transform,
            loss_fn=lambda pred: 0.5 * signed_c_r * F.mse_loss(pred, y_ar)
            - 0.5 * signed_c_f * F.mse_loss(pred, y_af),
        ),
        OracleFtCase(
            name="forget_to_zero",
            beta_eff=beta_eff_zero,
            loss_fn=lambda pred: 0.5 * positive_c_r * F.mse_loss(pred, y_r)
            + 0.5 * positive_c_f * F.mse_loss(pred, y_zero),
        ),
        OracleFtCase(
            name="forget_to_noise",
            beta_eff=beta_eff_noise,
            loss_fn=lambda pred: 0.5 * positive_c_r * F.mse_loss(pred, y_r)
            + 0.5 * positive_c_f * F.mse_loss(pred, y_noise),
        ),
    ]


def train_ft(
    x_ft: torch.Tensor,
    case: OracleFtCase,
    *,
    sqrt_k: torch.Tensor,
    seed: int,
    lr: float,
    epochs: int,
    pred_tol: float,
) -> tuple[torch.Tensor, float, int, str]:
    torch.manual_seed(seed)
    model = DiagonalNet(x_ft.shape[1], scaling=1.0)
    set_zero_beta_init_from_sqrt_k(model, sqrt_k)
    target = x_ft @ case.beta_eff
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    final_loss = float("nan")
    status = "max_epochs"
    final_epoch = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = case.loss_fn(model(x_ft))
        if not torch.isfinite(loss):
            status = "nonfinite_loss"
            final_loss = float(loss.detach().cpu())
            final_epoch = epoch
            break
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        final_epoch = epoch
        with torch.no_grad():
            pred_mse = F.mse_loss(model(x_ft), target).item()
        if pred_mse <= pred_tol:
            status = "pred_threshold"
            break

    return model.beta().detach(), final_loss, final_epoch, status


def run(args: argparse.Namespace) -> pd.DataFrame:
    make_deterministic(args.seed)
    torch.set_default_dtype(torch.float64)

    gen_pt = torch.Generator().manual_seed(args.seed + 1)
    gen_masks = torch.Generator().manual_seed(args.seed + 2)
    gen_ft = torch.Generator().manual_seed(args.seed + 3)
    gen_x_ft = torch.Generator().manual_seed(args.seed + 4)
    gen_noise = torch.Generator().manual_seed(args.seed + 5)

    beta_hat_pt = sample_teacher(args.d, args.pt_active_frac, gen_pt)
    sqrt_k = oracle_sqrt_k(
        beta_hat_pt,
        c_pt=args.c_pt,
        lambda_pt=args.lambda_pt,
        gamma_ft=args.gamma_ft,
    )

    retain, forget = make_masks(args.d, args.retain_frac, args.forget_frac, gen_masks)
    beta_ft = build_ft_teacher(
        beta_hat_pt,
        retain,
        forget,
        retain_scale=args.retain_scale,
        forget_scale=args.forget_scale,
        other_scale=args.other_scale,
        generator=gen_ft,
        other_noise_scale=args.other_noise_scale,
    )
    x_ft = torch.randn(args.n_ft, args.d, generator=gen_x_ft) / math.sqrt(args.d)
    cases = build_cases(
        x_ft,
        beta_ft,
        retain,
        forget,
        signed_c_r=args.signed_c_r,
        signed_c_f=args.signed_c_f,
        positive_c_r=args.positive_c_r,
        positive_c_f=args.positive_c_f,
        generator=gen_noise,
    )

    rows = []
    for index, case in enumerate(cases):
        beta_q, q_constraint_mse = solve_min_q_interpolant(
            x_ft,
            x_ft @ case.beta_eff,
            sqrt_k=sqrt_k,
            max_iter=args.dual_max_iter,
            tolerance_grad=args.dual_tolerance_grad,
        )
        beta_hat, final_loss, final_epoch, status = train_ft(
            x_ft,
            case,
            sqrt_k=sqrt_k,
            seed=args.seed + 1000 * index,
            lr=args.lr,
            epochs=args.epochs,
            pred_tol=args.pred_tol,
        )

        train_pred_mse = prediction_mse(x_ft, beta_hat, case.beta_eff)
        param_mse_to_q = parameter_mse(beta_hat, beta_q)
        rows.append(
            {
                "case": case.name,
                "seed": args.seed,
                "d": args.d,
                "n_ft": args.n_ft,
                "c_pt": args.c_pt,
                "lambda_pt": args.lambda_pt,
                "gamma_ft": args.gamma_ft,
                "sqrt_k_min": float(sqrt_k.min().item()),
                "sqrt_k_max": float(sqrt_k.max().item()),
                "signed_c_r": args.signed_c_r,
                "signed_c_f": args.signed_c_f,
                "positive_c_r": args.positive_c_r,
                "positive_c_f": args.positive_c_f,
                "status": status,
                "epoch": final_epoch,
                "final_loss": final_loss,
                "train_pred_mse_to_eff": train_pred_mse,
                "q_constraint_mse": q_constraint_mse,
                "param_mse_to_min_q": param_mse_to_q,
                "passed": (
                    status == "pred_threshold"
                    and train_pred_mse <= args.pass_pred_tol
                    and q_constraint_mse <= args.pass_pred_tol
                    and param_mse_to_q <= args.param_tol
                ),
            }
        )

    return pd.DataFrame(rows)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=60)
    parser.add_argument("--n_ft", type=int, default=15)
    parser.add_argument("--pt_active_frac", type=float, default=0.5)
    parser.add_argument("--retain_frac", type=float, default=0.35)
    parser.add_argument("--forget_frac", type=float, default=0.35)
    parser.add_argument("--retain_scale", type=float, default=1.0)
    parser.add_argument("--forget_scale", type=float, default=-1.0)
    parser.add_argument("--other_scale", type=float, default=0.0)
    parser.add_argument("--other_noise_scale", type=float, default=0.0)
    parser.add_argument("--c_pt", type=float, default=0.2)
    parser.add_argument("--lambda_pt", type=float, default=0.0)
    parser.add_argument("--gamma_ft", type=float, default=0.1)
    parser.add_argument("--signed_c_r", type=float, default=1.0)
    parser.add_argument("--signed_c_f", type=float, default=0.35)
    parser.add_argument("--positive_c_r", type=float, default=1.0)
    parser.add_argument("--positive_c_f", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=250_000)
    parser.add_argument("--pred_tol", type=float, default=1e-10)
    parser.add_argument("--pass_pred_tol", type=float, default=1e-8)
    parser.add_argument("--param_tol", type=float, default=1e-4)
    parser.add_argument("--dual_max_iter", type=int, default=500)
    parser.add_argument("--dual_tolerance_grad", type=float, default=1e-12)
    parser.add_argument("--csv", type=str, default=None)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    results = run(args)
    print(results.to_string(index=False))
    if args.csv is not None:
        results.to_csv(args.csv, index=False)
        print(f"\nSaved results to {args.csv}")
    failures = results[~results["passed"]]
    if not failures.empty:
        raise SystemExit("At least one oracle PT -> FT implicit-bias check failed.")


if __name__ == "__main__":
    main()
