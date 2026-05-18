"""Quick empirical checks for effective-teacher reductions.

This script is intentionally single-task: it trains diagonal linear networks on
synthetic targets and checks that stable loss variants converge to their
corresponding effective teacher.

The math in the notes writes predictions as X^T beta with X in R^{D x N}.
Here samples are rows, so predictions are X @ beta with X in R^{N x D}.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class DiagonalNet(nn.Module):
    def __init__(self, inp_dim: int, scaling: float = 0.1) -> None:
        super().__init__()
        self.w_pos = nn.Parameter(scaling * torch.ones(inp_dim))
        self.v_pos = nn.Parameter(scaling * torch.ones(inp_dim))
        self.v_neg = nn.Parameter(scaling * torch.ones(inp_dim))
        self.w_neg = nn.Parameter(scaling * torch.ones(inp_dim))

    def beta(self) -> torch.Tensor:
        return self.w_pos * self.v_pos - self.w_neg * self.v_neg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.beta()


@dataclass(frozen=True)
class LossCase:
    name: str
    beta_eff: torch.Tensor
    stable: bool
    loss_fn: Callable[[torch.Tensor], torch.Tensor]
    projection_id: str
    retain_frac: float
    forget_frac: float
    c_r: float
    c_f: float


def sample_teacher(d: int, active_frac: float, generator: torch.Generator) -> torch.Tensor:
    mask = torch.rand(d, generator=generator) < active_frac
    beta = torch.zeros(d)
    n_active = int(mask.sum().item())
    if n_active > 0:
        beta[mask] = torch.randn(n_active, generator=generator) / math.sqrt(active_frac)
    return beta


def make_masks(d: int, retain_frac: float, forget_frac: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if retain_frac + forget_frac > 1.0:
        raise ValueError("retain_frac + forget_frac must be <= 1")

    perm = torch.randperm(d, generator=generator)
    n_retain = int(round(retain_frac * d))
    n_forget = int(round(forget_frac * d))
    retain = torch.zeros(d)
    forget = torch.zeros(d)
    retain[perm[:n_retain]] = 1.0
    forget[perm[n_retain : n_retain + n_forget]] = 1.0
    return retain, forget


def parse_pair_list(value: str) -> list[tuple[float, float]]:
    pairs = []
    for raw_pair in value.split(";"):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        left, right = raw_pair.split(",", maxsplit=1)
        pairs.append((float(left), float(right)))
    if not pairs:
        raise ValueError("Expected at least one pair.")
    return pairs


def prediction_mse(x: torch.Tensor, beta_a: torch.Tensor, beta_b: torch.Tensor) -> float:
    return F.mse_loss(x @ beta_a, x @ beta_b).item()


def parameter_mse(beta_a: torch.Tensor, beta_b: torch.Tensor) -> float:
    return F.mse_loss(beta_a, beta_b).item()


def beta_from_dual(x: torch.Tensor, dual: torch.Tensor, sqrt_k: torch.Tensor) -> torch.Tensor:
    return 0.5 * sqrt_k * torch.sinh(2.0 * (x.T @ dual))


def solve_min_q_interpolant(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sqrt_k: torch.Tensor,
    max_iter: int,
    tolerance_grad: float,
) -> tuple[torch.Tensor, float]:
    dual = torch.zeros(x.shape[0], dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [dual],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=tolerance_grad,
        tolerance_change=1e-14,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x.T @ dual
        dual_objective = torch.sum(0.25 * sqrt_k * torch.cosh(2.0 * logits)) - torch.dot(dual, y)
        dual_objective.backward()
        return dual_objective

    optimizer.step(closure)
    with torch.no_grad():
        beta_opt = beta_from_dual(x, dual, sqrt_k).detach()
        constraint_mse = F.mse_loss(x @ beta_opt, y).item()
    return beta_opt, constraint_mse


def train_case(
    x: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    d: int,
    seed: int,
    epochs: int,
    lr: float,
    threshold: float,
    scaling: float,
    optimizer_name: str,
    stop_target: torch.Tensor | None = None,
    stop_pred_tol: float | None = None,
) -> tuple[torch.Tensor, float, int, str]:
    torch.manual_seed(seed)
    model = DiagonalNet(d, scaling=scaling)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    status = "max_epochs"
    final_loss = float("nan")
    final_epoch = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred)
        if not torch.isfinite(loss):
            status = "nonfinite_loss"
            final_loss = float(loss.detach().cpu())
            final_epoch = epoch
            break

        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        final_epoch = epoch

        if stop_target is not None and stop_pred_tol is not None:
            with torch.no_grad():
                pred_mse = F.mse_loss(model(x), stop_target).item()
            if pred_mse < stop_pred_tol:
                status = "pred_threshold"
                break
        elif abs(final_loss) < threshold:
            status = "loss_threshold"
            break

    return model.beta().detach(), final_loss, final_epoch, status


def build_cases(
    x: torch.Tensor,
    beta_ft: torch.Tensor,
    retain: torch.Tensor,
    forget: torch.Tensor,
    generator: torch.Generator,
    *,
    projection_id: str,
    retain_frac: float,
    forget_frac: float,
    signed_pairs: list[tuple[float, float]],
    positive_pairs: list[tuple[float, float]],
    include_unstable: bool,
) -> list[LossCase]:
    beta_r = retain * beta_ft
    beta_f = forget * beta_ft
    y_r = x @ beta_r
    y_f = x @ beta_f

    a_r = torch.ones_like(beta_ft)
    a_f = torch.zeros_like(beta_ft)
    a_r[retain.bool()] = 0.7
    a_f[forget.bool()] = -1.2
    beta_ar = a_r * beta_ft
    beta_af = a_f * beta_ft
    y_ar = x @ beta_ar
    y_af = x @ beta_af

    y_zero = torch.zeros(x.shape[0])

    beta_noise = torch.randn(beta_ft.shape[0], generator=generator)
    y_noise = x @ beta_noise

    cases = []
    for c_r, c_f in signed_pairs:
        if c_r <= c_f:
            raise ValueError(f"Stable signed pairs require c_r > c_f, got {(c_r, c_f)}")
        beta_eff_signed = (c_r * beta_r - c_f * beta_f) / (c_r - c_f)
        beta_eff_transform = (c_r * beta_ar - c_f * beta_af) / (c_r - c_f)
        cases.extend(
            [
                LossCase(
                    name="signed_binary_mask",
                    beta_eff=beta_eff_signed,
                    stable=True,
                    projection_id=projection_id,
                    retain_frac=retain_frac,
                    forget_frac=forget_frac,
                    c_r=c_r,
                    c_f=c_f,
                    loss_fn=lambda pred, c_r=c_r, c_f=c_f: 0.5 * c_r * F.mse_loss(pred, y_r)
                    - 0.5 * c_f * F.mse_loss(pred, y_f),
                ),
                LossCase(
                    name="signed_coordinate_transform",
                    beta_eff=beta_eff_transform,
                    stable=True,
                    projection_id=projection_id,
                    retain_frac=retain_frac,
                    forget_frac=forget_frac,
                    c_r=c_r,
                    c_f=c_f,
                    loss_fn=lambda pred, c_r=c_r, c_f=c_f: 0.5 * c_r * F.mse_loss(pred, y_ar)
                    - 0.5 * c_f * F.mse_loss(pred, y_af),
                ),
            ]
        )

    for c_r, c_f in positive_pairs:
        if c_r <= 0.0 or c_f <= 0.0:
            raise ValueError(f"Positive pairs require c_r,c_f > 0, got {(c_r, c_f)}")
        beta_eff_zero = (c_r / (c_r + c_f)) * beta_r
        beta_eff_noise = (c_r * beta_r + c_f * beta_noise) / (c_r + c_f)
        cases.extend(
            [
                LossCase(
                    name="forget_to_zero",
                    beta_eff=beta_eff_zero,
                    stable=True,
                    projection_id=projection_id,
                    retain_frac=retain_frac,
                    forget_frac=forget_frac,
                    c_r=c_r,
                    c_f=c_f,
                    loss_fn=lambda pred, c_r=c_r, c_f=c_f: 0.5 * c_r * F.mse_loss(pred, y_r)
                    + 0.5 * c_f * F.mse_loss(pred, y_zero),
                ),
                LossCase(
                    name="forget_to_noise",
                    beta_eff=beta_eff_noise,
                    stable=True,
                    projection_id=projection_id,
                    retain_frac=retain_frac,
                    forget_frac=forget_frac,
                    c_r=c_r,
                    c_f=c_f,
                    loss_fn=lambda pred, c_r=c_r, c_f=c_f: 0.5 * c_r * F.mse_loss(pred, y_r)
                    + 0.5 * c_f * F.mse_loss(pred, y_noise),
                ),
            ]
        )

    if include_unstable:
        c_bad_r = 0.25
        c_bad_f = 1.0
        beta_eff_unstable = (c_bad_r * beta_r - c_bad_f * beta_f) / (c_bad_r - c_bad_f)
        cases.append(
            LossCase(
                name="signed_unstable_expected",
                beta_eff=beta_eff_unstable,
                stable=False,
                projection_id=projection_id,
                retain_frac=retain_frac,
                forget_frac=forget_frac,
                c_r=c_bad_r,
                c_f=c_bad_f,
                loss_fn=lambda pred: 0.5 * c_bad_r * F.mse_loss(pred, y_r) - 0.5 * c_bad_f * F.mse_loss(pred, y_f),
            )
        )

    return cases


def run(args: argparse.Namespace) -> pd.DataFrame:
    make_deterministic(args.seed)
    torch.set_default_dtype(torch.float64)

    gen_data = torch.Generator().manual_seed(args.seed)
    gen_teacher = torch.Generator().manual_seed(args.seed + 1)
    gen_noise = torch.Generator().manual_seed(args.seed + 3)

    x = torch.randn(args.n, args.d, generator=gen_data) / math.sqrt(args.d)
    beta_ft = sample_teacher(args.d, args.active_frac, gen_teacher)

    signed_pairs = parse_pair_list(args.signed_pairs)
    positive_pairs = parse_pair_list(args.positive_pairs)
    projection_configs = parse_pair_list(args.projection_configs)

    cases = []
    for projection_index, (retain_frac, forget_frac) in enumerate(projection_configs):
        gen_masks = torch.Generator().manual_seed(args.seed + 2 + projection_index)
        retain, forget = make_masks(args.d, retain_frac, forget_frac, gen_masks)
        cases.extend(
            build_cases(
                x,
                beta_ft,
                retain,
                forget,
                gen_noise,
                projection_id=f"p{projection_index}",
                retain_frac=retain_frac,
                forget_frac=forget_frac,
                signed_pairs=signed_pairs,
                positive_pairs=positive_pairs,
                include_unstable=args.include_unstable,
            )
        )
    rows = []

    for index, case in enumerate(cases):
        target = x @ case.beta_eff
        beta_hat, final_loss, final_epoch, status = train_case(
            x,
            case.loss_fn,
            d=args.d,
            seed=args.seed + 100 * index,
            epochs=args.epochs,
            lr=args.lr,
            threshold=args.threshold,
            scaling=args.scaling,
            optimizer_name=args.optimizer,
            stop_target=target if case.stable else None,
            stop_pred_tol=args.pred_tol * 0.1,
        )

        baseline_loss = lambda pred, target=target: 0.5 * F.mse_loss(pred, target)
        beta_baseline, baseline_loss_value, baseline_epoch, baseline_status = train_case(
            x,
            baseline_loss,
            d=args.d,
            seed=args.seed + 100 * index,
            epochs=args.epochs,
            lr=args.lr,
            threshold=args.threshold,
            scaling=args.scaling,
            optimizer_name=args.optimizer,
            stop_target=target,
            stop_pred_tol=args.pred_tol * 0.1,
        )

        pred_mse_to_eff = prediction_mse(x, beta_hat, case.beta_eff)
        param_mse_to_eff = parameter_mse(beta_hat, case.beta_eff)
        baseline_pred_mse_to_eff = prediction_mse(x, beta_baseline, case.beta_eff)
        pred_mse_vs_baseline = prediction_mse(x, beta_hat, beta_baseline)

        if case.stable:
            passed = pred_mse_to_eff <= args.pred_tol and baseline_pred_mse_to_eff <= args.pred_tol
        else:
            passed = status in {"nonfinite_loss", "max_epochs"} or pred_mse_to_eff > args.pred_tol

        rows.append(
            {
                "case": case.name,
                "stable": case.stable,
                "projection_id": case.projection_id,
                "retain_frac": case.retain_frac,
                "forget_frac": case.forget_frac,
                "c_r": case.c_r,
                "c_f": case.c_f,
                "status": status,
                "epoch": final_epoch,
                "final_loss": final_loss,
                "pred_mse_to_eff": pred_mse_to_eff,
                "param_mse_to_eff": param_mse_to_eff,
                "baseline_status": baseline_status,
                "baseline_epoch": baseline_epoch,
                "baseline_final_loss": baseline_loss_value,
                "baseline_pred_mse_to_eff": baseline_pred_mse_to_eff,
                "pred_mse_vs_baseline": pred_mse_vs_baseline,
                "passed": passed,
            }
        )

    return pd.DataFrame(rows)


def run_implicit_bias_check(args: argparse.Namespace) -> pd.DataFrame:
    make_deterministic(args.seed)
    torch.set_default_dtype(torch.float64)

    gen_data = torch.Generator().manual_seed(args.seed + 10_000)
    gen_teacher = torch.Generator().manual_seed(args.seed + 10_001)
    gen_noise = torch.Generator().manual_seed(args.seed + 10_003)

    x = torch.randn(args.implicit_n, args.d, generator=gen_data) / math.sqrt(args.d)
    beta_ft = sample_teacher(args.d, args.active_frac, gen_teacher)

    signed_pairs = parse_pair_list(args.signed_pairs)
    positive_pairs = parse_pair_list(args.positive_pairs)
    projection_configs = parse_pair_list(args.projection_configs)

    cases = []
    for projection_index, (retain_frac, forget_frac) in enumerate(projection_configs):
        gen_masks = torch.Generator().manual_seed(args.seed + 10_002 + projection_index)
        retain, forget = make_masks(args.d, retain_frac, forget_frac, gen_masks)
        cases.extend(
            build_cases(
                x,
                beta_ft,
                retain,
                forget,
                gen_noise,
                projection_id=f"p{projection_index}",
                retain_frac=retain_frac,
                forget_frac=forget_frac,
                signed_pairs=signed_pairs,
                positive_pairs=positive_pairs,
                include_unstable=False,
            )
        )

    cases = cases[: args.implicit_max_cases]
    sqrt_k = torch.full((args.d,), 4.0 * args.scaling**2)
    rows = []

    for index, case in enumerate(cases):
        y_eff = x @ case.beta_eff
        beta_q, q_constraint_mse = solve_min_q_interpolant(
            x,
            y_eff,
            sqrt_k=sqrt_k,
            max_iter=args.dual_max_iter,
            tolerance_grad=args.dual_tolerance_grad,
        )
        beta_hat, final_loss, final_epoch, status = train_case(
            x,
            case.loss_fn,
            d=args.d,
            seed=args.seed + 20_000 + 100 * index,
            epochs=args.implicit_epochs,
            lr=args.implicit_lr,
            threshold=args.threshold,
            scaling=args.scaling,
            optimizer_name="sgd",
            stop_target=y_eff,
            stop_pred_tol=args.implicit_pred_tol,
        )

        pred_mse_to_eff = prediction_mse(x, beta_hat, case.beta_eff)
        q_pred_mse_to_eff = F.mse_loss(x @ beta_q, y_eff).item()
        param_mse_to_q = parameter_mse(beta_hat, beta_q)

        rows.append(
            {
                "case": case.name,
                "projection_id": case.projection_id,
                "retain_frac": case.retain_frac,
                "forget_frac": case.forget_frac,
                "c_r": case.c_r,
                "c_f": case.c_f,
                "status": status,
                "epoch": final_epoch,
                "final_loss": final_loss,
                "train_pred_mse_to_eff": pred_mse_to_eff,
                "q_constraint_mse": q_constraint_mse,
                "q_pred_mse_to_eff": q_pred_mse_to_eff,
                "param_mse_to_min_q": param_mse_to_q,
                "passed": (
                    pred_mse_to_eff <= args.implicit_pass_pred_tol
                    and q_constraint_mse <= args.implicit_pass_pred_tol
                    and param_mse_to_q <= args.implicit_param_tol
                ),
            }
        )

    return pd.DataFrame(rows)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=40)
    parser.add_argument("--n", type=int, default=240)
    parser.add_argument("--active_frac", type=float, default=0.5)
    parser.add_argument(
        "--projection_configs",
        type=str,
        default="0.4,0.4;0.25,0.5;0.6,0.2",
        help="Semicolon-separated retain,forget projection fractions.",
    )
    parser.add_argument(
        "--signed_pairs",
        type=str,
        default="1.0,0.25;1.0,0.5;2.0,0.75",
        help="Semicolon-separated stable c_r,c_f pairs for signed losses.",
    )
    parser.add_argument(
        "--positive_pairs",
        type=str,
        default="1.0,0.5;0.5,1.5",
        help="Semicolon-separated positive c_r,c_f pairs for target-replacement losses.",
    )
    parser.add_argument("--include_unstable", action="store_true", help="Also run one expected-unstable signed case.")
    parser.add_argument("--scaling", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--threshold", type=float, default=1e-12)
    parser.add_argument("--pred_tol", type=float, default=1e-5)
    parser.add_argument("--csv", type=str, default=None, help="Optional path to save the summary CSV.")
    parser.add_argument("--check_implicit_bias", action="store_true")
    parser.add_argument("--implicit_n", type=int, default=12)
    parser.add_argument("--implicit_max_cases", type=int, default=8)
    parser.add_argument("--implicit_lr", type=float, default=0.05)
    parser.add_argument("--implicit_epochs", type=int, default=200_000)
    parser.add_argument("--implicit_pred_tol", type=float, default=1e-10)
    parser.add_argument("--implicit_pass_pred_tol", type=float, default=1e-8)
    parser.add_argument("--implicit_param_tol", type=float, default=5e-4)
    parser.add_argument("--dual_max_iter", type=int, default=500)
    parser.add_argument("--dual_tolerance_grad", type=float, default=1e-12)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    results = run(args)
    print(results.to_string(index=False))

    stable_failures = results[results["stable"] & ~results["passed"]]
    if args.csv is not None:
        results.to_csv(args.csv, index=False)
        print(f"\nSaved summary to {args.csv}")
    if not stable_failures.empty:
        raise SystemExit("At least one stable effective-teacher check failed.")

    if args.check_implicit_bias:
        implicit_results = run_implicit_bias_check(args)
        print("\nImplicit-bias checks:")
        print(implicit_results.to_string(index=False))
        implicit_failures = implicit_results[~implicit_results["passed"]]
        if not implicit_failures.empty:
            raise SystemExit("At least one implicit-bias check failed.")


if __name__ == "__main__":
    main()
