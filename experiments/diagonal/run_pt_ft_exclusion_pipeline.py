"""Sequential PT -> FT -> exclusion-task pipeline for diagonal networks.

This is a toy sample-efficiency setup for unlearning-like behavior:

  1. Pretrain on beta_pt with ordinary MSE.
  2. Fine-tune on beta_ft, whose support has controlled overlap with beta_pt.
  3. Fine-tune again on beta_excl = beta_pt restricted to coordinates active in
     beta_pt but inactive in beta_ft.

The third stage asks how many samples are needed to recover the PT-only
coordinates after the model has moved through the FT task.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from check_effective_teacher_losses import DiagonalNet, make_deterministic, parameter_mse, prediction_mse


def sample_sparse_teacher(
    d: int,
    active_dim: int,
    generator: torch.Generator,
    *,
    scale_to_unit_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if active_dim > d:
        raise ValueError("active_dim must be <= d.")
    support = torch.zeros(d, dtype=torch.bool)
    idx = torch.randperm(d, generator=generator)[:active_dim]
    support[idx] = True

    beta = torch.zeros(d)
    values = torch.randn(active_dim, generator=generator)
    if scale_to_unit_norm and active_dim > 0:
        values = values / torch.linalg.norm(values)
    beta[idx] = values
    return beta, support


def sample_ft_teacher(
    beta_pt: torch.Tensor,
    support_pt: torch.Tensor,
    *,
    active_dim_ft: int,
    overlap_dim: int,
    same_values_on_overlap: bool,
    generator: torch.Generator,
    scale_to_unit_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = beta_pt.shape[0]
    n_pt = int(support_pt.sum().item())
    if overlap_dim > min(active_dim_ft, n_pt):
        raise ValueError("overlap_dim must be <= min(active_dim_ft, active_dim_pt).")

    pt_idx = torch.where(support_pt)[0]
    non_pt_idx = torch.where(~support_pt)[0]
    n_new = active_dim_ft - overlap_dim
    if n_new > len(non_pt_idx):
        raise ValueError("Not enough non-PT coordinates for requested FT support.")

    overlap_idx = pt_idx[torch.randperm(len(pt_idx), generator=generator)[:overlap_dim]]
    new_idx = non_pt_idx[torch.randperm(len(non_pt_idx), generator=generator)[:n_new]]
    ft_idx = torch.cat([overlap_idx, new_idx])

    support_ft = torch.zeros(d, dtype=torch.bool)
    support_ft[ft_idx] = True

    beta_ft = torch.zeros(d)
    if same_values_on_overlap and overlap_dim > 0:
        beta_ft[overlap_idx] = beta_pt[overlap_idx]
        if n_new > 0:
            beta_ft[new_idx] = torch.randn(n_new, generator=generator)
    else:
        beta_ft[ft_idx] = torch.randn(active_dim_ft, generator=generator)

    if scale_to_unit_norm and active_dim_ft > 0:
        norm = torch.linalg.norm(beta_ft)
        if norm > 0:
            beta_ft = beta_ft / norm
    return beta_ft, support_ft


def sample_design(n: int, d: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(n, d, generator=generator) / math.sqrt(d)


def train_to_teacher(
    model: DiagonalNet,
    x: torch.Tensor,
    beta_teacher: torch.Tensor,
    *,
    lr: float,
    epochs: int,
    pred_tol: float,
    optimizer_name: str,
) -> tuple[DiagonalNet, float, int, str]:
    y = x @ beta_teacher
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    final_loss = float("nan")
    final_epoch = 0
    status = "max_epochs"
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0.5 * F.mse_loss(model(x), y)
        if not torch.isfinite(loss):
            final_loss = float(loss.detach().cpu())
            final_epoch = epoch
            status = "nonfinite_loss"
            break
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        final_epoch = epoch
        with torch.no_grad():
            pred_mse = F.mse_loss(model(x), y).item()
        if pred_mse <= pred_tol:
            status = "pred_threshold"
            break
    return model, final_loss, final_epoch, status


def parse_int_list(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def support_mse(beta_a: torch.Tensor, beta_b: torch.Tensor, support: torch.Tensor) -> float:
    if not bool(support.any()):
        return 0.0
    return F.mse_loss(beta_a[support], beta_b[support]).item()


def run(args: argparse.Namespace) -> pd.DataFrame:
    make_deterministic(args.seed)
    torch.set_default_dtype(torch.float64)

    gen_pt = torch.Generator().manual_seed(args.seed + 1)
    gen_ft = torch.Generator().manual_seed(args.seed + 2)
    gen_x_pt = torch.Generator().manual_seed(args.seed + 3)
    gen_x_ft = torch.Generator().manual_seed(args.seed + 4)
    gen_x_test = torch.Generator().manual_seed(args.seed + 5)

    beta_pt, support_pt = sample_sparse_teacher(
        args.d,
        args.active_dim_pt,
        gen_pt,
        scale_to_unit_norm=args.unit_norm_teachers,
    )
    beta_ft, support_ft = sample_ft_teacher(
        beta_pt,
        support_pt,
        active_dim_ft=args.active_dim_ft,
        overlap_dim=args.overlap_dim,
        same_values_on_overlap=args.same_values_on_overlap,
        generator=gen_ft,
        scale_to_unit_norm=args.unit_norm_teachers,
    )

    support_excl = support_pt & ~support_ft
    beta_excl = torch.zeros_like(beta_pt)
    beta_excl[support_excl] = beta_pt[support_excl]

    x_pt = sample_design(args.n_pt, args.d, gen_x_pt)
    x_ft = sample_design(args.n_ft, args.d, gen_x_ft)
    x_test = sample_design(args.n_test, args.d, gen_x_test)

    model0 = DiagonalNet(args.d, scaling=args.init_scaling)
    model_pt, pt_loss, pt_epoch, pt_status = train_to_teacher(
        model0,
        x_pt,
        beta_pt,
        lr=args.pt_lr,
        epochs=args.pt_epochs,
        pred_tol=args.pt_pred_tol,
        optimizer_name=args.optimizer,
    )
    beta_hat_pt = model_pt.beta().detach()

    model_ft, ft_loss, ft_epoch, ft_status = train_to_teacher(
        model_pt,
        x_ft,
        beta_ft,
        lr=args.ft_lr,
        epochs=args.ft_epochs,
        pred_tol=args.ft_pred_tol,
        optimizer_name=args.optimizer,
    )
    beta_hat_ft = model_ft.beta().detach()
    rows = []

    for n_excl in parse_int_list(args.n_excl_values):
        gen_x_excl = torch.Generator().manual_seed(args.seed + 10_000 + n_excl)
        x_excl = sample_design(n_excl, args.d, gen_x_excl)
        model_excl = copy.deepcopy(model_ft)
        model_excl, excl_loss, excl_epoch, excl_status = train_to_teacher(
            model_excl,
            x_excl,
            beta_excl,
            lr=args.excl_lr,
            epochs=args.excl_epochs,
            pred_tol=args.excl_pred_tol,
            optimizer_name=args.optimizer,
        )
        beta_hat_excl = model_excl.beta().detach()

        rows.append(
            {
                "seed": args.seed,
                "d": args.d,
                "active_dim_pt": args.active_dim_pt,
                "active_dim_ft": args.active_dim_ft,
                "overlap_dim": args.overlap_dim,
                "exclusion_dim": int(support_excl.sum().item()),
                "n_pt": args.n_pt,
                "n_ft": args.n_ft,
                "n_excl": n_excl,
                "alpha_excl": n_excl / max(1, int(support_excl.sum().item())),
                "optimizer": args.optimizer,
                "pt_status": pt_status,
                "pt_epoch": pt_epoch,
                "pt_loss": pt_loss,
                "pt_test_mse_to_beta_pt": prediction_mse(x_test, beta_hat_pt, beta_pt),
                "ft_status": ft_status,
                "ft_epoch": ft_epoch,
                "ft_loss": ft_loss,
                "ft_test_mse_to_beta_ft": prediction_mse(x_test, beta_hat_ft, beta_ft),
                "ft_exclusion_support_mse_to_beta_excl": support_mse(beta_hat_ft, beta_excl, support_excl),
                "excl_status": excl_status,
                "excl_epoch": excl_epoch,
                "excl_loss": excl_loss,
                "excl_train_mse_to_beta_excl": prediction_mse(x_excl, beta_hat_excl, beta_excl),
                "excl_test_mse_to_beta_excl": prediction_mse(x_test, beta_hat_excl, beta_excl),
                "excl_param_mse_to_beta_excl": parameter_mse(beta_hat_excl, beta_excl),
                "excl_support_mse_to_beta_excl": support_mse(beta_hat_excl, beta_excl, support_excl),
                "excl_support_mse_before_stage3": support_mse(beta_hat_ft, beta_excl, support_excl),
                "ft_support_mse_after_stage3": support_mse(beta_hat_excl, beta_ft, support_ft),
                "pt_support_mse_after_stage3": support_mse(beta_hat_excl, beta_pt, support_pt),
            }
        )

    return pd.DataFrame(rows)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=200)
    parser.add_argument("--active_dim_pt", type=int, default=40)
    parser.add_argument("--active_dim_ft", type=int, default=30)
    parser.add_argument("--overlap_dim", type=int, default=15)
    parser.add_argument("--n_pt", type=int, default=400)
    parser.add_argument("--n_ft", type=int, default=120)
    parser.add_argument("--n_excl_values", type=str, default="5,10,20,40,80,120")
    parser.add_argument("--n_test", type=int, default=5000)
    parser.add_argument("--init_scaling", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--pt_lr", type=float, default=0.03)
    parser.add_argument("--ft_lr", type=float, default=0.03)
    parser.add_argument("--excl_lr", type=float, default=0.03)
    parser.add_argument("--pt_epochs", type=int, default=50_000)
    parser.add_argument("--ft_epochs", type=int, default=50_000)
    parser.add_argument("--excl_epochs", type=int, default=50_000)
    parser.add_argument("--pt_pred_tol", type=float, default=1e-9)
    parser.add_argument("--ft_pred_tol", type=float, default=1e-9)
    parser.add_argument("--excl_pred_tol", type=float, default=1e-9)
    parser.add_argument("--unit_norm_teachers", action="store_true", default=True)
    parser.add_argument("--same_values_on_overlap", action="store_true", default=True)
    parser.add_argument("--csv", type=str, default=None)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    results = run(args)
    print(results.to_string(index=False))
    if args.csv is not None:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")


if __name__ == "__main__":
    main()
