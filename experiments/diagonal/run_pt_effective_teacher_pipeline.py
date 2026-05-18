"""Single-task PT -> FT pipeline for effective-teacher unlearning losses.

The pipeline is:
  1. Train a diagonal linear network on a pretraining teacher with ordinary MSE.
  2. Build a fine-tuning teacher from the pretrained predictor using diagonal
     projection/edit matrices.
  3. Fine-tune separate copies of the pretrained network with stable
     retain/forget target losses.
  4. Log whether each FT run converges to the corresponding effective teacher.

Samples are rows, so predictions are X @ beta. This is the row-sample version
of the math notation X^T beta.
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F

from check_effective_teacher_losses import (
    DiagonalNet,
    make_deterministic,
    make_masks,
    parameter_mse,
    parse_pair_list,
    prediction_mse,
    sample_teacher,
)


@dataclass(frozen=True)
class FtCase:
    name: str
    beta_eff: torch.Tensor
    beta_forget_target: torch.Tensor
    c_r: float
    c_f: float
    loss_fn: Callable[[torch.Tensor], torch.Tensor]


def train_model(
    model: DiagonalNet,
    x: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    lr: float,
    epochs: int,
    pred_target: torch.Tensor | None,
    pred_tol: float,
    optimizer_name: str,
) -> tuple[DiagonalNet, float, int, str]:
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    final_loss = float("nan")
    status = "max_epochs"
    final_epoch = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred)
        if not torch.isfinite(loss):
            final_loss = float(loss.detach().cpu())
            status = "nonfinite_loss"
            final_epoch = epoch
            break

        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        final_epoch = epoch

        if pred_target is not None:
            with torch.no_grad():
                pred_mse = F.mse_loss(model(x), pred_target).item()
            if pred_mse <= pred_tol:
                status = "pred_threshold"
                break
        elif abs(final_loss) <= pred_tol:
            status = "loss_threshold"
            break

    return model, final_loss, final_epoch, status


def build_finetuning_teacher(
    beta_pt_source: torch.Tensor,
    retain: torch.Tensor,
    forget: torch.Tensor,
    *,
    forget_scale: float,
    new_noise_scale: float,
    generator: torch.Generator,
) -> torch.Tensor:
    unused = 1.0 - retain - forget
    beta_ft = retain * beta_pt_source + forget_scale * forget * beta_pt_source
    if new_noise_scale != 0.0:
        beta_ft = beta_ft + new_noise_scale * unused * torch.randn(beta_pt_source.shape[0], generator=generator)
    return beta_ft


def build_ft_cases(
    x_ft: torch.Tensor,
    beta_ft: torch.Tensor,
    retain: torch.Tensor,
    forget: torch.Tensor,
    *,
    signed_pair: tuple[float, float],
    positive_pair: tuple[float, float],
    generator: torch.Generator,
) -> list[FtCase]:
    signed_c_r, signed_c_f = signed_pair
    if signed_c_r <= signed_c_f:
        raise ValueError("Signed FT loss requires c_r > c_f.")

    positive_c_r, positive_c_f = positive_pair
    if positive_c_r <= 0.0 or positive_c_f <= 0.0:
        raise ValueError("Positive FT losses require positive weights.")

    beta_r = retain * beta_ft
    beta_f = forget * beta_ft
    y_r = x_ft @ beta_r
    y_f = x_ft @ beta_f
    y_zero = torch.zeros(x_ft.shape[0], dtype=x_ft.dtype)

    a_r = torch.ones_like(beta_ft)
    a_f = torch.zeros_like(beta_ft)
    a_r[retain.bool()] = 0.65
    a_r[forget.bool()] = 0.15
    a_f[retain.bool()] = 0.20
    a_f[forget.bool()] = -1.10
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
        FtCase(
            name="ordinary_ft",
            beta_eff=beta_ft,
            beta_forget_target=beta_f,
            c_r=1.0,
            c_f=0.0,
            loss_fn=lambda pred: 0.5 * F.mse_loss(pred, x_ft @ beta_ft),
        ),
        FtCase(
            name="signed_binary_mask",
            beta_eff=beta_eff_signed,
            beta_forget_target=beta_f,
            c_r=signed_c_r,
            c_f=signed_c_f,
            loss_fn=lambda pred: 0.5 * signed_c_r * F.mse_loss(pred, y_r) - 0.5 * signed_c_f * F.mse_loss(pred, y_f),
        ),
        FtCase(
            name="signed_coordinate_transform",
            beta_eff=beta_eff_transform,
            beta_forget_target=beta_af,
            c_r=signed_c_r,
            c_f=signed_c_f,
            loss_fn=lambda pred: 0.5 * signed_c_r * F.mse_loss(pred, y_ar) - 0.5 * signed_c_f * F.mse_loss(pred, y_af),
        ),
        FtCase(
            name="forget_to_zero",
            beta_eff=beta_eff_zero,
            beta_forget_target=torch.zeros_like(beta_ft),
            c_r=positive_c_r,
            c_f=positive_c_f,
            loss_fn=lambda pred: 0.5 * positive_c_r * F.mse_loss(pred, y_r) + 0.5 * positive_c_f * F.mse_loss(pred, y_zero),
        ),
        FtCase(
            name="forget_to_noise",
            beta_eff=beta_eff_noise,
            beta_forget_target=beta_noise,
            c_r=positive_c_r,
            c_f=positive_c_f,
            loss_fn=lambda pred: 0.5 * positive_c_r * F.mse_loss(pred, y_r)
            + 0.5 * positive_c_f * F.mse_loss(pred, y_noise),
        ),
    ]


def apply_ft_init(model: DiagonalNet, *, mode: str, gamma: float) -> None:
    if mode == "copy_pt":
        return
    if mode == "readout_reinit":
        with torch.no_grad():
            model.v_pos.fill_(gamma)
            model.v_neg.fill_(gamma)
        return
    if mode == "zero_beta_readout_reinit":
        with torch.no_grad():
            w_mag = 0.5 * (model.w_pos.abs() + model.w_neg.abs())
            model.w_pos.copy_(w_mag)
            model.w_neg.copy_(w_mag)
            model.v_pos.fill_(gamma)
            model.v_neg.fill_(gamma)
        return
    raise ValueError(f"Unknown FT init mode: {mode}")


def run(args: argparse.Namespace) -> pd.DataFrame:
    make_deterministic(args.seed)
    torch.set_default_dtype(torch.float64)

    gen_pt_teacher = torch.Generator().manual_seed(args.seed + 1)
    gen_x_pt = torch.Generator().manual_seed(args.seed + 2)
    gen_masks = torch.Generator().manual_seed(args.seed + 3)
    gen_ft_teacher = torch.Generator().manual_seed(args.seed + 4)
    gen_x_ft = torch.Generator().manual_seed(args.seed + 5)
    gen_x_test = torch.Generator().manual_seed(args.seed + 6)
    gen_noise = torch.Generator().manual_seed(args.seed + 7)

    beta_pt = sample_teacher(args.d, args.pt_active_frac, gen_pt_teacher)
    x_pt = torch.randn(args.n_pt, args.d, generator=gen_x_pt) / math.sqrt(args.d)
    x_ft = torch.randn(args.n_ft, args.d, generator=gen_x_ft) / math.sqrt(args.d)
    x_test = torch.randn(args.n_test, args.d, generator=gen_x_test) / math.sqrt(args.d)
    y_pt = x_pt @ beta_pt

    pt_model = DiagonalNet(args.d, scaling=args.pt_init_scaling)
    pt_model, pt_loss, pt_epoch, pt_status = train_model(
        pt_model,
        x_pt,
        lambda pred: 0.5 * F.mse_loss(pred, y_pt),
        lr=args.pt_lr,
        epochs=args.pt_epochs,
        pred_target=y_pt,
        pred_tol=args.pt_pred_tol,
        optimizer_name=args.pt_optimizer,
    )
    beta_hat_pt = pt_model.beta().detach()

    retain, forget = make_masks(args.d, args.retain_frac, args.forget_frac, gen_masks)
    beta_ft_source = beta_hat_pt if args.ft_depends_on == "learned_pt" else beta_pt
    beta_ft = build_finetuning_teacher(
        beta_ft_source,
        retain,
        forget,
        forget_scale=args.ft_forget_scale,
        new_noise_scale=args.ft_new_noise_scale,
        generator=gen_ft_teacher,
    )

    signed_pair = parse_pair_list(args.signed_pair)[0]
    positive_pair = parse_pair_list(args.positive_pair)[0]
    cases = build_ft_cases(
        x_ft,
        beta_ft,
        retain,
        forget,
        signed_pair=signed_pair,
        positive_pair=positive_pair,
        generator=gen_noise,
    )

    rows = []
    for index, case in enumerate(cases):
        ft_model = copy.deepcopy(pt_model)
        apply_ft_init(ft_model, mode=args.ft_init, gamma=args.ft_gamma)

        target = x_ft @ case.beta_eff
        ft_model, ft_loss, ft_epoch, ft_status = train_model(
            ft_model,
            x_ft,
            case.loss_fn,
            lr=args.ft_lr,
            epochs=args.ft_epochs,
            pred_target=target,
            pred_tol=args.ft_pred_tol,
            optimizer_name=args.ft_optimizer,
        )
        beta_hat = ft_model.beta().detach()

        rows.append(
            {
                "case": case.name,
                "seed": args.seed,
                "d": args.d,
                "n_pt": args.n_pt,
                "n_ft": args.n_ft,
                "pt_status": pt_status,
                "pt_epoch": pt_epoch,
                "pt_loss": pt_loss,
                "pt_train_pred_mse": prediction_mse(x_pt, beta_hat_pt, beta_pt),
                "pt_test_pred_mse": prediction_mse(x_test, beta_hat_pt, beta_pt),
                "ft_depends_on": args.ft_depends_on,
                "ft_init": args.ft_init,
                "retain_frac": args.retain_frac,
                "forget_frac": args.forget_frac,
                "ft_forget_scale": args.ft_forget_scale,
                "ft_new_noise_scale": args.ft_new_noise_scale,
                "c_r": case.c_r,
                "c_f": case.c_f,
                "ft_status": ft_status,
                "ft_epoch": ft_epoch,
                "ft_loss": ft_loss,
                "train_pred_mse_to_beta_eff": prediction_mse(x_ft, beta_hat, case.beta_eff),
                "test_pred_mse_to_beta_eff": prediction_mse(x_test, beta_hat, case.beta_eff),
                "param_mse_to_beta_eff": parameter_mse(beta_hat, case.beta_eff),
                "retain_param_mse": parameter_mse(retain * beta_hat, retain * beta_ft),
                "forget_param_mse_to_case_target": parameter_mse(forget * beta_hat, forget * case.beta_forget_target),
                "pt_to_ft_teacher_param_mse": parameter_mse(beta_hat_pt, beta_ft),
                "passed": ft_status == "pred_threshold"
                and prediction_mse(x_ft, beta_hat, case.beta_eff) <= args.ft_pass_pred_tol,
            }
        )

    return pd.DataFrame(rows)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--d", type=int, default=80)
    parser.add_argument("--n_pt", type=int, default=240)
    parser.add_argument("--n_ft", type=int, default=80)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--pt_active_frac", type=float, default=0.5)
    parser.add_argument("--retain_frac", type=float, default=0.35)
    parser.add_argument("--forget_frac", type=float, default=0.35)
    parser.add_argument("--ft_forget_scale", type=float, default=-1.0)
    parser.add_argument("--ft_new_noise_scale", type=float, default=0.0)
    parser.add_argument("--ft_depends_on", choices=["learned_pt", "true_pt"], default="learned_pt")
    parser.add_argument("--signed_pair", type=str, default="1.0,0.35")
    parser.add_argument("--positive_pair", type=str, default="1.0,0.75")
    parser.add_argument("--pt_init_scaling", type=float, default=0.1)
    parser.add_argument("--pt_optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--pt_lr", type=float, default=0.03)
    parser.add_argument("--pt_epochs", type=int, default=50_000)
    parser.add_argument("--pt_pred_tol", type=float, default=1e-10)
    parser.add_argument("--ft_init", choices=["copy_pt", "readout_reinit", "zero_beta_readout_reinit"], default="copy_pt")
    parser.add_argument("--ft_gamma", type=float, default=0.1)
    parser.add_argument("--ft_optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--ft_lr", type=float, default=0.03)
    parser.add_argument("--ft_epochs", type=int, default=50_000)
    parser.add_argument("--ft_pred_tol", type=float, default=1e-9)
    parser.add_argument("--ft_pass_pred_tol", type=float, default=1e-7)
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
    failures = results[~results["passed"]]
    if not failures.empty:
        raise SystemExit("At least one PT -> FT case failed to reach its effective teacher.")


if __name__ == "__main__":
    main()
