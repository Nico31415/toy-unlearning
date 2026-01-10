#!/usr/bin/env python3
"""
Diagonal linear network (one hidden layer, diagonal) pretraining experiment
matching the diagonal-network setup used in implicit-bias analyses.

Ground truth: beta_true ~ Bernoulli–Gaussian with Var(beta_true_i)=1:
  beta_i = 0 w.p. (1-rho), else N(0, 1/rho).

Data: noiseless linear regression
  y = X beta_true
with Gaussian design scaled as X_ij ~ N(0, 1/n) so that ||X beta||^2 / n is O(1).

We sweep replica-style measurement ratio:
  beta_replica = p / n   (p = dimension, n = #samples)
Underdetermined corresponds to beta_replica > 1 (since n < p).

Model: one-hidden-layer diagonal linear net via signed-square parameterization:
  beta_hat = a_pos^2 - a_neg^2, with a_pos, a_neg >= 0 enforced by softplus.

Initialization: controlled by (c, lmda) per-coordinate constants:
  a_pos(0)^2 = (c + lmda)/2
  a_neg(0)^2 = (c - lmda)/2
So beta_hat(0) = lmda and a_pos(0)^2 + a_neg(0)^2 = c.
Requires: c >= |lmda| and c > 0.

Training: full-batch SGD (no momentum) on squared loss until interpolation.
Report generalization parameter MSE:
  gen_mse = (1/p) * ||beta_hat - beta_true||^2
and plot 10*log10(gen_mse).
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Determinism helpers
# ----------------------------
def make_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Force deterministic behavior as much as possible
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Single-thread for reproducibility
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# ----------------------------
# BG prior sampling
# ----------------------------
def sample_bg_beta(p: int, rho: float, g: torch.Generator, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    beta_i = 0 w.p. (1-rho), else N(0, 1/rho) so Var(beta_i)=1.
    """
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must be in (0,1)")
    var_nonzero = 1.0 / rho
    std_nonzero = math.sqrt(var_nonzero)

    active = torch.rand(p, generator=g, device=device, dtype=dtype) < rho
    beta = torch.zeros(p, device=device, dtype=dtype)
    k = int(active.sum().item())
    if k > 0:
        beta[active] = std_nonzero * torch.randn(k, generator=g, device=device, dtype=dtype)
    return beta


# ----------------------------
# Data generation
# ----------------------------
def sample_data(n: int, beta_true: torch.Tensor, g: torch.Generator, noise_std: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X_ij ~ N(0, 1/n), y = X beta_true + noise.
    This scaling is the standard one used in many high-dim regression analyses.
    """
    device = beta_true.device
    dtype = beta_true.dtype
    p = beta_true.numel()

    X = torch.randn(n, p, generator=g, device=device, dtype=dtype) / math.sqrt(float(n))
    y = X @ beta_true
    if noise_std > 0:
        y = y + float(noise_std) * torch.randn(n, generator=g, device=device, dtype=dtype)
    return X, y


# ----------------------------
# Softplus init utilities
# ----------------------------
def inv_softplus(y: float) -> float:
    """
    Inverse of softplus for y>0.
    softplus(x) = log(1+exp(x))
    """
    # For small y, exp(y)-1 may underflow; clamp.
    y = float(max(y, 1e-30))
    return float(math.log(math.expm1(y)))


def init_signed_square_from_c_lambda(c: float, lmda: float) -> Tuple[float, float]:
    """
    Returns (a_pos0, a_neg0) such that:
      a_pos0^2 = (c + lmda)/2
      a_neg0^2 = (c - lmda)/2
    Requires: c >= |lmda| and c > 0.
    """
    c = float(c)
    lmda = float(lmda)
    if c <= 0:
        raise ValueError("Require c > 0.")
    if c < abs(lmda):
        raise ValueError("Require c >= |lmda| for real initialization.")
    ap2 = 0.5 * (c + lmda)
    am2 = 0.5 * (c - lmda)
    ap2 = max(ap2, 0.0)
    am2 = max(am2, 0.0)
    return math.sqrt(ap2), math.sqrt(am2)


# ----------------------------
# Model: one-hidden-layer diagonal net
# ----------------------------
class DiagonalSignedSquareNet(nn.Module):
    """
    beta_hat = a_pos^2 - a_neg^2, with a_pos,a_neg >= 0 (enforced via softplus).
    """

    def __init__(self, p: int, c: float, lmda: float, dtype: torch.dtype, device: torch.device):
        super().__init__()
        ap0, am0 = init_signed_square_from_c_lambda(c, lmda)

        # raw params so that softplus(raw)=a0
        raw_ap0 = inv_softplus(ap0)
        raw_am0 = inv_softplus(am0)

        self.raw_ap = nn.Parameter(torch.full((p,), raw_ap0, dtype=dtype, device=device))
        self.raw_am = nn.Parameter(torch.full((p,), raw_am0, dtype=dtype, device=device))

    def a_pos(self) -> torch.Tensor:
        return F.softplus(self.raw_ap)

    def a_neg(self) -> torch.Tensor:
        return F.softplus(self.raw_am)

    def beta(self) -> torch.Tensor:
        ap = self.a_pos()
        am = self.a_neg()
        return ap * ap - am * am

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.beta()


# ----------------------------
# Training
# ----------------------------
def train_to_interpolation(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float,
    max_steps: int,
    threshold: float,
    print_every: int = 0,
) -> Tuple[float, bool, int]:
    """
    Full-batch SGD (no momentum) until MSE < threshold or max_steps.
    Returns: (best_loss, converged, steps_used)
    """
    opt = optim.SGD(model.parameters(), lr=float(lr), momentum=0.0)

    best = float("inf")
    converged = False
    steps_used = 0

    for t in range(int(max_steps)):
        opt.zero_grad(set_to_none=True)
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss_val = float(loss.item())

        if not math.isfinite(loss_val):
            break

        loss.backward()
        opt.step()

        steps_used = t + 1
        if loss_val < best:
            best = loss_val

        if print_every and (t % print_every == 0 or t == max_steps - 1):
            print(f"    step {t:6d}  train_mse={loss_val:.3e}")

        if loss_val <= threshold:
            converged = True
            best = loss_val
            break

    return best, converged, steps_used


# ----------------------------
# Metrics
# ----------------------------
def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a, b).item())


def to_db(x: float) -> float:
    return 10.0 * float(np.log10(max(x, 1e-20)))


# ----------------------------
# Experiment core
# ----------------------------
def run_single(
    *,
    seed: int,
    p: int,
    beta_replica: float,   # p/n
    rho: float,
    c: float,
    lmda: float,
    lr: float,
    max_steps: int,
    threshold: float,
    n_test: int,
    noise_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict:
    make_deterministic(seed)

    # Generators (keep separate streams)
    g_beta = torch.Generator(device="cpu").manual_seed(seed)
    g_tr = torch.Generator(device="cpu").manual_seed(seed + 1)
    g_te = torch.Generator(device="cpu").manual_seed(seed + 2)

    # Compute n from beta_replica = p/n
    beta_replica = float(beta_replica)
    if beta_replica <= 0:
        raise ValueError("beta_replica must be positive")
    n = int(round(p / beta_replica))
    n = max(1, n)

    # Sample ground truth
    beta_true = sample_bg_beta(p, rho, g_beta, device=device, dtype=dtype)

    # Data
    Xtr, ytr = sample_data(n, beta_true, g_tr, noise_std=float(noise_std))
    Xte, yte = sample_data(int(n_test), beta_true, g_te, noise_std=float(noise_std))

    # Model
    model = DiagonalSignedSquareNet(p, c=c, lmda=lmda, dtype=dtype, device=device)

    # Train
    train_mse_best, converged, steps_used = train_to_interpolation(
        model, Xtr, ytr, lr=lr, max_steps=max_steps, threshold=threshold, print_every=0
    )

    with torch.no_grad():
        beta_hat = model.beta().detach()
        gen_mse = mse(beta_hat, beta_true)
        test_mse = float(F.mse_loss(model(Xte), yte).item())

    return {
        "seed": seed,
        "p": p,
        "n_train": n,
        "beta_replica": p / n,           # exact realized p/n after rounding
        "rho": rho,
        "c": c,
        "lmda": lmda,
        "lr": lr,
        "max_steps": max_steps,
        "threshold": threshold,
        "noise_std": noise_std,
        "train_mse": float(train_mse_best),
        "gen_mse": float(gen_mse),
        "test_mse": float(test_mse),
        "converged": bool(converged),
        "steps_used": int(steps_used),
        "beta_true_nnz": int((beta_true != 0).sum().item()),
        "beta_true_l2": float(torch.linalg.norm(beta_true).item()),
    }


def run_sweep(args) -> pd.DataFrame:
    betas = np.linspace(args.beta_min, args.beta_max, args.beta_points).astype(float)

    device = torch.device("cpu")  # keep CPU for determinism
    dtype = torch.float64

    results = []
    total = len(betas) * args.n_seeds

    print("=" * 90)
    print("DIAGONAL NET BG SWEEP (replica-style beta = p/n)")
    print("=" * 90)
    print(f"p={args.p}, rho={args.rho}")
    print(f"beta_replica range: [{betas.min():.3f}, {betas.max():.3f}] with {len(betas)} points")
    print("  underdetermined is beta_replica > 1 (i.e., n < p)")
    print(f"seeds per beta: {args.n_seeds}  (total runs={total})")
    print(f"init: c={args.c:g}, lmda={args.lmda:g}  (requires c >= |lmda|)")
    print(f"train: SGD lr={args.lr:g}, max_steps={args.max_steps}, threshold={args.threshold:g}")
    print(f"data: X_ij ~ N(0,1/n), noise_std={args.noise_std:g}")
    print("=" * 90)

    for bi, beta_rep in enumerate(betas):
        print(f"\nβ_replica = p/n = {beta_rep:.3f}  (n≈{int(round(args.p/beta_rep))})")
        for r in range(args.n_seeds):
            seed = int(args.base_seed + r + 1_000_000 * bi)
            out = run_single(
                seed=seed,
                p=args.p,
                beta_replica=float(beta_rep),
                rho=float(args.rho),
                c=float(args.c),
                lmda=float(args.lmda),
                lr=float(args.lr),
                max_steps=int(args.max_steps),
                threshold=float(args.threshold),
                n_test=int(args.n_test),
                noise_std=float(args.noise_std),
                device=device,
                dtype=dtype,
            )
            results.append(out)
            if not args.verbose:
                print(
                    f"  rep {r:2d}: n={out['n_train']:4d} "
                    f"train={out['train_mse']:.2e}  gen={out['gen_mse']:.2e} ({to_db(out['gen_mse']):6.1f} dB) "
                    f"conv={out['converged']} steps={out['steps_used']}"
                )

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, save_path: str, args) -> None:
    grp = df.groupby("beta_replica")
    summary = grp.agg(
        gen_mse_median=("gen_mse", "median"),
        gen_mse_q25=("gen_mse", lambda x: x.quantile(0.25)),
        gen_mse_q75=("gen_mse", lambda x: x.quantile(0.75)),
        test_mse_median=("test_mse", "median"),
        converged_frac=("converged", "mean"),
        n_train=("n_train", "median"),
    ).reset_index().sort_values("beta_replica")

    summary["gen_mse_db"] = summary["gen_mse_median"].apply(to_db)
    summary["gen_mse_db_q25"] = summary["gen_mse_q25"].apply(to_db)
    summary["gen_mse_db_q75"] = summary["gen_mse_q75"].apply(to_db)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(summary["beta_replica"], summary["gen_mse_db"], "o-", linewidth=2.0, markersize=5, label="median")
    ax.fill_between(summary["beta_replica"], summary["gen_mse_db_q25"], summary["gen_mse_db_q75"], alpha=0.25, label="IQR")

    ax.set_xlabel(r"Replica measurement ratio $\beta = p/n$  (larger = fewer samples)", fontsize=13)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=13)
    ax.set_title(f"Diagonal Net (signed-square) | BG prior | p={args.p}, rho={args.rho}", fontsize=13)
    ax.grid(True, linestyle=":", alpha=0.8)
    ax.legend()

    y_min = float(summary["gen_mse_db_q25"].min()) - 2
    y_max = float(summary["gen_mse_db_q75"].max()) + 2
    ax.set_ylim(y_min, min(y_max, 30))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    # Print table
    print("\nSummary:")
    print("-" * 78)
    print(f"{'beta=p/n':>10} {'n_train':>8} {'gen(dB)':>10} {'conv%':>8} {'test_mse':>12}")
    print("-" * 78)
    for _, row in summary.iterrows():
        print(
            f"{float(row['beta_replica']):>10.3f} {int(round(row['n_train'])):>8d} "
            f"{float(row['gen_mse_db']):>10.2f} {float(row['converged_frac']):>7.0%} "
            f"{float(row['test_mse_median']):>12.3e}"
        )
    print("-" * 78)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagonal net BG sweep (replica beta = p/n)")

    # Problem
    p.add_argument("--p", type=int, default=1024, help="dimension p")
    p.add_argument("--rho", type=float, default=0.1, help="BG sparsity rho in (0,1)")
    p.add_argument("--noise_std", type=float, default=0.0, help="label noise std (0 = noiseless)")

    # Sweep beta_replica = p/n
    p.add_argument("--beta_min", type=float, default=0.5, help="min beta_replica = p/n")
    p.add_argument("--beta_max", type=float, default=3.0, help="max beta_replica = p/n")
    p.add_argument("--beta_points", type=int, default=15, help="# beta points")

    # Repeats
    p.add_argument("--n_seeds", type=int, default=10, help="seeds per beta")
    p.add_argument("--base_seed", type=int, default=0, help="base seed")

    # Init (c, lmda)
    p.add_argument("--c", type=float, default=1e-5, help="init scale c (requires c >= |lmda| and c>0)")
    p.add_argument("--lmda", type=float, default=0.0, help="init predictor value (beta_hat(0)=lmda)")

    # Training
    p.add_argument("--lr", type=float, default=1e-4, help="SGD learning rate (small approximates gradient flow)")
    p.add_argument("--max_steps", type=int, default=300000, help="max SGD steps")
    p.add_argument("--threshold", type=float, default=1e-12, help="train MSE threshold for interpolation")
    p.add_argument("--n_test", type=int, default=20000, help="test set size")

    # Output
    p.add_argument("--save_folder", type=str, default="experiments/diagonal/bg_beta_sweep_fixed",
                   help="folder to save csv and plot")
    p.add_argument("--verbose", action="store_true", help="more prints")

    return p


def main() -> None:
    args = get_parser().parse_args()

    Path(args.save_folder).mkdir(parents=True, exist_ok=True)

    # Sanity checks
    if args.c <= 0:
        raise ValueError("c must be > 0")
    if args.c < abs(args.lmda):
        raise ValueError("Require c >= |lmda|")

    df = run_sweep(args)

    csv_path = os.path.join(args.save_folder, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    plot_path = os.path.join(args.save_folder, "gen_mse_vs_beta_replica.png")
    plot_results(df, plot_path, args)

    print("\nDone.")


if __name__ == "__main__":
    main()