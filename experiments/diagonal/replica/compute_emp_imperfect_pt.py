"""
Empirical PT+FT with imperfect pretraining — diagonal networks.

Two PT imperfection modes:
  "underdetermined": alpha_pt < 1 (fewer PT samples than coordinates)
  "noisy":           sigma0_pt > 0 (PT labels have additive Gaussian noise)

In both cases, after PT training we extract beta_hat_pt, apply the Cosyne
reinit mapping (construct_infinite_pt_state + apply_finetune_reinit_from_pt_state),
and train the FT network from that initialisation.

Quick-check sweeps at the bottom:
  - alpha_pt=0.2, sigma0_pt=0 (underdetermined)
  - alpha_pt=1.0, sigma0_pt=0.01 (noisy)
  both for alpha_ft in {0.05, 0.1, 0.15}, n_seeds seeds.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

# --- sys.path: add repo root and experiments/diagonal so imports resolve ---
_HERE      = Path(__file__).resolve().parent        # .../experiments/diagonal/replica/
_DIAG_DIR  = _HERE.parent                           # .../experiments/diagonal/
_REPO_ROOT = _HERE.parents[2]                       # .../multi-task2/
for _p in (_REPO_ROOT, _DIAG_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diagonal_network_pretrain_bg import (
    DiagonalNet,
    train,
    make_deterministic,
)
from ptft_empirical_finetune_df import (
    sample_pt_teacher_deterministic,
    sample_ft_teacher_with_overlap,
    construct_infinite_pt_state,
    apply_finetune_reinit_from_pt_state,
    _assign_net_params,
    _alpha_to_n_train,
)


# ---------------------------------------------------------------------------
# Fixed experiment parameters
# ---------------------------------------------------------------------------

RHO_PT       = 0.10
RHO_FT       = 0.10
OMEGA        = 1.0
C_PT         = 1e-3
LAMBDA_PT    = 0.0
GAMMA_REINIT = 0.0
A_PT         = 1.0

INP_DIM   = 1000
N_TEST    = 10_000
LR        = 0.5
EPOCHS    = 5_000_000
THRESHOLD = 1e-4
TEST_EVERY_N_EPOCHS = 2000

# Quick-check grid
ALPHA_FT_LIST = [0.05, 0.1, 0.15]
N_SEEDS       = 5       # can be overridden at bottom


# ---------------------------------------------------------------------------
# Core single-run function
# ---------------------------------------------------------------------------

def run_one(
    *,
    pt_mode: str,           # "underdetermined" | "noisy"
    alpha_ft: float,
    alpha_pt: float = 1.0,  # used when pt_mode="underdetermined"
    sigma0_pt: float = 0.0, # used when pt_mode="noisy"
    seed: int,
    inp_dim: int = INP_DIM,
    n_test: int = N_TEST,
    rho_pt: float = RHO_PT,
    rho_ft: float = RHO_FT,
    omega: float = OMEGA,
    c_pt: float = C_PT,
    lambda_pt: float = LAMBDA_PT,
    gamma_reinit: float = GAMMA_REINIT,
    a_pt: float = A_PT,
    lr: float = LR,
    epochs: int = EPOCHS,
    threshold: float = THRESHOLD,
    test_every_n_epochs: int = TEST_EVERY_N_EPOCHS,
) -> Dict[str, Any]:
    """
    One PT+FT run with imperfect pretraining.

    pt_mode="underdetermined": train PT on n_pt = round(alpha_pt * d) samples
    pt_mode="noisy":           train PT on n_pt = d samples, labels += N(0, sigma0_pt)
    """
    assert pt_mode in ("underdetermined", "noisy"), f"Unknown pt_mode={pt_mode!r}"
    t0 = time.time()

    make_deterministic(seed, use_gpu=False)
    torch.set_default_dtype(torch.float64)

    # Separate generators per random component (match ptft_empirical_finetune_df convention)
    gen_pt_teacher   = torch.Generator(device="cpu").manual_seed(seed + 0)
    gen_ft_teacher   = torch.Generator(device="cpu").manual_seed(seed + 1)
    gen_pt_train_x   = torch.Generator(device="cpu").manual_seed(seed + 2)
    gen_pt_test_x    = torch.Generator(device="cpu").manual_seed(seed + 3)
    gen_ft_train_x   = torch.Generator(device="cpu").manual_seed(seed + 4)
    gen_ft_test_x    = torch.Generator(device="cpu").manual_seed(seed + 5)
    gen_pt_noise     = torch.Generator(device="cpu").manual_seed(seed + 6)

    # --- Sample teachers ---
    beta_pt, support_pt = sample_pt_teacher_deterministic(inp_dim, rho_pt, a_pt, gen_pt_teacher)
    beta_ft, support_ft = sample_ft_teacher_with_overlap(inp_dim, rho_ft, omega, support_pt, gen_ft_teacher)

    n_ft_active = int(support_ft.sum().item())
    n_overlap   = int((support_pt & support_ft).sum().item())
    emp_omega   = n_overlap / max(1, n_ft_active)

    # --- PT phase ---
    if pt_mode == "underdetermined":
        n_train_pt = _alpha_to_n_train(alpha_pt, inp_dim)
    else:  # noisy
        n_train_pt = inp_dim  # fully determined (alpha_pt=1 in theory)

    # PT data (replica scaling: X ~ N(0, 1/d))
    x_pt_train = torch.randn(n_train_pt, inp_dim, generator=gen_pt_train_x) / math.sqrt(inp_dim)
    x_pt_test  = torch.randn(n_test,     inp_dim, generator=gen_pt_test_x)  / math.sqrt(inp_dim)

    y_pt_train = x_pt_train @ beta_pt
    y_pt_test  = x_pt_test  @ beta_pt

    if pt_mode == "noisy" and sigma0_pt > 0.0:
        noise = torch.randn(n_train_pt, generator=gen_pt_noise, dtype=torch.float64) * sigma0_pt
        y_pt_train = y_pt_train + noise

    # Train PT network
    pt_net = DiagonalNet(inp_dim, scaling=1.0, lmda=float(lambda_pt), c=float(c_pt),
                         c_vec=None, init_method="complex")
    pt_df, pt_net, _, pt_stop, pt_epoch = train(
        pt_net,
        (x_pt_train, y_pt_train),
        (x_pt_test,  y_pt_test),
        beta_pt,
        test_every_n_epochs=test_every_n_epochs,
        lr=float(lr),
        epochs=int(epochs),
        lr_tuning=True,
        threshold=float(threshold),
    )

    with torch.no_grad():
        beta_hat_pt = pt_net.beta().detach().cpu()

    pt_param_mse = float(torch.nn.functional.mse_loss(beta_hat_pt, beta_pt).item())

    # --- Cosyne mapping: beta_hat_pt -> FT init ---
    pt_state = construct_infinite_pt_state(
        beta_pt=beta_hat_pt, c_pt=float(c_pt), lambda_pt=float(lambda_pt)
    )
    w_pos0, v_pos0, v_neg0, w_neg0 = apply_finetune_reinit_from_pt_state(
        pt_state=pt_state, gamma_reinit=float(gamma_reinit)
    )

    # --- FT phase ---
    n_train_ft = _alpha_to_n_train(alpha_ft, inp_dim)
    alpha_ft_eff = n_train_ft / inp_dim

    x_ft_train = torch.randn(n_train_ft, inp_dim, generator=gen_ft_train_x) / math.sqrt(inp_dim)
    x_ft_test  = torch.randn(n_test,     inp_dim, generator=gen_ft_test_x)  / math.sqrt(inp_dim)

    y_ft_train = x_ft_train @ beta_ft
    y_ft_test  = x_ft_test  @ beta_ft

    ft_net = DiagonalNet(inp_dim, scaling=1.0, lmda=0.0, c=float(c_pt),
                         c_vec=None, init_method="complex")
    _assign_net_params(ft_net, w_pos0, v_pos0, v_neg0, w_neg0)

    ft_df, ft_net, _, ft_stop, ft_epoch = train(
        ft_net,
        (x_ft_train, y_ft_train),
        (x_ft_test,  y_ft_test),
        beta_ft,
        test_every_n_epochs=test_every_n_epochs,
        lr=float(lr),
        epochs=int(epochs),
        lr_tuning=True,
        threshold=float(threshold),
    )

    ft_test_last = ft_df[ft_df["split"] == "test"].sort_values("epoch").iloc[-1]

    return {
        "pt_mode":       pt_mode,
        "alpha_pt":      float(alpha_pt) if pt_mode == "underdetermined" else 1.0,
        "sigma0_pt":     0.0 if pt_mode == "underdetermined" else float(sigma0_pt),
        "alpha_ft":      float(alpha_ft_eff),
        "alpha_ft_req":  float(alpha_ft),
        "seed":          int(seed),
        "inp_dim":       int(inp_dim),
        "rho_pt":        float(rho_pt),
        "rho_ft":        float(rho_ft),
        "omega":         float(omega),
        "emp_omega":     float(emp_omega),
        "c_pt":          float(c_pt),
        "lambda_pt":     float(lambda_pt),
        "gamma_reinit":  float(gamma_reinit),
        "a_pt":          float(a_pt),
        "n_train_pt":    int(n_train_pt),
        "n_train_ft":    int(n_train_ft),
        "pt_param_mse":  float(pt_param_mse),
        "pt_stop":       pt_stop,
        "pt_epoch":      int(pt_epoch),
        "ft_param_mse":  float(ft_test_last["param_mse"]),
        "ft_pred_mse":   float(ft_test_last["pred_mse"]),
        "ft_stop":       ft_stop,
        "ft_epoch":      int(ft_epoch),
        "wall_s":        float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def sweep(
    *,
    pt_mode: str,
    alpha_pt_or_sigma0_pt: float,  # alpha_pt if underdetermined, sigma0_pt if noisy
    alpha_ft_list: Sequence[float] = ALPHA_FT_LIST,
    n_seeds: int = N_SEEDS,
    **kw,
) -> List[Dict[str, Any]]:
    rows = []
    for alpha_ft in alpha_ft_list:
        for seed in range(n_seeds):
            if pt_mode == "underdetermined":
                kw_run = dict(pt_mode="underdetermined", alpha_ft=alpha_ft,
                              alpha_pt=alpha_pt_or_sigma0_pt, seed=seed, **kw)
            else:
                kw_run = dict(pt_mode="noisy", alpha_ft=alpha_ft,
                              sigma0_pt=alpha_pt_or_sigma0_pt, seed=seed, **kw)
            label = (f"  pt_mode={pt_mode}, val={alpha_pt_or_sigma0_pt}, "
                     f"alpha_ft={alpha_ft:.3f}, seed={seed}")
            print(label, flush=True)
            row = run_one(**kw_run)
            print(f"    ft_param_mse={row['ft_param_mse']:.4e}  "
                  f"pt_param_mse={row['pt_param_mse']:.4e}  "
                  f"wall={row['wall_s']:.1f}s", flush=True)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main: quick sanity-check sweeps
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_rows = []

    # Case 1: underdetermined PT (alpha_pt=0.2)
    print("\n=== Case 1: underdetermined PT, alpha_pt=0.2 ===\n")
    all_rows += sweep(pt_mode="underdetermined", alpha_pt_or_sigma0_pt=0.2)

    # Case 2: noisy PT (sigma0_pt=0.01)
    print("\n=== Case 2: noisy PT, sigma0_pt=0.01 ===\n")
    all_rows += sweep(pt_mode="noisy", alpha_pt_or_sigma0_pt=0.01)

    df = pd.DataFrame(all_rows)
    out_csv = _HERE / "emp_imperfect_pt_quick.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} rows → {out_csv}")
    print(df[["pt_mode", "alpha_pt", "sigma0_pt", "alpha_ft", "seed",
              "pt_param_mse", "ft_param_mse"]].to_string(index=False))
