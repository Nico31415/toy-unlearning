"""
Empirical runner for imperfect-PT experiments (finite alpha_pt / noisy PT).

Exposes a single `run_one(...)` function that:
  1. Trains a DiagonalNet on PT data (underdetermined or noisy labels).
  2. Applies the reinit rule to get a FT initialisation.
  3. Fine-tunes on FT data.
  4. Returns a summary dict compatible with emp_imperfect_pt_quick.csv.

Used by:
  - compute_emp_imperfect_pt_worker.py
  - compute_emp_imperfect_pt_alpha_pt_sweep_worker.py
  - run_emp_new.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# --------------------------------------------------------------------------
# sys.path setup
# --------------------------------------------------------------------------
_HERE      = Path(__file__).resolve().parent
_DIAG_DIR  = _HERE.parent
_REPO_ROOT = _HERE.parents[2]
for _p in (_REPO_ROOT, _DIAG_DIR, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diagonal_network_pretrain_bg import (
    DiagonalNet,
    train,
    make_deterministic,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _n(alpha: float, d: int) -> int:
    return max(1, int(round(alpha * d)))


def _sample_bg_teacher(d: int, rho: float, a: float, gen: torch.Generator):
    """Bernoulli-Gaussian teacher with E[beta_i^2] = a^2."""
    supp = torch.rand(d, generator=gen) < rho
    beta = torch.zeros(d, dtype=torch.float64)
    k = int(supp.sum().item())
    if k > 0:
        std = a / math.sqrt(rho)
        beta[supp] = torch.randn(k, generator=gen, dtype=torch.float64) * std
    return beta, supp


def _sample_ft_teacher(d: int, rho_ft: float, omega: float,
                       supp_pt: torch.Tensor, gen: torch.Generator):
    """FT teacher with controlled overlap omega = |S_pt ∩ S_ft| / |S_ft|."""
    n_ft  = int(round(rho_ft * d))
    n_ov  = int(round(omega * n_ft))
    n_new = n_ft - n_ov

    pt_idx     = torch.where(supp_pt)[0]
    non_pt_idx = torch.where(~supp_pt)[0]

    ov_idx  = pt_idx[torch.randperm(len(pt_idx),  generator=gen)[:n_ov]]
    new_idx = non_pt_idx[torch.randperm(len(non_pt_idx), generator=gen)[:n_new]]

    supp_ft = torch.zeros(d, dtype=torch.bool)
    supp_ft[ov_idx]  = True
    supp_ft[new_idx] = True

    beta_ft = torch.zeros(d, dtype=torch.float64)
    k = int(supp_ft.sum().item())
    if k > 0:
        beta_ft[supp_ft] = (
            torch.randn(k, generator=gen, dtype=torch.float64) / math.sqrt(rho_ft)
        )
    emp_omega = float(n_ov) / max(1, k)
    return beta_ft, supp_ft, emp_omega


def _reinit_from_pt_net(net: DiagonalNet, gamma_reinit: float):
    """
    Apply the reinit rule after PT training:
      w_pos0 = w_neg0 = w_pos_pt + w_neg_pt
      v_pos0 = v_neg0 = gamma_reinit / sqrt(2)
    => beta(0) = 0, c_ft_i = (w_pos_pt+w_neg_pt)^2 + 0.5*gamma^2
    """
    gamma_v = gamma_reinit / math.sqrt(2.0)
    with torch.no_grad():
        w_sum = net.w_pos + net.w_neg
        net.w_pos.copy_(w_sum)
        net.w_neg.copy_(w_sum)
        net.v_pos.fill_(gamma_v)
        net.v_neg.fill_(gamma_v)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def run_one(
    *,
    pt_mode: str,           # "underdetermined" or "noisy"
    alpha_pt: float,        # n_pt / d
    alpha_ft: float,        # n_ft / d
    sigma0_pt: float = 0.0, # label-noise std for noisy PT
    seed: int = 0,
    inp_dim: int = 1000,
    n_test: int = 10_000,
    # teacher / init knobs
    rho_pt: float = 0.1,
    a_pt:   float = 1.0,
    rho_ft: float = 0.1,
    omega:  float = 1.0,
    c_pt:        float = 1e-3,
    lambda_pt:   float = 0.0,
    gamma_reinit: float = 0.0,
    # training knobs
    lr:            float = 0.5,
    epochs:        int   = 5_000_000,
    threshold:     float = 1e-12,
    stop_pred_mse:   Optional[float] = None,
    stop_grad_norm:  float = 0.0,
    stop_beta_rate:  float = 0.0,
    lr_decay:          float = 1.0,
    lr_decay_interval: int   = 2000,
) -> Dict[str, Any]:
    t0 = time.time()

    d      = int(inp_dim)
    n_test = int(n_test)
    n_pt   = _n(alpha_pt, d)
    n_ft   = _n(alpha_ft, d)

    make_deterministic(seed, use_gpu=False)
    torch.set_default_dtype(torch.float64)

    gen_pt       = torch.Generator().manual_seed(seed)
    gen_ft       = torch.Generator().manual_seed(seed + 1)
    gen_x_pt     = torch.Generator().manual_seed(seed + 2)
    gen_x_ft     = torch.Generator().manual_seed(seed + 3 + 10_000 * n_ft)
    gen_x_test   = torch.Generator().manual_seed(seed + 4)
    gen_noise_pt = torch.Generator().manual_seed(seed + 5)

    # ── Teachers ──────────────────────────────────────────────────────────
    beta_pt, supp_pt = _sample_bg_teacher(d, rho_pt, a_pt, gen_pt)
    beta_ft, supp_ft, emp_omega = _sample_ft_teacher(d, rho_ft, omega, supp_pt, gen_ft)

    # ── Design matrices (X_ij ~ N(0, 1/d)) ────────────────────────────────
    X_pt   = torch.randn(n_pt,   d, generator=gen_x_pt,   dtype=torch.float64) / math.sqrt(d)
    X_ft   = torch.randn(n_ft,   d, generator=gen_x_ft,   dtype=torch.float64) / math.sqrt(d)
    X_test = torch.randn(n_test, d, generator=gen_x_test, dtype=torch.float64) / math.sqrt(d)

    # ── PT labels ─────────────────────────────────────────────────────────
    y_pt = X_pt @ beta_pt
    if pt_mode == "noisy" and float(sigma0_pt) > 0.0:
        y_pt = y_pt + float(sigma0_pt) * torch.randn(
            n_pt, generator=gen_noise_pt, dtype=torch.float64
        )
    elif pt_mode not in ("underdetermined", "noisy"):
        raise ValueError(f"Unknown pt_mode={pt_mode!r}")

    # ── PT training ────────────────────────────────────────────────────────
    net_pt = DiagonalNet(
        d, scaling=1.0, lmda=float(lambda_pt), c=float(c_pt),
        c_vec=None, init_method="complex",
    )
    df_pt, net_pt, _, pt_stop, pt_epoch = train(
        net_pt,
        (X_pt, y_pt),
        (X_test, X_test @ beta_pt),
        beta_pt,
        test_every_n_epochs=500,
        lr=float(lr),
        epochs=int(epochs),
        lr_tuning=True,
        threshold=float(threshold),
        stop_pred_mse=stop_pred_mse,
        stop_beta_rate=float(stop_beta_rate),
        stop_grad_norm=float(stop_grad_norm),
        lr_decay=float(lr_decay),
        lr_decay_interval=int(lr_decay_interval),
    )
    with torch.no_grad():
        beta_hat_pt = net_pt.beta().detach().cpu()
    pt_param_mse = float(((beta_hat_pt - beta_pt) ** 2).mean().item())

    # ── Reinit for FT ──────────────────────────────────────────────────────
    _reinit_from_pt_net(net_pt, gamma_reinit)
    net_ft = net_pt   # weights updated in-place

    # ── FT training ────────────────────────────────────────────────────────
    y_ft   = X_ft   @ beta_ft
    y_test = X_test @ beta_ft

    df_ft, net_ft, _, ft_stop, ft_epoch = train(
        net_ft,
        (X_ft, y_ft),
        (X_test, y_test),
        beta_ft,
        test_every_n_epochs=500,
        lr=float(lr),
        epochs=int(epochs),
        lr_tuning=True,
        threshold=float(threshold),
        stop_pred_mse=stop_pred_mse,
        stop_beta_rate=float(stop_beta_rate),
        stop_grad_norm=float(stop_grad_norm),
        lr_decay=float(lr_decay),
        lr_decay_interval=int(lr_decay_interval),
    )

    df_ft_test = df_ft[df_ft["split"] == "test"].sort_values("epoch")
    last       = df_ft_test.iloc[-1].to_dict()
    ft_param_mse = float(last["param_mse"])
    ft_pred_mse  = float(last["pred_mse"])

    return {
        "pt_mode":      pt_mode,
        "alpha_pt":     float(alpha_pt),
        "sigma0_pt":    float(sigma0_pt),
        "alpha_ft":     float(n_ft / d),
        "alpha_ft_req": float(alpha_ft),
        "seed":         int(seed),
        "inp_dim":      d,
        "rho_pt":       float(rho_pt),
        "rho_ft":       float(rho_ft),
        "omega":        float(omega),
        "emp_omega":    emp_omega,
        "c_pt":         float(c_pt),
        "lambda_pt":    float(lambda_pt),
        "gamma_reinit": float(gamma_reinit),
        "a_pt":         float(a_pt),
        "n_train_pt":   n_pt,
        "n_train_ft":   n_ft,
        "pt_param_mse": pt_param_mse,
        "pt_stop":      pt_stop,
        "pt_epoch":     int(pt_epoch),
        "ft_param_mse": ft_param_mse,
        "ft_pred_mse":  ft_pred_mse,
        "ft_stop":      ft_stop,
        "ft_epoch":     int(ft_epoch),
        "wall_s":       float(time.time() - t0),
    }
