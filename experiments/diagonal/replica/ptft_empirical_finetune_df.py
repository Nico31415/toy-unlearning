"""
Empirical diagonal-network sweeps -> tidy dataframe (and SLURM-array entrypoint).
Located next to `ptft_replica_qk.py` for symmetry.

Implements TWO settings:

1) setting="single_task"  (pretraining-only)
   - Teacher: Bernoulli–Gaussian PT teacher beta_pt with sparsity rho_pt and scale a_pt.
   - Data: y = X @ beta_pt, with X_{ij} ~ N(0, 1/n) so alpha = n / d.
   - Init: "no prior" homogeneous complex init that maps to beta=0 but encodes (c_pt, lambda_pt):
       DiagonalNet(..., c=c_pt, lmda=lambda_pt, init_method="complex").
   - Train to convergence and record generalization.

2) setting="ptft"  (PT+FT finetuning-only, infinite-PT limit)
   - PT teacher (for overlap + induced k/initialization only): deterministic amplitude a_pt on PT support.
   - Assume infinite PT pretraining, so we take beta_hat = beta_pt exactly.
   - Construct a *PT final* parameter state (w_pos, v_pos, v_neg, w_neg) that satisfies:
       beta = beta_pt
       lambda_i = w_pos_i^2 - v_pos_i^2 = lambda_pt  (homogeneous target)
       c_i      = w_pos_i*w_neg_i + v_pos_i*v_neg_i = c_pt  (homogeneous target)
     (This is not unique; we choose a simple per-coordinate gauge.)
   - Reinitialization step:
       set readout weights (the w's) to gamma_reinit (same in + and - paths),
       set input weights to their average: v_pos = v_neg = 0.5*(v_pos_pt + v_neg_pt),
     so the finetune starts at beta(0)=0.
   - FT task: sample FT teacher beta_ft (BG) with overlap omega relative to PT support.
   - Train ONLY on FT data and record generalization.

Outputs:
  - In-memory: `build_empirical_curves_dataframe(...) -> pd.DataFrame`
  - SLURM: run one task by --array_id and append a master CSV under a file lock.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

# Allow running this file directly from subdirectories (e.g. replica/) while
# keeping absolute imports like `from experiments...` working.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diagonal_network_pretrain_bg import (
        DiagonalNet,
        train,
        make_deterministic,
        get_parameters,
    )


# -------------------------
# Small utilities
# -------------------------

DEFAULT_FT_TEACHER_NORM = "unit_total_var"

def _to_list(x):
    return x if isinstance(x, (list, tuple, np.ndarray)) else [x]


def _alpha_to_n_train(alpha: float, inp_dim: int) -> int:
    return max(1, int(round(float(alpha) * int(inp_dim))))


def _alphas_from_grid(
    *,
    alphas: Optional[Sequence[float]],
    alpha_min: float,
    alpha_max: float,
    n_alpha: int,
) -> np.ndarray:
    if alphas is not None:
        arr = np.asarray(list(alphas), dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("alphas must be a non-empty 1D sequence")
        return arr
    return np.linspace(float(alpha_min), float(alpha_max), int(n_alpha), dtype=float)


def _ptft_feasible(rho_pt: float, rho_ft: float, omega: float, eps: float = 1e-12) -> Tuple[bool, str]:
    rho_pt = float(rho_pt)
    rho_ft = float(rho_ft)
    omega = float(omega)
    if not (0.0 < rho_pt < 1.0):
        return False, f"rho_pt must be in (0,1), got {rho_pt}"
    if not (0.0 < rho_ft < 1.0):
        return False, f"rho_ft must be in (0,1), got {rho_ft}"
    if not (0.0 <= omega <= 1.0):
        return False, f"omega must be in [0,1], got {omega}"
    if omega * rho_ft > rho_pt + eps:
        return False, f"omega*rho_ft > rho_pt ({omega*rho_ft:.6g} > {rho_pt:.6g})"
    if rho_pt + (1.0 - omega) * rho_ft > 1.0 + eps:
        return False, f"rho_pt+(1-omega)*rho_ft > 1 ({rho_pt + (1-omega)*rho_ft:.6g} > 1)"
    return True, "ok"


def _safe_csv_upsert_row(csv_path: Path, row: Dict[str, Any], key_cols: Sequence[str]) -> None:
    """
    Append `row` to `csv_path` under a file lock iff the row's key does not already exist.
    """
    csv_path = Path(csv_path)
    lock_path = Path(str(csv_path) + ".lock")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(60):
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                    except Exception:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()

                if df.empty:
                    pd.DataFrame([row]).to_csv(csv_path, index=False)
                    return

                for k in key_cols:
                    if k not in df.columns:
                        # Back-compat: if old master CSV predates this key, assume default convention.
                        if k == "ft_teacher_norm":
                            df[k] = DEFAULT_FT_TEACHER_NORM
                        else:
                            df[k] = np.nan

                mask = np.ones(len(df), dtype=bool)
                for k in key_cols:
                    col = df[k]
                    if k == "ft_teacher_norm":
                        col = col.fillna(DEFAULT_FT_TEACHER_NORM)
                    mask &= (col.astype(str) == str(row.get(k)))
                if bool(mask.any()):
                    return

                new_df = pd.DataFrame([row])
                all_cols = sorted(set(df.columns.tolist() + new_df.columns.tolist()))
                out = pd.concat([df.reindex(columns=all_cols), new_df.reindex(columns=all_cols)], ignore_index=True)
                out.to_csv(csv_path, index=False)
                return
        except Exception:
            if attempt == 59:
                raise
            time.sleep(0.1 * (2 ** min(attempt, 6)) + random.uniform(0, 0.05))


# -------------------------
# Teacher sampling (single_task: PT BG; ptft: PT deterministic; FT: BG with overlap)
# -------------------------

def sample_pt_teacher_bg(inp_dim: int, rho_pt: float, a_pt: float, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PT teacher (Bernoulli–Gaussian), matching the replica single-task convention:
      - support ~ Bernoulli(rho_pt) i.i.d.
      - nonzeros ~ Normal(0, a_pt^2 / rho_pt) so that E[beta_i^2] = a_pt^2

    Returns (beta_pt, support_pt_mask).
    """
    inp_dim = int(inp_dim)
    rho_pt = float(rho_pt)
    a_pt = float(a_pt)
    if not (0.0 < rho_pt < 1.0):
        raise ValueError(f"rho_pt must be in (0,1), got {rho_pt}")

    support_pt = (torch.rand(inp_dim, generator=generator) < rho_pt)
    beta_pt = torch.zeros(inp_dim, dtype=torch.float64)
    if bool(support_pt.any()):
        std = a_pt / math.sqrt(rho_pt)
        beta_pt[support_pt] = torch.randn(int(support_pt.sum().item()), generator=generator, dtype=torch.float64) * std
    return beta_pt, support_pt


def sample_pt_teacher_deterministic(inp_dim: int, rho_pt: float, a_pt: float, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PT teacher (deterministic amplitude), matching the PTFT oracle assumption:
      - support size = round(rho_pt * d) chosen uniformly without replacement
      - nonzeros are exactly a_pt on support

    Returns (beta_pt, support_pt_mask).
    """
    inp_dim = int(inp_dim)
    rho_pt = float(rho_pt)
    a_pt = float(a_pt)
    if not (0.0 < rho_pt < 1.0):
        raise ValueError(f"rho_pt must be in (0,1), got {rho_pt}")

    n_active = int(round(rho_pt * inp_dim))
    perm = torch.randperm(inp_dim, generator=generator)
    support_pt = torch.zeros(inp_dim, dtype=torch.bool)
    support_pt[perm[:n_active]] = True
    beta_pt = torch.zeros(inp_dim, dtype=torch.float64)
    beta_pt[support_pt] = a_pt
    return beta_pt, support_pt


def sample_ft_teacher_with_overlap(
    inp_dim: int,
    rho_ft: float,
    omega: float,
    support_pt: torch.Tensor,
    generator: torch.Generator,
    ft_teacher_norm: str = DEFAULT_FT_TEACHER_NORM,
    beta_pt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FT teacher: Bernoulli-Gaussian with controlled overlap with PT support.
    Overlap definition: omega = |S_pt ∩ S_ft| / |S_ft|.

    ft_teacher_norm:
      - "unit_total_var":    nonzeros ~ Normal(0, 1/rho_ft)  (DEFAULT)
      - "unit_nonzero_var":  nonzeros ~ Normal(0, 1)
      - "aligned_overlap":   overlap coords get beta_pt value (same sign/magnitude as PT),
                             new-FT coords get Normal(0, 1/rho_ft)
      - "opposite_overlap":  overlap coords get -beta_pt value (flipped sign),
                             new-FT coords get Normal(0, 1/rho_ft)
      - "zero_overlap":      overlap coords are 0 (FT ignores shared features),
                             new-FT coords get Normal(0, 1/rho_ft)

    beta_pt is required for aligned_overlap / opposite_overlap / zero_overlap.
    """
    inp_dim = int(inp_dim)
    rho_ft = float(rho_ft)
    omega = float(omega)

    n_ft_active = int(round(rho_ft * inp_dim))
    n_overlap = int(round(omega * n_ft_active))
    n_new = n_ft_active - n_overlap

    pt_idx = torch.where(support_pt)[0]
    non_pt_idx = torch.where(~support_pt)[0]
    if n_overlap > int(pt_idx.numel()):
        raise ValueError("infeasible overlap: not enough PT coordinates")
    if n_new > int(non_pt_idx.numel()):
        raise ValueError("infeasible overlap: not enough non-PT coordinates")

    overlap_idx = pt_idx[torch.randperm(int(pt_idx.numel()), generator=generator)[:n_overlap]]
    new_idx = non_pt_idx[torch.randperm(int(non_pt_idx.numel()), generator=generator)[:n_new]]

    support_ft = torch.zeros(inp_dim, dtype=torch.bool)
    support_ft[overlap_idx] = True
    support_ft[new_idx] = True

    beta_ft = torch.zeros(inp_dim, dtype=torch.float64)
    ft_teacher_norm = str(ft_teacher_norm)

    n_new_act = int(new_idx.numel())
    sigma_new = 1.0 / math.sqrt(rho_ft)

    if ft_teacher_norm == "unit_total_var":
        k = int(support_ft.sum().item())
        if k > 0:
            beta_ft[support_ft] = torch.randn(k, generator=generator, dtype=torch.float64) / math.sqrt(rho_ft)
    elif ft_teacher_norm == "unit_nonzero_var":
        k = int(support_ft.sum().item())
        if k > 0:
            beta_ft[support_ft] = torch.randn(k, generator=generator, dtype=torch.float64)
    elif ft_teacher_norm in ("aligned_overlap", "opposite_overlap", "zero_overlap"):
        if beta_pt is None:
            raise ValueError(f"ft_teacher_norm={ft_teacher_norm!r} requires beta_pt to be passed")
        # New-FT coords (g=1): always independent normal
        if n_new_act > 0:
            beta_ft[new_idx] = torch.randn(n_new_act, generator=generator, dtype=torch.float64) * sigma_new
        # Overlap coords (g=0): depends on convention
        if int(overlap_idx.numel()) > 0:
            if ft_teacher_norm == "aligned_overlap":
                beta_ft[overlap_idx] = beta_pt[overlap_idx].to(torch.float64)
            elif ft_teacher_norm == "opposite_overlap":
                beta_ft[overlap_idx] = -beta_pt[overlap_idx].to(torch.float64)
            # zero_overlap: leave beta_ft[overlap_idx] = 0 (already initialised)
    else:
        raise ValueError(f"Unknown ft_teacher_norm={ft_teacher_norm!r}")

    return beta_ft, support_ft


# -------------------------
# "Infinite PT" parameter construction + finetune init
# -------------------------

@dataclass(frozen=True)
class PTState:
    w_pos: torch.Tensor
    v_pos: torch.Tensor
    v_neg: torch.Tensor
    w_neg: torch.Tensor


def construct_infinite_pt_state(
    *,
    beta_pt: torch.Tensor,
    c_pt: float,
    lambda_pt: float,
) -> PTState:
    """
    Build one explicit parameterization achieving, for every coordinate i:
      beta_i = beta_pt_i
      w_pos_i^2 - v_pos_i^2 = lambda_pt
      w_neg_i^2 - v_neg_i^2 = lambda_pt
      w_pos_i*w_neg_i + v_pos_i*v_neg_i = c_pt

    This is not unique; we choose a smooth per-coordinate construction:
      - if lambda_pt != 0: use a hyperbolic-parameter solver (acos/asinh) that enforces the symmetric constraint
        w_pos^2 - v_pos^2 = w_neg^2 - v_neg^2 = lambda_pt.
      - if lambda_pt == 0: use a closed-form solution with gauge w=v on both branches.

    This avoids PT training while representing the infinite-sample PT solution in parameters.
    """
    beta_pt = beta_pt.detach().to(dtype=torch.float64, device="cpu")
    c = float(c_pt)
    lam = float(lambda_pt)

    d = int(beta_pt.numel())
    w_pos = torch.empty(d, dtype=torch.float64)
    v_pos = torch.empty(d, dtype=torch.float64)
    v_neg = torch.empty(d, dtype=torch.float64)
    w_neg = torch.empty(d, dtype=torch.float64)

    beta_np = beta_pt.numpy()
    for i in range(d):
        wp, wn, vp, vn = _reconstruct_wv_from_beta_scalar(beta=float(beta_np[i]), lambda_pt=lam, c_pt=c)
        w_pos[i] = wp
        v_pos[i] = vp
        w_neg[i] = wn
        v_neg[i] = vn

    return PTState(w_pos=w_pos, v_pos=v_pos, v_neg=v_neg, w_neg=w_neg)


def _reconstruct_wv_from_beta_scalar(*, beta: float, lambda_pt: float, c_pt: float) -> Tuple[float, float, float, float]:
    """
    Reconstruct (w_plus, w_minus, v_plus, v_minus) for ONE coordinate from (beta, lambda_pt, c_pt).

    Enforced constraints (up to numerical precision):
      beta = v_plus * w_plus - v_minus * w_minus
      w_plus^2 - v_plus^2 = lambda_pt
      w_minus^2 - v_minus^2 = lambda_pt
      w_plus*w_minus + v_plus*v_minus = c_pt

    Notes:
      - Requires c_pt > 0 and c_pt >= |lambda_pt|.
      - Supports lambda_pt == 0 via a closed-form degenerate solution (gauge w=v on each branch).
    """
    b = float(beta)
    lam = float(lambda_pt)
    c = float(c_pt)

    if not (c > 0.0):
        raise ValueError("c_pt must be > 0")
    if c < abs(lam) - 1e-14:
        raise ValueError(f"Require c_pt >= |lambda_pt|. Got c_pt={c:.6g}, |lambda_pt|={abs(lam):.6g}")

    # Degenerate case: lambda_pt == 0.
    # Choose gauge w_plus=v_plus=a >= 0 and w_minus=v_minus=b2 >= 0, then solve:
    #   beta = a^2 - b2^2
    #   c    = 2 a b2
    if lam == 0.0:
        s = 0.5 * (b + math.sqrt(b * b + c * c))  # s = a^2 >= 0 for c>0
        a = math.sqrt(max(s, 1e-300))
        b2 = c / (2.0 * a)
        w_plus, v_plus = a, a
        w_minus, v_minus = b2, b2
        return w_plus, w_minus, v_plus, v_minus

    # Non-degenerate case: use hyperbolic parameterization.
    # For lambda>0:
    #   w = sqrt(lam)*cosh(theta), v = sqrt(lam)*sinh(theta) -> w^2 - v^2 = lam
    # For lambda<0:
    #   v = sqrt(mu)*cosh(theta), w = sqrt(mu)*sinh(theta)  -> w^2 - v^2 = -mu = lam
    if lam > 0.0:
        # Need c/lam >= 1
        A = math.acosh(c / lam)
        B = math.asinh(b / c)
        th_p = 0.5 * (A + B)
        th_m = 0.5 * (A - B)
        r = math.sqrt(lam)
        w_plus = r * math.cosh(th_p)
        v_plus = r * math.sinh(th_p)
        w_minus = r * math.cosh(th_m)
        v_minus = r * math.sinh(th_m)
        return w_plus, w_minus, v_plus, v_minus

    mu = -lam
    A = math.acosh(c / mu)
    B = math.asinh(b / c)
    th_p = 0.5 * (A + B)
    th_m = 0.5 * (A - B)
    r = math.sqrt(mu)
    v_plus = r * math.cosh(th_p)
    w_plus = r * math.sinh(th_p)
    v_minus = r * math.cosh(th_m)
    w_minus = r * math.sinh(th_m)
    return w_plus, w_minus, v_plus, v_minus


def apply_finetune_reinit_from_pt_state(
    *,
    pt_state: PTState,
    gamma_reinit: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finetune re-init rule (PT -> FT) used by the empirical PTFT runner.

    We overwrite parameters so that the FT model starts with beta(0)=0 (no signal),
    while preserving a PT-dependent per-coordinate c that maps to the replica k_d proxy.

    Implementation (per-coordinate, matching the replica mapping's +0.5 * gamma^2 term):
      - w_pos0 = w_neg0 = w_sum, where w_sum = (w_pos_pt + w_neg_pt)
      - v_pos0 = v_neg0 = gamma / sqrt(2)

    Then:
      beta(0) = v_pos0*w_pos0 - v_neg0*w_neg0 = 0
      c_ft(0) = w_pos0*w_neg0 + v_pos0*v_neg0 = w_sum^2 + 0.5 * gamma^2
    """
    gamma = float(gamma_reinit)
    gamma_v = gamma / math.sqrt(2.0)

    # v_avg = 0.5 * (pt_state.v_pos + pt_state.v_neg)
    # w_pos0 = torch.full_like(v_avg, gamma)
    # w_neg0 = torch.full_like(v_avg, gamma)
    # v_pos0 = v_avg.clone()
    # v_neg0 = v_avg.clone()

    w_sum = (pt_state.w_pos + pt_state.w_neg)
    v_pos0 = torch.full_like(w_sum, gamma_v)
    v_neg0 = torch.full_like(w_sum, gamma_v)
    w_pos0 = w_sum.clone()
    w_neg0 = w_sum.clone()
    return w_pos0, v_pos0, v_neg0, w_neg0


def _assign_net_params(net: DiagonalNet, w_pos, v_pos, v_neg, w_neg) -> None:
    with torch.no_grad():
        net.w_pos.copy_(w_pos.to(net.w_pos.dtype))
        net.v_pos.copy_(v_pos.to(net.v_pos.dtype))
        net.v_neg.copy_(v_neg.to(net.v_neg.dtype))
        net.w_neg.copy_(w_neg.to(net.w_neg.dtype))


# -------------------------
# Core runner: one experiment -> one summary row
# -------------------------

def run_one(
    *,
    setting: str,  # "single_task" or "ptft"
    seed: int,
    inp_dim: int,
    alpha: float,
    n_test: int,
    # common init knobs
    c_pt: float,
    lambda_pt: float,
    gamma_reinit: float,
    # PT teacher knobs
    rho_pt: float,
    a_pt: float,
    # FT knobs (only used for ptft)
    rho_ft: float,
    omega: float,
    ft_teacher_norm: str = DEFAULT_FT_TEACHER_NORM,
    # training knobs
    lr: float,
    epochs: int,
    test_every_n_epochs: int,
    log_every_n_epochs: Optional[int] = None,
    no_tuning: bool,
    threshold: float,
    stop_pred_mse: Optional[float],
    stop_beta_rate: float,
    stop_grad_norm: float,
    lr_decay: float,
    lr_decay_interval: int,
    eps_active: float = 1e-6,
    save_folder: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    setting = str(setting)
    if setting not in {"single_task", "ptft"}:
        raise ValueError("setting must be one of {'single_task','ptft'}")

    seed = int(seed)
    inp_dim = int(inp_dim)
    n_test = int(n_test)

    make_deterministic(seed, use_gpu=False)
    torch.set_default_dtype(torch.float64)

    n_train = _alpha_to_n_train(alpha, inp_dim)
    alpha_eff = n_train / inp_dim

    base: Dict[str, Any] = {
        "status": "ok",
        "setting": setting,
        "seed": seed,
        "inp_dim": inp_dim,
        "alpha_requested": float(alpha),
        "n_train": int(n_train),
        "alpha": float(alpha_eff),
        "n_test": n_test,
        "c_pt": float(c_pt),
        "lambda_pt": float(lambda_pt),
        "gamma_reinit": float(gamma_reinit),
        "rho_pt": float(rho_pt),
        "a_pt": float(a_pt),
        "rho_ft": float(rho_ft),
        "omega": float(omega),
        "ft_teacher_norm": str(ft_teacher_norm),
        "save_folder": save_folder,
        "lr": float(lr),
        "epochs": int(epochs),
        "test_every_n_epochs": int(test_every_n_epochs),
        "log_every_n_epochs": None if log_every_n_epochs is None else int(log_every_n_epochs),
        "no_tuning": bool(no_tuning),
        "threshold": float(threshold),
        "stop_pred_mse": None if stop_pred_mse is None else float(stop_pred_mse),
        "stop_beta_rate": float(stop_beta_rate),
        "stop_grad_norm": float(stop_grad_norm),
        "lr_decay": float(lr_decay),
        "lr_decay_interval": int(lr_decay_interval),
    }

    try:
        # Generators
        gen_pt = torch.Generator(device="cpu").manual_seed(seed + 0)
        gen_train_x = torch.Generator(device="cpu").manual_seed(seed + 2 + 10_000 * n_train)
        gen_test_x = torch.Generator(device="cpu").manual_seed(seed + 3)

        # PT teacher differs by setting:
        #  - single_task: BG teacher (to match single-task replica convention)
        #  - ptft: deterministic-amplitude teacher (to match PTFT oracle assumption)
        if setting == "ptft":
            beta_pt, support_pt = sample_pt_teacher_deterministic(inp_dim, rho_pt, a_pt, gen_pt)
        else:
            beta_pt, support_pt = sample_pt_teacher_bg(inp_dim, rho_pt, a_pt, gen_pt)

        # Sample design matrices with RS scaling
        # x_train = torch.randn(n_train, inp_dim, generator=gen_train_x) / math.sqrt(n_train)
        # x_test = torch.randn(n_test, inp_dim, generator=gen_test_x) / math.sqrt(n_test)

        # Replica/theory scaling: X_{ij} ~ N(0, 1/d)
        x_train = torch.randn(n_train, inp_dim, generator=gen_train_x) / math.sqrt(inp_dim)
        x_test  = torch.randn(n_test,  inp_dim, generator=gen_test_x)  / math.sqrt(inp_dim)

        if save_folder is not None:
            Path(save_folder).mkdir(parents=True, exist_ok=True)

        if setting == "single_task":
            # Single-task: learn beta_pt directly
            y_train = x_train @ beta_pt
            y_test = x_test @ beta_pt

            # "No prior": homogeneous complex init encoding (c_pt, lambda_pt) but mapping to beta=0
            if float(c_pt) ** 2 < float(lambda_pt) ** 2:
                return {**base, "status": "skipped_invalid_init", "why": "need c_pt^2 >= lambda_pt^2 for complex init"}

            net = DiagonalNet(inp_dim, scaling=1.0, lmda=float(lambda_pt), c=float(c_pt), c_vec=None, init_method="complex")

            df, net, norm_df, stop_reason, final_epoch = train(
                net,
                (x_train, y_train),
                (x_test, y_test),
                beta_pt,
                test_every_n_epochs=int(test_every_n_epochs),
                log_every_n_epochs=log_every_n_epochs,
                lr=float(lr),
                epochs=int(epochs),
                lr_tuning=(not bool(no_tuning)),
                threshold=float(threshold),
                stop_pred_mse=stop_pred_mse,
                stop_beta_rate=float(stop_beta_rate),
                stop_grad_norm=float(stop_grad_norm),
                lr_decay=float(lr_decay),
                lr_decay_interval=int(lr_decay_interval),
                save_folder=save_folder,
            )

            # Extract final test metrics
            df_test = df[df["split"] == "test"].sort_values("epoch")
            last = df_test.iloc[-1].to_dict()
            with torch.no_grad():
                beta_hat = net.beta().detach().cpu()
            active_frac = float((beta_hat.abs() > float(eps_active)).double().mean().item())

            if save_folder is not None:
                df.to_feather(os.path.join(save_folder, "df.feather"))
                norm_df.to_feather(os.path.join(save_folder, "norm_df.feather"))
                torch.save(beta_pt, os.path.join(save_folder, "beta_pt.pt"))
                torch.save(net.state_dict(), os.path.join(save_folder, "model.pt"))
                with open(os.path.join(save_folder, "config.json"), "w") as f:
                    json.dump({**base, "alpha": float(alpha_eff)}, f, indent=2)

            return {
                **base,
                "stop_reason": stop_reason,
                "final_epoch": int(final_epoch),
                "final_test_pred_mse": float(last["pred_mse"]),
                "final_param_mse": float(last["param_mse"]),
                "final_grad_norm": float(last["grad_norm"]),
                "final_beta_update_rate": float(last["beta_update_rate"]),
                "final_beta_l1": float(last["beta_l1"]),
                "final_beta_l2": float(last["beta_l2"]),
                "final_active_frac": active_frac,
                "wall_s": float(time.time() - t0),
            }

        # ---------------- PT+FT finetune-only ----------------
        feasible, why = _ptft_feasible(rho_pt, rho_ft, omega)
        if not feasible:
            return {**base, "status": "skipped_infeasible", "why": why}

        gen_ft = torch.Generator(device="cpu").manual_seed(seed + 1)
        beta_ft, support_ft = sample_ft_teacher_with_overlap(
            inp_dim, rho_ft, omega, support_pt, gen_ft,
            ft_teacher_norm=ft_teacher_norm, beta_pt=beta_pt,
        )

        y_train = x_train @ beta_ft
        y_test = x_test @ beta_ft

        # 1) Build PT-final parameter state consistent with (beta_pt, c_pt, lambda_pt)
        pt_state = construct_infinite_pt_state(beta_pt=beta_pt, c_pt=float(c_pt), lambda_pt=float(lambda_pt))

        # 2) Apply reinit rule: w's := gamma, v's := average -> beta(0)=0
        w_pos0, v_pos0, v_neg0, w_neg0 = apply_finetune_reinit_from_pt_state(pt_state=pt_state, gamma_reinit=float(gamma_reinit))

        # Create net (init_method doesn't matter; we overwrite parameters immediately)
        net = DiagonalNet(inp_dim, scaling=1.0, lmda=0.0, c=float(c_pt), c_vec=None, init_method="complex")
        _assign_net_params(net, w_pos0, v_pos0, v_neg0, w_neg0)

        # Sanity: beta must start at (approximately) 0
        with torch.no_grad():
            beta0 = net.beta().detach().cpu().numpy()
        beta0_norm = float(np.linalg.norm(beta0))

        df, net, norm_df, stop_reason, final_epoch = train(
            net,
            (x_train, y_train),
            (x_test, y_test),
            beta_ft,
            test_every_n_epochs=int(test_every_n_epochs),
            log_every_n_epochs=log_every_n_epochs,
            lr=float(lr),
            epochs=int(epochs),
            lr_tuning=(not bool(no_tuning)),
            threshold=float(threshold),
            stop_pred_mse=stop_pred_mse,
            stop_beta_rate=float(stop_beta_rate),
            stop_grad_norm=float(stop_grad_norm),
            lr_decay=float(lr_decay),
            lr_decay_interval=int(lr_decay_interval),
            save_folder=save_folder,
        )

        df_test = df[df["split"] == "test"].sort_values("epoch")
        last = df_test.iloc[-1].to_dict()
        with torch.no_grad():
            beta_hat = net.beta().detach().cpu()
        active_frac = float((beta_hat.abs() > float(eps_active)).double().mean().item())

        # Group diagnostics
        ov = int((support_pt & support_ft).sum().item())
        new = int((~support_pt & support_ft).sum().item())
        ptonly = int((support_pt & ~support_ft).sum().item())
        none = int((~support_pt & ~support_ft).sum().item())
        empirical_omega = float(ov / max(1, int(support_ft.sum().item())))

        if save_folder is not None:
            df.to_feather(os.path.join(save_folder, "df.feather"))
            norm_df.to_feather(os.path.join(save_folder, "norm_df.feather"))
            torch.save(beta_pt, os.path.join(save_folder, "beta_pt.pt"))
            torch.save(beta_ft, os.path.join(save_folder, "beta_ft.pt"))
            torch.save(net.state_dict(), os.path.join(save_folder, "model.pt"))
            torch.save(support_pt, os.path.join(save_folder, "support_pt.pt"))
            torch.save(support_ft, os.path.join(save_folder, "support_ft.pt"))
            # Save initial-state diagnostics
            np.save(os.path.join(save_folder, "beta0.npy"), beta0)
            with open(os.path.join(save_folder, "config.json"), "w") as f:
                json.dump({**base, "alpha": float(alpha_eff), "beta0_l2": beta0_norm}, f, indent=2)

        return {
            **base,
            "stop_reason": stop_reason,
            "final_epoch": int(final_epoch),
            "final_test_pred_mse": float(last["pred_mse"]),
            "final_param_mse": float(last["param_mse"]),
            "final_grad_norm": float(last["grad_norm"]),
            "final_beta_update_rate": float(last["beta_update_rate"]),
            "final_beta_l1": float(last["beta_l1"]),
            "final_beta_l2": float(last["beta_l2"]),
            "final_active_frac": active_frac,
            "beta0_l2": beta0_norm,
            "empirical_omega": empirical_omega,
            "n_ov": ov,
            "n_new": new,
            "n_ptonly": ptonly,
            "n_none": none,
            "wall_s": float(time.time() - t0),
        }

    except Exception as e:
        return {**base, "status": "error", "error": repr(e), "wall_s": float(time.time() - t0)}


# -------------------------
# Dataframe builders
# -------------------------

def build_single_task_curves_dataframe(
    *,
    rho_pt: Union[float, List[float]] = 0.10,
    a_pt: float = 1.0,
    c_pt: Union[float, List[float]] = 0.001,
    lambda_pt: Union[float, List[float]] = 0.0,
    # alpha grid
    alphas: Optional[Sequence[float]] = None,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
    n_alpha: int = 12,
    # misc
    inp_dim: int = 1000,
    n_test: int = 10_000,
    seeds: Sequence[int] = (0, 1, 2),
    # training knobs
    lr: float = 0.5,
    epochs: int = 5_000_000,
    test_every_n_epochs: int = 200,
    log_every_n_epochs: Optional[int] = None,
    no_tuning: bool = True,
    threshold: float = 1e-12,
    stop_pred_mse: Optional[float] = None,
    stop_beta_rate: float = 0.0,
    stop_grad_norm: float = 0.0,
    lr_decay: float = 1.0,
    lr_decay_interval: int = 2000,
) -> pd.DataFrame:
    alphas_arr = _alphas_from_grid(alphas=alphas, alpha_min=alpha_min, alpha_max=alpha_max, n_alpha=n_alpha)
    rows: List[Dict[str, Any]] = []
    for rp in _to_list(rho_pt):
        for c in _to_list(c_pt):
            for lam in _to_list(lambda_pt):
                for sd in seeds:
                    for a in alphas_arr:
                        rows.append(
                            run_one(
                                setting="single_task",
                                seed=int(sd),
                                inp_dim=int(inp_dim),
                                alpha=float(a),
                                n_test=int(n_test),
                                c_pt=float(c),
                                lambda_pt=float(lam),
                                gamma_reinit=0.0,
                                rho_pt=float(rp),
                                a_pt=float(a_pt),
                                rho_ft=0.0,
                                omega=0.0,
                                lr=float(lr),
                                epochs=int(epochs),
                                test_every_n_epochs=int(test_every_n_epochs),
                                log_every_n_epochs=log_every_n_epochs,
                                no_tuning=bool(no_tuning),
                                threshold=float(threshold),
                                stop_pred_mse=stop_pred_mse,
                                stop_beta_rate=float(stop_beta_rate),
                                stop_grad_norm=float(stop_grad_norm),
                                lr_decay=float(lr_decay),
                                lr_decay_interval=int(lr_decay_interval),
                                save_folder=None,
                            )
                        )
    return pd.DataFrame(rows)


def build_ptft_finetune_curves_dataframe(
    *,
    rho_pt: Union[float, List[float]] = 0.10,
    rho_ft: Union[float, List[float]] = 0.04,
    omega: Union[float, List[float]] = 1.00,
    ft_teacher_norm: str = DEFAULT_FT_TEACHER_NORM,
    a_pt: float = 1.0,
    c_pt: Union[float, List[float]] = 0.001,
    lambda_pt: Union[float, List[float]] = 0.0,
    gamma_reinit: Union[float, List[float]] = 0.0,
    # alpha grid
    alphas: Optional[Sequence[float]] = None,
    alpha_min: float = 0.05,
    alpha_max: float = 1.0,
    n_alpha: int = 12,
    # misc
    inp_dim: int = 1000,
    n_test: int = 10_000,
    seeds: Sequence[int] = (0, 1, 2),
    # training knobs
    lr: float = 0.5,
    epochs: int = 5_000_000,
    test_every_n_epochs: int = 200,
    log_every_n_epochs: Optional[int] = None,
    no_tuning: bool = True,
    threshold: float = 1e-12,
    stop_pred_mse: Optional[float] = None,
    stop_beta_rate: float = 0.0,
    stop_grad_norm: float = 0.0,
    lr_decay: float = 1.0,
    lr_decay_interval: int = 2000,
) -> pd.DataFrame:
    alphas_arr = _alphas_from_grid(alphas=alphas, alpha_min=alpha_min, alpha_max=alpha_max, n_alpha=n_alpha)
    rows: List[Dict[str, Any]] = []
    for rp in _to_list(rho_pt):
        for rf in _to_list(rho_ft):
            for om in _to_list(omega):
                for c in _to_list(c_pt):
                    for lam in _to_list(lambda_pt):
                        for gam in _to_list(gamma_reinit):
                            for sd in seeds:
                                for a in alphas_arr:
                                    rows.append(
                                        run_one(
                                            setting="ptft",
                                            seed=int(sd),
                                            inp_dim=int(inp_dim),
                                            alpha=float(a),
                                            n_test=int(n_test),
                                            c_pt=float(c),
                                            lambda_pt=float(lam),
                                            gamma_reinit=float(gam),
                                            rho_pt=float(rp),
                                            a_pt=float(a_pt),
                                            rho_ft=float(rf),
                                            omega=float(om),
                                            ft_teacher_norm=str(ft_teacher_norm),
                                            lr=float(lr),
                                            epochs=int(epochs),
                                            test_every_n_epochs=int(test_every_n_epochs),
                                            log_every_n_epochs=log_every_n_epochs,
                                            no_tuning=bool(no_tuning),
                                            threshold=float(threshold),
                                            stop_pred_mse=stop_pred_mse,
                                            stop_beta_rate=float(stop_beta_rate),
                                            stop_grad_norm=float(stop_grad_norm),
                                            lr_decay=float(lr_decay),
                                            lr_decay_interval=int(lr_decay_interval),
                                            save_folder=None,
                                        )
                                    )
    return pd.DataFrame(rows)


# -------------------------
# SLURM-array CLI
# -------------------------

@dataclass(frozen=True)
class Task:
    setting: str  # "single_task" or "ptft"
    seed: int
    alpha: float
    # common
    inp_dim: int
    n_test: int
    rho_pt: float
    a_pt: float
    c_pt: float
    lambda_pt: float
    gamma_reinit: float
    # ptft-only
    rho_ft: float
    omega: float
    ft_teacher_norm: str = DEFAULT_FT_TEACHER_NORM


def build_tasks(
    *,
    setting: str,
    alphas: Sequence[float],
    seeds: Sequence[int],
    inp_dim: int,
    n_test: int,
    rho_pt: Union[float, Sequence[float]],
    a_pt: float,
    c_pt: Union[float, Sequence[float]],
    lambda_pt: Union[float, Sequence[float]],
    gamma_reinit: Union[float, Sequence[float]],
    rho_ft: Union[float, Sequence[float]],
    omega: Union[float, Sequence[float]],
    ft_teacher_norm: str = DEFAULT_FT_TEACHER_NORM,
) -> List[Task]:
    setting = str(setting)
    if setting not in {"single_task", "ptft"}:
        raise ValueError("setting must be one of {'single_task','ptft'}")

    tasks: List[Task] = []
    if setting == "single_task":
        for rp in _to_list(rho_pt):
            for c in _to_list(c_pt):
                for lam in _to_list(lambda_pt):
                    for sd in list(seeds):
                        for a in list(alphas):
                            tasks.append(
                                Task(
                                    setting="single_task",
                                    seed=int(sd),
                                    alpha=float(a),
                                    inp_dim=int(inp_dim),
                                    n_test=int(n_test),
                                    rho_pt=float(rp),
                                    a_pt=float(a_pt),
                                    c_pt=float(c),
                                    lambda_pt=float(lam),
                                    gamma_reinit=0.0,
                                    rho_ft=0.0,
                                    omega=0.0,
                                    ft_teacher_norm=DEFAULT_FT_TEACHER_NORM,
                                )
                            )
        return tasks

    # ptft
    for rp in _to_list(rho_pt):
        for rf in _to_list(rho_ft):
            for om in _to_list(omega):
                for c in _to_list(c_pt):
                    for lam in _to_list(lambda_pt):
                        for gam in _to_list(gamma_reinit):
                            for sd in list(seeds):
                                for a in list(alphas):
                                    tasks.append(
                                        Task(
                                            setting="ptft",
                                            seed=int(sd),
                                            alpha=float(a),
                                            inp_dim=int(inp_dim),
                                            n_test=int(n_test),
                                            rho_pt=float(rp),
                                            a_pt=float(a_pt),
                                            c_pt=float(c),
                                            lambda_pt=float(lam),
                                            gamma_reinit=float(gam),
                                            rho_ft=float(rf),
                                            omega=float(om),
                                            ft_teacher_norm=str(ft_teacher_norm),
                                        )
                                    )
    return tasks


def _task_save_folder(save_root: str, t: Task) -> str:
    n_train = _alpha_to_n_train(t.alpha, t.inp_dim)
    alpha_eff = n_train / t.inp_dim
    if t.setting == "single_task":
        return str(
            Path(save_root)
            / "single_task"
            / f"rho_pt={t.rho_pt:.6g}--c_pt={t.c_pt:.6g}--lam={t.lambda_pt:.6g}--alpha={alpha_eff:.6f}--seed={t.seed}"
        )
    return str(
        Path(save_root)
        / "ptft"
        / (
            f"rho_pt={t.rho_pt:.6g}--rho_ft={t.rho_ft:.6g}--om={t.omega:.6g}"
            f"--c_pt={t.c_pt:.6g}--lam={t.lambda_pt:.6g}--gam={t.gamma_reinit:.6g}"
            f"{'' if str(t.ft_teacher_norm) == DEFAULT_FT_TEACHER_NORM else f'--ftnorm={t.ft_teacher_norm}'}"
            f"--alpha={alpha_eff:.6f}--seed={t.seed}"
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", required=True, choices=["single_task", "ptft"])
    p.add_argument("--array_id", type=int, required=True)
    p.add_argument("--save_root", type=str, default="results/diagonal/replica_like_empirical/")
    p.add_argument("--master_csv", type=str, default="experiment_results_empirical_replica_like.csv")
    p.add_argument("--alphas_json", type=str, default=None, help='JSON list, e.g. "[0.05,0.1,0.2]"')
    p.add_argument("--alpha_min", type=float, default=0.05)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--n_alpha", type=int, default=12)
    p.add_argument("--seeds_json", type=str, default='[0,1,2]')
    p.add_argument("--inp_dim", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=10000)
    p.add_argument("--rho_pt_json", type=str, default='[0.10]')
    p.add_argument("--rho_ft_json", type=str, default='[0.04]')
    p.add_argument("--omega_json", type=str, default='[1.0]')
    p.add_argument(
        "--ft_teacher_norm",
        type=str,
        default=DEFAULT_FT_TEACHER_NORM,
        choices=["unit_total_var", "unit_nonzero_var"],
        help="FT teacher amplitude convention (default preserves existing behavior).",
    )
    p.add_argument("--a_pt", type=float, default=1.0)
    p.add_argument("--c_pt_json", type=str, default='[0.001]')
    p.add_argument("--lambda_pt_json", type=str, default='[0.0]')
    p.add_argument("--gamma_reinit_json", type=str, default='[0.0]')
    # training
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=5_000_000)
    p.add_argument("--test_every_n_epochs", type=int, default=200)
    p.add_argument("--log_every_n_epochs", type=int, default=None)
    p.add_argument("--no_tuning", action="store_true")
    p.add_argument("--threshold", type=float, default=1e-12)
    p.add_argument("--stop_pred_mse", type=float, default=None)
    p.add_argument("--stop_beta_rate", type=float, default=0.0)
    p.add_argument("--stop_grad_norm", type=float, default=0.0)
    p.add_argument("--lr_decay", type=float, default=1.0)
    p.add_argument("--lr_decay_interval", type=int, default=2000)
    args = p.parse_args()

    if args.alphas_json is not None:
        alphas = json.loads(args.alphas_json)
    else:
        alphas = _alphas_from_grid(alphas=None, alpha_min=args.alpha_min, alpha_max=args.alpha_max, n_alpha=args.n_alpha).tolist()

    seeds = json.loads(args.seeds_json)

    tasks = build_tasks(
        setting=args.setting,
        alphas=alphas,
        seeds=seeds,
        inp_dim=args.inp_dim,
        n_test=args.n_test,
        rho_pt=json.loads(args.rho_pt_json),
        a_pt=args.a_pt,
        c_pt=json.loads(args.c_pt_json),
        lambda_pt=json.loads(args.lambda_pt_json),
        gamma_reinit=json.loads(args.gamma_reinit_json),
        rho_ft=json.loads(args.rho_ft_json),
        omega=json.loads(args.omega_json),
        ft_teacher_norm=str(args.ft_teacher_norm),
    )

    if not (0 <= args.array_id < len(tasks)):
        raise SystemExit(f"array_id {args.array_id} out of range [0, {len(tasks)-1}]")
    t = tasks[args.array_id]
    save_folder = _task_save_folder(args.save_root, t)

    row = run_one(
        setting=t.setting,
        seed=t.seed,
        inp_dim=t.inp_dim,
        alpha=t.alpha,
        n_test=t.n_test,
        c_pt=t.c_pt,
        lambda_pt=t.lambda_pt,
        gamma_reinit=t.gamma_reinit,
        rho_pt=t.rho_pt,
        a_pt=t.a_pt,
        rho_ft=t.rho_ft,
        omega=t.omega,
        ft_teacher_norm=str(t.ft_teacher_norm),
        lr=args.lr,
        epochs=args.epochs,
        test_every_n_epochs=args.test_every_n_epochs,
        log_every_n_epochs=args.log_every_n_epochs,
        no_tuning=bool(args.no_tuning),
        threshold=args.threshold,
        stop_pred_mse=args.stop_pred_mse,
        stop_beta_rate=args.stop_beta_rate,
        stop_grad_norm=args.stop_grad_norm,
        lr_decay=args.lr_decay,
        lr_decay_interval=args.lr_decay_interval,
        save_folder=save_folder,
    )

    master_csv = Path(args.master_csv)
    key_cols = [
        "setting",
        "seed",
        "inp_dim",
        "n_train",
        "rho_pt",
        "a_pt",
        "c_pt",
        "lambda_pt",
    ]
    if args.setting == "ptft":
        key_cols += ["rho_ft", "omega", "gamma_reinit", "ft_teacher_norm"]
    _safe_csv_upsert_row(master_csv, row, key_cols=key_cols)

    print(f"Task {args.array_id}/{len(tasks)-1} done. status={row.get('status')}")
    print(f"save_folder={save_folder}")
    print(f"master_csv={master_csv}")


if __name__ == "__main__":
    main()

