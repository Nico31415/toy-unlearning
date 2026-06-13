#!/usr/bin/env python3
"""
Stage 2 unlearning sweep — SLURM worker script.

Each invocation handles one (seed, c_PT, lambda_frac, loss_config, alpha_2)
combination, identified by a zero-based task_id.

Usage (local test):
    python run_unlearning_sweep.py --task_id 0 --dry_run

Usage (SLURM, via submit script):
    sbatch run_unlearning_sweep.sh

Sweep axes  (total 3×2×3×3×5 = 270 tasks)
------------------------------------------
  seed          : [0, 1, 2]
  c_PT          : [1e-3, 1e-1]
  lambda_frac   : [-0.95, 0.0, 0.95]   (lambda_PT = frac * c_PT)
  loss_config   : retain_only | forget_balanced | forget_strong
  alpha_2       : [0.1, 0.2, 0.3, 0.4, 0.5]

Stage-1 uses alpha_PT=50 (oracle-like large-sample pretraining).
Stage-2 models are saved for later Stage-3 relearning experiments.

Save layout
-----------
  <out_dir>/seed={seed}/cpt={cpt}_lam={lam}/{loss_name}/alpha={alpha}/
    config.json
    beta_pt.pt
    mask_forget.pt  mask_retain.pt
    weights_stage1.pt  weights_stage2.pt
    beta_unlearn.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from functions.unlearning import (
    DiagWeights,
    make_init_weights,
    compute_effective_teacher,
    _train_to_convergence,
    run_stage2,
)

# ---------------------------------------------------------------------------
# Sweep definition — single source of truth
# ---------------------------------------------------------------------------

SEEDS           = [0, 1, 2]
C_PT_LIST       = [1e-3, 1e-1]
LAMBDA_FRACS    = [-0.95, 0.0, 0.95]   # lambda_PT = frac * c_PT
LOSS_CONFIGS    = [
    dict(name="retain_only",     loss_type="retain_only",    c_r=1.0, c_f=0.0),
    dict(name="forget_balanced", loss_type="forget_to_zero", c_r=1.0, c_f=1.0),
    dict(name="forget_strong",   loss_type="forget_to_zero", c_r=1.0, c_f=4.0),
]
ALPHA_2_LIST    = [0.1, 0.2, 0.3, 0.4, 0.5]

# Build ordered list of all (seed, c_PT, lam_frac, loss_idx, alpha_2) combos
TASKS = list(itertools.product(
    SEEDS, C_PT_LIST, LAMBDA_FRACS, range(len(LOSS_CONFIGS)), ALPHA_2_LIST
))
N_TASKS = len(TASKS)   # 270


def get_task(task_id: int) -> dict:
    seed, c_PT, lam_frac, loss_idx, alpha_2 = TASKS[task_id]
    return dict(
        seed=seed,
        c_PT=c_PT,
        lambda_PT=lam_frac * c_PT,
        lam_frac=lam_frac,
        loss=LOSS_CONFIGS[loss_idx],
        alpha_2=alpha_2,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_bg_teacher(D: int, rho: float, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    mask = torch.rand(D, generator=rng) < rho
    vals = torch.randn(D, generator=rng) / math.sqrt(rho)
    return mask.float() * vals.float()


def make_forget_retain_masks(beta_pt: torch.Tensor, forget_fraction: float, seed: int):
    active_idx = torch.where(beta_pt != 0)[0]
    n_active = len(active_idx)
    n_forget = max(1, int(math.floor(forget_fraction * n_active)))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_active, generator=rng)
    forget_idx = active_idx[perm[:n_forget]]
    retain_idx = active_idx[perm[n_forget:]]
    mask_forget = torch.zeros(len(beta_pt), dtype=torch.bool)
    mask_retain = torch.zeros(len(beta_pt), dtype=torch.bool)
    mask_forget[forget_idx] = True
    mask_retain[retain_idx] = True
    return mask_forget, mask_retain


def sample_data(N: int, D: int, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(N, D, generator=rng) / math.sqrt(D)


def weights_to_dict(w: DiagWeights) -> dict:
    return {k: v.cpu() for k, v in w._asdict().items()}


def fmt_lam(lam_frac: float, c_PT: float) -> str:
    v = lam_frac * c_PT
    return f"{v:+.4e}".replace("+", "p").replace("-", "m")


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(
    task_id: int,
    D: int = 2000,
    rho_PT: float = 0.1,
    forget_fraction: float = 0.1,
    alpha_PT: float = 50.0,
    lr: float = 0.5,
    epochs: int = 5_000_000,
    threshold: float = 1e-10,
    out_dir: str = "results/unlearning_sweep",
    dry_run: bool = False,
) -> None:
    torch.set_default_dtype(torch.float32)

    cfg = get_task(task_id)
    seed      = cfg["seed"]
    c_PT      = cfg["c_PT"]
    lambda_PT = cfg["lambda_PT"]
    lam_frac  = cfg["lam_frac"]
    lc        = cfg["loss"]
    alpha_2   = cfg["alpha_2"]

    cpt_str  = f"{c_PT:.0e}"
    lam_str  = fmt_lam(lam_frac, c_PT)
    pt_tag   = f"cpt={cpt_str}_lam={lam_str}"

    save_dir = (
        Path(__file__).parent / out_dir
        / f"seed={seed}"
        / pt_tag
        / lc["name"]
        / f"alpha={alpha_2}"
    )

    print(f"[task {task_id}/{N_TASKS-1}] seed={seed} {pt_tag} "
          f"loss={lc['name']} alpha_2={alpha_2}", flush=True)

    if (save_dir / "config.json").exists():
        print("  already done, skipping.", flush=True)
        return

    if dry_run:
        print("  dry_run: would save to", save_dir)
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    N_PT = int(round(alpha_PT * D))
    N_2  = max(1, int(round(alpha_2 * D)))

    # -- Stage 1: large-alpha pretraining (oracle-like) --------------------
    beta_PT = sample_bg_teacher(D, rho_PT, seed=seed)
    mask_forget, mask_retain = make_forget_retain_masks(
        beta_PT, forget_fraction, seed=seed + 1000
    )

    X_PT = sample_data(N_PT, D, seed=seed + 100)
    y_PT = X_PT @ beta_PT

    weights_init_s1 = make_init_weights(D, c=c_PT, lmda=lambda_PT)
    weights_s1, _ = _train_to_convergence(
        weights_init=weights_init_s1,
        X=X_PT, y=y_PT,
        lr=lr, epochs=epochs, threshold=threshold,
    )

    beta_s1  = weights_s1.w_pos * weights_s1.v_pos - weights_s1.w_neg * weights_s1.v_neg
    pt_err   = F.mse_loss(beta_s1, beta_PT).item()
    print(f"  Stage-1 MSE(β̂,β_PT)={pt_err:.2e}", flush=True)

    # -- Stage 2: unlearning -----------------------------------------------
    beta_r   = beta_PT * mask_retain.float()
    beta_f   = beta_PT * mask_forget.float()
    beta_eff = compute_effective_teacher(
        loss_type=lc["loss_type"],
        beta_r=beta_r, beta_f=beta_f,
        c_r=lc["c_r"], c_f=lc["c_f"],
    )

    X_2 = sample_data(N_2, D, seed=seed + 200 + int(alpha_2 * 100))

    weights_s2, beta_ul, _ = run_stage2(
        weights_prev=weights_s1,
        X=X_2,
        beta_eff=beta_eff,
        reset_config={"mode": "A", "gamma": 0.0},
        lr=lr, epochs=epochs, threshold=threshold,
    )

    ul_err_F = F.mse_loss(beta_ul[mask_forget], beta_PT[mask_forget]).item()
    ul_err_R = F.mse_loss(beta_ul[mask_retain], beta_PT[mask_retain]).item()
    print(f"  errF={ul_err_F:.3f}  errR={ul_err_R:.3f}", flush=True)

    # -- Save --------------------------------------------------------------
    torch.save(beta_PT,                     save_dir / "beta_pt.pt")
    torch.save(mask_forget,                 save_dir / "mask_forget.pt")
    torch.save(mask_retain,                 save_dir / "mask_retain.pt")
    torch.save(weights_to_dict(weights_s1), save_dir / "weights_stage1.pt")
    torch.save(weights_to_dict(weights_s2), save_dir / "weights_stage2.pt")
    torch.save(beta_ul,                     save_dir / "beta_unlearn.pt")

    config = dict(
        task_id=task_id,
        seed=seed, D=D, rho_PT=rho_PT,
        forget_fraction=forget_fraction,
        alpha_PT=alpha_PT, N_PT=N_PT,
        c_PT=c_PT, lambda_PT=lambda_PT,
        loss_name=lc["name"], loss_type=lc["loss_type"],
        c_r=lc["c_r"], c_f=lc["c_f"],
        alpha_2=alpha_2, N_2=N_2,
        n_forget=int(mask_forget.sum().item()),
        n_retain=int(mask_retain.sum().item()),
        ul_err_F=ul_err_F, ul_err_R=ul_err_R,
    )
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  saved → {save_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True,
                        help=f"SLURM array task id, 0-{N_TASKS-1}")
    parser.add_argument("--D",              type=int,   default=2000)
    parser.add_argument("--rho_PT",         type=float, default=0.1)
    parser.add_argument("--forget_fraction",type=float, default=0.1)
    parser.add_argument("--alpha_PT",       type=float, default=50.0)
    parser.add_argument("--lr",             type=float, default=0.5)
    parser.add_argument("--epochs",         type=int,   default=5_000_000)
    parser.add_argument("--threshold",      type=float, default=1e-10)
    parser.add_argument("--out_dir",        type=str,
                        default="results/unlearning_sweep")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.task_id < N_TASKS:
        parser.error(f"task_id must be in [0, {N_TASKS-1}]")

    run_task(
        task_id=args.task_id,
        D=args.D,
        rho_PT=args.rho_PT,
        forget_fraction=args.forget_fraction,
        alpha_PT=args.alpha_PT,
        lr=args.lr,
        epochs=args.epochs,
        threshold=args.threshold,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
