#!/usr/bin/env python3
"""
Custom diagonal-network experiment driver (SLURM-array friendly).
One SLURM task = one parameter-config × (all alphas) × (all seeds).

This script is primarily used for the PT+FT *oracle* diagonal-network runs:
  - experiments/diagonal/diagonal_ptft_oracle.py

The config grid below is intentionally curated for throughput (not a full Cartesian
product of every imaginable knob). Update `_make_configs()` to change what is run.

Alpha grid:
  alpha ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5}
Seeds:
  seed ∈ {0, 1, 2}

Output strategy:
  - Each (config, alpha, seed, optimizer condition) writes into a unique save_folder.
  - A single master CSV is appended-to safely under a file lock.
  - Idempotent: reruns do NOT overwrite; existing runs are skipped; existing master rows are not duplicated.

NOTE ON FEASIBILITY (PT+FT oracle):
The PT+FT oracle teacher sampler enforces feasibility via available coordinates:
  omega * rho_ft <= rho_pt
  rho_pt + (1-omega) * rho_ft <= 1
If infeasible, we record a row with status="skipped_infeasible" and do not run training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fcntl
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Import the two experiment entrypoints (call their main() directly)
#
# IMPORTANT:
# This repo does not treat `experiments/` as a Python package (no experiments/__init__.py),
# so we load the two scripts by file path rather than `import experiments.diagonal...`.
# -----------------------------------------------------------------------------
import sys
import importlib.util


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create module spec for {name} at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


bg = _load_module_from_path("diagonal_network_pretrain_bg", THIS_DIR / "diagonal_network_pretrain_bg.py")
ptft = _load_module_from_path("diagonal_ptft_oracle", THIS_DIR / "diagonal_ptft_oracle.py")


# ----------------------------
# Fixed sweep knobs (requested)
# ----------------------------

ALPHA_VALUES: List[float] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS: List[int] = [0, 1, 2]

INP_DIM_BG = 1000
INP_DIM_PTFT = 1000
N_TEST = 10_000

LR = 0.5
EPOCHS_BG = 5_000_000
EPOCHS_PTFT = 5_000_000
THRESHOLD = 1e-12
TEST_EVERY_N_EPOCHS = 200
NO_TUNING = True


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _fmt_float(x: float) -> str:
    return f"{float(x):.6g}".replace("+", "")


def _ptft_feasible(rho_pt: float, rho_ft: float, omega: float, eps: float = 1e-12) -> Tuple[bool, str]:
    """
    Feasibility constraints for the PT+FT oracle overlap sampler:
      omega * rho_ft <= rho_pt
      rho_pt + (1-omega) * rho_ft <= 1
    """
    rho_pt = float(rho_pt)
    rho_ft = float(rho_ft)
    omega = float(omega)

    if not (0.0 < rho_pt < 1.0):
        return False, f"rho_pt must be in (0,1), got {rho_pt}"
    if not (0.0 < rho_ft < 1.0):
        return False, f"rho_ft must be in (0,1), got {rho_ft}"
    if not (0.0 <= omega <= 1.0):
        return False, f"omega must be in [0,1], got {omega}"

    lhs1 = omega * rho_ft
    if lhs1 > rho_pt + eps:
        return False, f"omega*rho_ft > rho_pt ({lhs1:.6g} > {rho_pt:.6g})"

    lhs2 = rho_pt + (1.0 - omega) * rho_ft
    if lhs2 > 1.0 + eps:
        return False, f"rho_pt+(1-omega)*rho_ft > 1 ({lhs2:.6g} > 1)"

    return True, "ok"


def _alpha_to_n_train(alpha: float, inp_dim: int) -> int:
    # Match other scripts: use rounding then derive alpha_eff from n_train / inp_dim
    return max(1, int(round(float(alpha) * int(inp_dim))))


def _safe_csv_upsert_row(csv_path: Path, row: Dict[str, Any], key_cols: List[str]) -> None:
    """
    Append row to csv_path under a file lock iff the row's key does not already exist.
    This avoids duplicates on reruns without overwriting prior results.
    """
    csv_path = Path(csv_path)
    lock_path = Path(str(csv_path) + ".lock")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Exponential-backoff lock acquisition (robust under bursts)
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

                # Empty file: just write
                if df.empty:
                    pd.DataFrame([row]).to_csv(csv_path, index=False)
                    return

                # Ensure key columns exist before checking
                for k in key_cols:
                    if k not in df.columns:
                        df[k] = np.nan

                # Check if key already exists
                mask = np.ones(len(df), dtype=bool)
                for k in key_cols:
                    mask &= (df[k].astype(str) == str(row.get(k)))
                if bool(mask.any()):
                    return  # already present

                # Append (align columns)
                new_df = pd.DataFrame([row])
                all_cols = sorted(set(df.columns.tolist() + new_df.columns.tolist()))
                df = df.reindex(columns=all_cols)
                new_df = new_df.reindex(columns=all_cols)
                out = pd.concat([df, new_df], ignore_index=True)
                out.to_csv(csv_path, index=False)
                return
        except Exception:
            # Brief jittered backoff then retry
            if attempt == 59:
                raise
            time.sleep(0.1 * (2 ** min(attempt, 6)) + random.uniform(0, 0.05))


@dataclass(frozen=True)
class Config:
    exp: str  # "bg" or "ptft_oracle"
    params: Tuple[Tuple[str, Any], ...]  # sorted items for hashing + stable repr

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for (k, v) in self.params}


def _make_configs() -> List[Config]:
    """
    Build the de-duplicated config list matching the user's requested sets,
    but WITHOUT filtering infeasible configs (we will mark them as skipped at runtime).
    """
    cfg_set = set()
    cfgs: List[Config] = []

    def add(exp: str, **kwargs: Any) -> None:
        items = tuple(sorted(kwargs.items(), key=lambda kv: kv[0]))
        c = Config(exp=exp, params=items)
        if c not in cfg_set:
            cfg_set.add(c)
            cfgs.append(c)

    # ---- PT+FT oracle grid (user-requested) ----
    # Goal: fast-ish grid for optimizer comparisons while varying:
    #   - overlap omega in {0, 1}
    #   - lambda_pt in {-0.99*c_pt, 0, +0.99*c_pt}
    #   - c_pt in {1e-6, 1e-3, 1}
    #   - gamma_reinit in {0, 1, 10}
    #
    # We keep the two regimes used previously:
    #   - Dense PT regime: rho_pt=0.9999, omega fixed to 1.0, rho_ft in {0.1, 0.9}
    #   - Sparse PT overlap regime: rho_pt=0.1, rho_ft=0.1, omega endpoints {0, 1}
    a_pt = 1.0
    c_pts = [1e-6, 1e-3, 1.0]
    gammas = [0.0, 1.0, 10.0]
    omega_endpoints = [0.0, 1.0]

    def lambda_choices(c_pt: float) -> List[float]:
        c_pt = float(c_pt)
        return [-0.99 * c_pt, 0.0, 0.99 * c_pt]

    # Dense PT regime (full overlap only)
    rho_pt_dense = 0.9999
    omega_full = 1.0
    for rho_ft in [0.1, 0.9]:
        for c_pt in c_pts:
            for lambda_pt in lambda_choices(c_pt):
                for gamma_reinit in gammas:
                    add(
                        "ptft_oracle",
                        rho_pt=float(rho_pt_dense),
                        rho_ft=float(rho_ft),
                        omega=float(omega_full),
                        a_pt=float(a_pt),
                        c_pt=float(c_pt),
                        lambda_pt=float(lambda_pt),
                        gamma_reinit=float(gamma_reinit),
                    )

    # Sparse PT overlap regime (omega endpoints)
    rho_pt_sparse = 0.1
    rho_ft_sparse = 0.1
    for omega in omega_endpoints:
        for c_pt in c_pts:
            for lambda_pt in lambda_choices(c_pt):
                for gamma_reinit in gammas:
                    add(
                        "ptft_oracle",
                        rho_pt=float(rho_pt_sparse),
                        rho_ft=float(rho_ft_sparse),
                        omega=float(omega),
                        a_pt=float(a_pt),
                        c_pt=float(c_pt),
                        lambda_pt=float(lambda_pt),
                        gamma_reinit=float(gamma_reinit),
                    )

    # Stable ordering: bg first (by c), then ptft (by rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit)
    def sort_key(c: Config) -> Tuple:
        d = c.as_dict()
        if c.exp == "bg":
            return (0, float(d["c"]))
        return (
            1,
            float(d["rho_pt"]),
            float(d["rho_ft"]),
            float(d["omega"]),
            float(d["c_pt"]),
            float(d["lambda_pt"]),
            float(d["gamma_reinit"]),
        )

    cfgs.sort(key=sort_key)
    return cfgs


def _run_bg_one(save_folder: Path, *, seed: int, alpha: float, rho: float, c: float, lmda: float) -> Dict[str, Any]:
    n_train = _alpha_to_n_train(alpha, INP_DIM_BG)
    alpha_eff = n_train / float(INP_DIM_BG)
    save_folder.mkdir(parents=True, exist_ok=True)

    meta_path = save_folder / "results_meta.json"
    cfg_path = save_folder / "config.json"

    ran_now = False
    if not meta_path.exists():
        parser = bg.get_parser()
        argv = [
            "--seed", str(int(seed)),
            "--save_folder", str(save_folder),
            "--n_train", str(int(n_train)),
            "--n_test", str(int(N_TEST)),
            "--inp_dim", str(int(INP_DIM_BG)),
            "--rho", str(float(rho)),
            "--c", str(float(c)),
            "--lmda", str(float(lmda)),
            "--lr", str(float(LR)),
            "--epochs", str(int(EPOCHS_BG)),
            "--threshold", str(float(THRESHOLD)),
            "--stop_pred_mse", str(float(THRESHOLD)),  # avoid default 1e-10 mismatch
            "--test_every_n_epochs", str(int(TEST_EVERY_N_EPOCHS)),
        ]
        if NO_TUNING:
            argv.append("--no_tuning")
        args = parser.parse_args(argv)
        bg.main(args)
        ran_now = True

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cfgj = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return {
        "status": "ok" if meta else "missing_meta",
        "ran_now": bool(ran_now),
        "alpha": float(alpha_eff),
        "n_train": int(n_train),
        "inp_dim": int(INP_DIM_BG),
        "seed": int(seed),
        "rho": float(rho),
        "c": float(c),
        "lmda": float(lmda),
        "test_pred_mse": meta.get("final_test_pred_mse", np.nan),
        "train_pred_mse": meta.get("final_train_pred_mse", np.nan),
        "param_mse": meta.get("final_param_mse", np.nan),
        "stop_reason": meta.get("stop_reason", None),
        "final_epoch": meta.get("final_epoch", None),
        "save_folder": str(save_folder),
        "config_json": cfgj,
    }


def _run_ptft_one(
    save_folder: Path,
    *,
    seed: int,
    alpha: float,
    rho_pt: float,
    rho_ft: float,
    omega: float,
    a_pt: float,
    c_pt: float,
    lambda_pt: float,
    gamma_reinit: float,
    optimizer: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    fixed_point_beta_rate: float,
    fixed_point_consecutive_evals: int,
    fixed_point_grad_norm: float,
) -> Dict[str, Any]:
    n_train = _alpha_to_n_train(alpha, INP_DIM_PTFT)
    alpha_eff = n_train / float(INP_DIM_PTFT)
    save_folder.mkdir(parents=True, exist_ok=True)

    meta_path = save_folder / "results_meta.json"
    cfg_path = save_folder / "config.json"

    feasible, feas_msg = _ptft_feasible(rho_pt, rho_ft, omega)
    if not feasible:
        return {
            "status": "skipped_infeasible",
            "infeasible_reason": feas_msg,
            "ran_now": False,
            "alpha": float(alpha_eff),
            "n_train": int(n_train),
            "inp_dim": int(INP_DIM_PTFT),
            "seed": int(seed),
            "rho_pt": float(rho_pt),
            "rho_ft": float(rho_ft),
            "omega": float(omega),
            "a_pt": float(a_pt),
            "c_pt": float(c_pt),
            "lambda_pt": float(lambda_pt),
            "gamma_reinit": float(gamma_reinit),
            "test_pred_mse": np.nan,
            "train_pred_mse": np.nan,
            "param_mse": np.nan,
            "stop_reason": "skipped_infeasible",
            "final_epoch": None,
            "save_folder": str(save_folder),
        }

    ran_now = False
    if not meta_path.exists():
        parser = ptft.get_parser()
        argv = [
            "--seed", str(int(seed)),
            "--save_folder", str(save_folder),
            "--inp_dim", str(int(INP_DIM_PTFT)),
            "--n_train", str(int(n_train)),
            "--n_test", str(int(N_TEST)),
            "--rho_pt", str(float(rho_pt)),
            "--rho_ft", str(float(rho_ft)),
            "--omega", str(float(omega)),
            "--a_pt", str(float(a_pt)),
            "--c_pt", str(float(c_pt)),
            # NOTE: for negative floats, use --flag=value so argparse does not
            # misinterpret the next token as another option.
            f"--lambda_pt={float(lambda_pt)}",
            "--gamma_reinit", str(float(gamma_reinit)),
            "--lr", str(float(lr)),
            "--epochs", str(int(epochs)),
            "--threshold", str(float(THRESHOLD)),
            "--test_every_n_epochs", str(int(TEST_EVERY_N_EPOCHS)),
            "--optimizer", str(optimizer),
            "--weight_decay", str(float(weight_decay)),
            "--fixed_point_beta_rate", str(float(fixed_point_beta_rate)),
            "--fixed_point_consecutive_evals", str(int(fixed_point_consecutive_evals)),
            "--fixed_point_grad_norm", str(float(fixed_point_grad_norm)),
        ]
        if NO_TUNING:
            argv.append("--no_tuning")
        args = parser.parse_args(argv)
        ptft.main(args)
        ran_now = True

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cfgj = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    return {
        "status": "ok" if meta else "missing_meta",
        "infeasible_reason": None,
        "ran_now": bool(ran_now),
        "alpha": float(alpha_eff),
        "n_train": int(n_train),
        "inp_dim": int(INP_DIM_PTFT),
        "seed": int(seed),
        "rho_pt": float(rho_pt),
        "rho_ft": float(rho_ft),
        "omega": float(omega),
        "a_pt": float(a_pt),
        "c_pt": float(c_pt),
        "lambda_pt": float(lambda_pt),
        "gamma_reinit": float(gamma_reinit),
        "optimizer": str(optimizer),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "test_pred_mse": meta.get("final_test_pred_mse", np.nan),
        "train_pred_mse": meta.get("final_train_pred_mse", np.nan),
        "param_mse": meta.get("final_param_mse", np.nan),
        "stop_reason": meta.get("stop_reason", None),
        "final_epoch": meta.get("final_epoch", None),
        "save_folder": str(save_folder),
        "config_json": cfgj,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Custom diagonal oracle sweeps (SLURM-array driver)")
    p.add_argument("array_id", type=int, nargs="?", default=None, help="SLURM array task ID (0-based)")
    p.add_argument("--run_name", type=str, default=None, help="Run name (default: timestamped)")
    p.add_argument(
        "--base_dir",
        type=str,
        default="results/diagonal/custom_oracle_sweeps",
        help="Base output directory (relative to repo root)",
    )
    p.add_argument("--list", action="store_true", help="List configs (with feasibility) and exit")
    p.add_argument("--dry_run", action="store_true", help="Print what would run for array_id then exit")
    # Training overrides (useful for optimizer comparisons via env-var passthrough from SLURM)
    p.add_argument("--optimizer", type=str, default="full_batch", choices=["full_batch", "sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=EPOCHS_PTFT)
    p.add_argument("--fixed_point_beta_rate", type=float, default=1e-6)
    p.add_argument("--fixed_point_consecutive_evals", type=int, default=2)
    p.add_argument("--fixed_point_grad_norm", type=float, default=0.0)
    args = p.parse_args()

    cfgs = _make_configs()
    n_cfg = len(cfgs)

    run_name = args.run_name or f"run_{_now_tag()}"
    base_dir = Path(args.base_dir) / run_name
    master_csv = base_dir / "master.csv"

    # List mode
    if args.list:
        print(f"Total configs: {n_cfg}")
        for i, c in enumerate(cfgs):
            d = c.as_dict()
            if c.exp == "bg":
                feas = "n/a"
            else:
                ok, msg = _ptft_feasible(d["rho_pt"], d["rho_ft"], d["omega"])
                feas = "✓" if ok else f"✗ ({msg})"
            print(f"[{i:02d}] {c.exp} {d}  feas={feas}")
        print(f"\nRun dir would be: {base_dir}")
        print(f"Master CSV would be: {master_csv}")
        return 0

    if args.array_id is None:
        raise SystemExit("ERROR: provide array_id or use --list")
    if args.array_id < 0 or args.array_id >= n_cfg:
        raise SystemExit(f"ERROR: array_id {args.array_id} out of range [0, {n_cfg - 1}]")

    cfg_id = int(args.array_id)
    cfg = cfgs[cfg_id]
    cfg_dict = cfg.as_dict()

    print("=" * 80)
    print(f"Custom oracle sweep driver: cfg_id={cfg_id} / {n_cfg - 1}")
    print(f"run_name: {run_name}")
    print(f"run_dir: {base_dir}")
    print(f"master_csv: {master_csv}")
    print(f"config: exp={cfg.exp} params={cfg_dict}")
    print(f"training: optimizer={args.optimizer} lr={args.lr} weight_decay={args.weight_decay}")
    print(
        f"stopping: epochs={args.epochs} fixed_point_beta_rate={args.fixed_point_beta_rate} "
        f"fixed_point_consecutive_evals={args.fixed_point_consecutive_evals} fixed_point_grad_norm={args.fixed_point_grad_norm}"
    )
    print("=" * 80)

    key_cols = ["row_id"]

    if args.dry_run:
        print("DRY RUN: would run these (alpha, seed) pairs:")
        for alpha in ALPHA_VALUES:
            for seed in SEEDS:
                print(f"  alpha={alpha} seed={seed}")
        return 0

    # Run all (alpha, seed) pairs for this config
    for alpha in ALPHA_VALUES:
        for seed in SEEDS:
            n_train = _alpha_to_n_train(alpha, INP_DIM_BG if cfg.exp == "bg" else INP_DIM_PTFT)
            alpha_eff = n_train / float(INP_DIM_BG if cfg.exp == "bg" else INP_DIM_PTFT)

            # Unique row key and save folder
            row_id = (
                f"{cfg.exp}__cfg={cfg_id:02d}"
                f"__opt={args.optimizer}__wd={float(args.weight_decay):.6g}__lr={float(args.lr):.6g}"
                f"__alpha={alpha_eff:.6f}__seed={seed}"
            )
            run_subdir = (
                base_dir
                / "empirical_runs"
                / cfg.exp
                / f"opt={args.optimizer}__wd={float(args.weight_decay):.6g}__lr={float(args.lr):.6g}"
                / f"cfg={cfg_id:02d}"
            )
            save_folder = run_subdir / f"alpha={alpha_eff:.6f}__seed={seed}"

            if cfg.exp == "bg":
                res = _run_bg_one(
                    save_folder,
                    seed=seed,
                    alpha=alpha,
                    rho=float(cfg_dict["rho"]),
                    c=float(cfg_dict["c"]),
                    lmda=float(cfg_dict["lmda"]),
                )
                row = {
                    "row_id": row_id,
                    "exp": "bg",
                    "config_id": cfg_id,
                    "alpha": res["alpha"],
                    "n_train": res["n_train"],
                    "inp_dim": res["inp_dim"],
                    "seed": res["seed"],
                    "rho": res["rho"],
                    "c": res["c"],
                    "lmda": res["lmda"],
                    "status": res["status"],
                    "ran_now": res["ran_now"],
                    "test_pred_mse": res["test_pred_mse"],
                    "train_pred_mse": res["train_pred_mse"],
                    "param_mse": res["param_mse"],
                    "stop_reason": res["stop_reason"],
                    "final_epoch": res["final_epoch"],
                    "save_folder": res["save_folder"],
                    "run_name": run_name,
                }
            else:
                res = _run_ptft_one(
                    save_folder,
                    seed=seed,
                    alpha=alpha,
                    rho_pt=float(cfg_dict["rho_pt"]),
                    rho_ft=float(cfg_dict["rho_ft"]),
                    omega=float(cfg_dict["omega"]),
                    a_pt=float(cfg_dict["a_pt"]),
                    c_pt=float(cfg_dict["c_pt"]),
                    lambda_pt=float(cfg_dict["lambda_pt"]),
                    gamma_reinit=float(cfg_dict["gamma_reinit"]),
                    optimizer=str(args.optimizer),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    epochs=int(args.epochs),
                    fixed_point_beta_rate=float(args.fixed_point_beta_rate),
                    fixed_point_consecutive_evals=int(args.fixed_point_consecutive_evals),
                    fixed_point_grad_norm=float(args.fixed_point_grad_norm),
                )
                row = {
                    "row_id": row_id,
                    "exp": "ptft_oracle",
                    "config_id": cfg_id,
                    "alpha": res["alpha"],
                    "n_train": res["n_train"],
                    "inp_dim": res["inp_dim"],
                    "seed": res["seed"],
                    "rho_pt": res["rho_pt"],
                    "rho_ft": res["rho_ft"],
                    "omega": res["omega"],
                    "a_pt": res["a_pt"],
                    "c_pt": res["c_pt"],
                    "lambda_pt": res["lambda_pt"],
                    "gamma_reinit": res["gamma_reinit"],
                    "optimizer": res.get("optimizer", str(args.optimizer)),
                    "lr": res.get("lr", float(args.lr)),
                    "weight_decay": res.get("weight_decay", float(args.weight_decay)),
                    "status": res["status"],
                    "infeasible_reason": res.get("infeasible_reason", None),
                    "ran_now": res["ran_now"],
                    "test_pred_mse": res["test_pred_mse"],
                    "train_pred_mse": res["train_pred_mse"],
                    "param_mse": res["param_mse"],
                    "stop_reason": res["stop_reason"],
                    "final_epoch": res["final_epoch"],
                    "save_folder": res["save_folder"],
                    "run_name": run_name,
                }

            _safe_csv_upsert_row(master_csv, row, key_cols=key_cols)
            print(f"[master_csv] recorded row_id={row_id} status={row['status']} ran_now={row['ran_now']}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

