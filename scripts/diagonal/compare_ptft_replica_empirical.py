#!/usr/bin/env python3
"""
Thin wrapper: PT+FT diagonal (oracle) — replica curve(s) + empirical runs + one overlay plot.

Design goals:
- Reuse existing, validated logic:
  * empirical: experiments/diagonal/diagonal_ptft_oracle.py
  * replica + overlay: scripts/diagonal/plot_step3_validation.py
- Keep ft_regulariser_scale (replica regularization strength) distinct from lambda_pt (PT init param).

This file is intentionally a *script* (scripts/ isn't a Python package). To reuse local helpers,
we add this directory to sys.path and import plot_step3_validation by filename.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    import plot_step3_validation as p3  # reuse aggregation + replica + overlay
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import plot_step3_validation from scripts/diagonal. "
        "Run from repo root with the correct environment."
    ) from e

try:
    from experiments.diagonal.diagonal_ptft_oracle import get_parser as ptft_get_parser
    from experiments.diagonal.diagonal_ptft_oracle import main as ptft_main
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import experiments.diagonal.diagonal_ptft_oracle. "
        "Make sure you are using the project environment (e.g. mtl_ft) with torch installed."
    ) from e


def _as_list(x: float | Sequence[float]) -> List[float]:
    if isinstance(x, (int, float, np.floating)):
        return [float(x)]
    return [float(v) for v in x]


def _fmt_float(x: float) -> str:
    # Stable, filename-friendly float formatting
    return f"{float(x):.6g}".replace("+", "")


def _run_tag(
    *,
    rho_pt: float,
    rho_ft: float,
    a_pt: float,
    c_pt: float,
    lambda_pt: float,
    gamma_reinit: float,
    ft_regulariser_scale: float,
) -> str:
    return (
        f"ptft_oracle__"
        f"rpt={_fmt_float(rho_pt)}__rft={_fmt_float(rho_ft)}__"
        f"apt={_fmt_float(a_pt)}__cpt={_fmt_float(c_pt)}__"
        f"lpt={_fmt_float(lambda_pt)}__gam={_fmt_float(gamma_reinit)}__"
        f"ftreg={_fmt_float(ft_regulariser_scale)}"
    )


def _replica_cache_path(
    cache_dir: Path,
    *,
    rho_pt: float,
    rho_ft: float,
    omega: float,
    a_pt: float,
    c_pt: float,
    lambda_pt: float,
    gamma_reinit: float,
    ft_regulariser_scale: float,
    alpha_min: float,
    alpha_max: float,
    alpha_points: int,
    mc_samples: int,
    seed: int,
) -> Path:
    name = (
        "replica_curve_"
        f"teacher=ptft_oracle--"
        f"rpt={rho_pt:.4f}--rft={rho_ft:.4f}--om={omega:.4f}--"
        f"apt={a_pt:.2f}--cpt={c_pt:.4f}--lpt={lambda_pt:.2f}--gam={gamma_reinit:.2f}--"
        f"ft_reg={ft_regulariser_scale:.6e}--"
        f"alpha_min={alpha_min:.4f}--alpha_max={alpha_max:.4f}--alpha_points={alpha_points}--"
        f"mc_samples={mc_samples}--seed={seed}.csv"
    )
    return cache_dir / name


def _check_feasible(rho_pt: float, rho_ft: float, omega: float) -> None:
    # Same feasibility constraints used by the PTFT sampler:
    #   omega * rho_ft <= rho_pt
    #   rho_pt + (1-omega) * rho_ft <= 1
    rho_pt = float(rho_pt)
    rho_ft = float(rho_ft)
    omega = float(omega)
    if not (0.0 < rho_pt < 1.0):
        raise ValueError(f"rho_pt must be in (0,1), got {rho_pt}")
    if not (0.0 < rho_ft < 1.0):
        raise ValueError(f"rho_ft must be in (0,1), got {rho_ft}")
    if not (0.0 <= omega <= 1.0):
        raise ValueError(f"omega must be in [0,1], got {omega}")
    if omega * rho_ft > rho_pt + 1e-12:
        raise ValueError(
            f"Infeasible: omega*rho_ft <= rho_pt violated. "
            f"omega*rho_ft={omega*rho_ft:.6g} > rho_pt={rho_pt:.6g}"
        )
    if rho_pt + (1.0 - omega) * rho_ft > 1.0 + 1e-12:
        raise ValueError(
            f"Infeasible: rho_pt + (1-omega)*rho_ft <= 1 violated. "
            f"lhs={rho_pt + (1.0-omega)*rho_ft:.6g} > 1"
        )


def compare_ptft_replica_vs_empirical(
    *,
    rho_pt: float,
    rho_ft: float,
    omega: float | Sequence[float],
    c_pt: float,
    lambda_pt: float,
    gamma_reinit: float,
    a_pt: float = 1.0,
    ft_regulariser_scale: float = 1e-6,
    num_seeds: int = 3,
    seed_base: int = 0,
    replica_seed: int = 12345,
    inp_dim: int = 1000,
    n_test: int = 10000,
    alpha_values: Optional[Sequence[float]] = None,
    # training knobs (empirical)
    lr: float = 0.5,
    epochs: int = 5_000_000,
    threshold: float = 1e-12,
    test_every_n_epochs: int = 200,
    no_tuning: bool = True,
    # replica knobs
    alpha_min: float = 0.008,
    alpha_max: float = 1.0,
    alpha_points: int = 100,
    mc_samples: int = 50_000,
    # IO/control
    output_dir: str = "figures/ptft_compare",
    run_empirical: bool = True,
    run_replica: bool = True,
    make_plot: bool = True,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Compute replica curve(s), optionally run empirical PT+FT oracle experiments, and make overlay plot.

    Returns a dict with:
      - run_dir
      - empirical_csv
      - empirical_raw (DataFrame)
      - empirical_agg (DataFrame or empty)
      - replica_curves (dict)
      - acceptance_results (dict or empty)
      - plot_paths (dict)
    """
    omegas = _as_list(omega)
    for om in omegas:
        _check_feasible(rho_pt, rho_ft, om)
    if float(c_pt) <= 0.0:
        raise ValueError(f"c_pt must be > 0, got {c_pt}")
    if int(num_seeds) < 1:
        raise ValueError(f"num_seeds must be >= 1, got {num_seeds}")

    if alpha_values is None:
        alpha_values = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_values = [float(a) for a in alpha_values]
    for a in alpha_values:
        if not (0.0 < a <= 1.0):
            raise ValueError(f"alpha_values must be in (0,1], got {a}")

    out_root = Path(output_dir)
    run_dir = out_root / _run_tag(
        rho_pt=rho_pt,
        rho_ft=rho_ft,
        a_pt=a_pt,
        c_pt=c_pt,
        lambda_pt=lambda_pt,
        gamma_reinit=gamma_reinit,
        ft_regulariser_scale=ft_regulariser_scale,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    emp_runs_dir = run_dir / "empirical_runs"
    emp_runs_dir.mkdir(parents=True, exist_ok=True)
    replica_cache_dir = run_dir / "replica_cache"
    replica_cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Empirical runs
    # -------------------------
    rows: List[Dict[str, Any]] = []
    if run_empirical:
        parser = ptft_get_parser()
        for om in omegas:
            for s in range(int(num_seeds)):
                seed = int(seed_base + s)
                for alpha in alpha_values:
                    n_train = max(1, int(round(alpha * int(inp_dim))))
                    alpha_eff = n_train / float(inp_dim)
                    save_folder = emp_runs_dir / f"omega={om:.3f}__alpha={alpha_eff:.6f}__seed={seed}"
                    save_folder.mkdir(parents=True, exist_ok=True)

                    meta_path = save_folder / "results_meta.json"
                    cfg_path = save_folder / "config.json"

                    did_run = False
                    if not (skip_existing and meta_path.exists()):
                        argv = [
                            "--seed", str(seed),
                            "--save_folder", str(save_folder),
                            "--inp_dim", str(int(inp_dim)),
                            "--n_train", str(int(n_train)),
                            "--n_test", str(int(n_test)),
                            "--rho_pt", str(float(rho_pt)),
                            "--rho_ft", str(float(rho_ft)),
                            "--omega", str(float(om)),
                            "--a_pt", str(float(a_pt)),
                            "--c_pt", str(float(c_pt)),
                            "--lambda_pt", str(float(lambda_pt)),
                            "--gamma_reinit", str(float(gamma_reinit)),
                            "--lr", str(float(lr)),
                            "--epochs", str(int(epochs)),
                            "--threshold", str(float(threshold)),
                            "--test_every_n_epochs", str(int(test_every_n_epochs)),
                        ]
                        if bool(no_tuning):
                            argv.append("--no_tuning")

                        args = parser.parse_args(argv)
                        ptft_main(args)
                        did_run = True

                    # Load meta + config (even if skipped)
                    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                    cfgj = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

                    rows.append(
                        {
                            "omega": float(om),
                            "alpha": float(alpha_eff),
                            "n_train": int(n_train),
                            "seed": int(seed),
                            "empirical_omega": float(cfgj.get("empirical_omega", om)),
                            "rho_pt": float(rho_pt),
                            "rho_ft": float(rho_ft),
                            "a_pt": float(a_pt),
                            "c_pt": float(c_pt),
                            "lambda_pt": float(lambda_pt),
                            "gamma_reinit": float(gamma_reinit),
                            "ft_regulariser_scale": float(ft_regulariser_scale),
                            "test_pred_mse": meta.get("final_test_pred_mse", np.nan),
                            "train_pred_mse": meta.get("final_train_pred_mse", np.nan),
                            "param_mse": meta.get("final_param_mse", np.nan),
                            "stop_reason": meta.get("stop_reason", None),
                            "final_epoch": meta.get("final_epoch", None),
                            "save_folder": str(save_folder),
                            "ran_now": bool(did_run),
                        }
                    )

    df_emp = pd.DataFrame(rows)
    empirical_csv = run_dir / "empirical_results.csv"
    if not df_emp.empty:
        df_emp.to_csv(empirical_csv, index=False)

    # -------------------------
    # Aggregate empirical (reuse)
    # -------------------------
    agg_df = pd.DataFrame()
    if not df_emp.empty:
        agg_df = p3.aggregate_empirical_results(str(empirical_csv))

    # -------------------------
    # Replica curves (reuse + cache)
    # -------------------------
    replica_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if run_replica:
        sigma0_2 = 0.0
        beta_min = 1.0 / float(alpha_max)
        beta_max = 1.0 / float(alpha_min)
        cfg = p3.build_config(
            rho=float(rho_ft),
            sigma0_2=float(sigma0_2),
            beta_min=float(beta_min),
            beta_max=float(beta_max),
            beta_points=int(alpha_points),
        )
        alpha_range = np.linspace(float(alpha_min), float(alpha_max), int(alpha_points))

        for om in omegas:
            seed = int(replica_seed + int(round(1000.0 * float(om))))
            cache_path = _replica_cache_path(
                replica_cache_dir,
                rho_pt=float(rho_pt),
                rho_ft=float(rho_ft),
                omega=float(om),
                a_pt=float(a_pt),
                c_pt=float(c_pt),
                lambda_pt=float(lambda_pt),
                gamma_reinit=float(gamma_reinit),
                ft_regulariser_scale=float(ft_regulariser_scale),
                alpha_min=float(alpha_min),
                alpha_max=float(alpha_max),
                alpha_points=int(alpha_points),
                mc_samples=int(mc_samples),
                seed=int(seed),
            )

            if cache_path.exists():
                dfc = pd.read_csv(cache_path)
                replica_curves[str(float(om))] = (dfc["alpha"].values, dfc["mse"].values)
            else:
                rng = np.random.default_rng(seed)
                alpha_rep, mse_rep = p3.compute_replica_curve_ptft_oracle(
                    float(om),
                    float(rho_pt),
                    float(rho_ft),
                    float(a_pt),
                    float(c_pt),
                    float(lambda_pt),
                    float(gamma_reinit),
                    float(ft_regulariser_scale),
                    alpha_range,
                    cfg,
                    int(mc_samples),
                    rng,
                )
                replica_curves[str(float(om))] = (alpha_rep, mse_rep)
                pd.DataFrame({"alpha": alpha_rep, "mse": mse_rep}).to_csv(cache_path, index=False)

    # -------------------------
    # Overlay plot (reuse)
    # -------------------------
    acceptance_results: Dict[str, Any] = {}
    plot_paths: Dict[str, str] = {}
    if make_plot and (not agg_df.empty) and replica_curves:
        acceptance_results = p3.check_acceptance_criteria(agg_df, replica_curves)
        p3.plot_step3_overlay(
            agg_df,
            replica_curves,
            str(run_dir),
            acceptance_results,
            rho_pt=float(rho_pt),
            rho_ft=float(rho_ft),
            a_pt=float(a_pt),
            c_pt=float(c_pt),
            lambda_pt=float(lambda_pt),
            gamma_reinit=float(gamma_reinit),
            ft_regulariser_scale=float(ft_regulariser_scale),
        )
        plot_paths = {
            "png": str(run_dir / "step3_validation_overlay.png"),
            "pdf": str(run_dir / "step3_validation_overlay.pdf"),
        }

    # Persist acceptance results (nice for quick iteration/regression)
    if acceptance_results:
        (run_dir / "acceptance_results.json").write_text(json.dumps(acceptance_results, indent=2))

    return {
        "run_dir": str(run_dir),
        "empirical_csv": str(empirical_csv),
        "empirical_raw": df_emp,
        "empirical_agg": agg_df,
        "replica_curves": replica_curves,
        "acceptance_results": acceptance_results,
        "plot_paths": plot_paths,
    }


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="PT+FT diagonal oracle: empirical vs replica overlay")
    p.add_argument("--rho_pt", type=float, required=True)
    p.add_argument("--rho_ft", type=float, required=True)
    p.add_argument("--omega", type=float, nargs="+", required=True)
    p.add_argument("--c_pt", type=float, default=0.001)
    p.add_argument("--lambda_pt", type=float, default=0.0)
    p.add_argument("--gamma_reinit", type=float, default=0.0)
    p.add_argument("--a_pt", type=float, default=1.0)
    p.add_argument("--ft_regulariser_scale", type=float, default=1e-6)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--seed_base", type=int, default=0)
    p.add_argument("--replica_seed", type=int, default=12345)
    p.add_argument("--inp_dim", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=10000)
    p.add_argument("--alpha_values", type=float, nargs="+", default=None)
    p.add_argument("--output_dir", type=str, default="figures/ptft_compare")
    p.add_argument("--run_empirical", action="store_true")
    p.add_argument("--run_replica", action="store_true")
    p.add_argument("--no_plot", action="store_true")
    p.add_argument("--no_skip_existing", action="store_true")
    # fast knobs
    p.add_argument("--epochs", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=1e-12)
    p.add_argument("--test_every_n_epochs", type=int, default=200)
    p.add_argument("--no_tuning", action="store_true")
    p.add_argument("--alpha_min", type=float, default=0.008)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--alpha_points", type=int, default=100)
    p.add_argument("--mc_samples", type=int, default=50_000)

    args = p.parse_args()

    res = compare_ptft_replica_vs_empirical(
        rho_pt=args.rho_pt,
        rho_ft=args.rho_ft,
        omega=args.omega,
        c_pt=args.c_pt,
        lambda_pt=args.lambda_pt,
        gamma_reinit=args.gamma_reinit,
        a_pt=args.a_pt,
        ft_regulariser_scale=args.ft_regulariser_scale,
        num_seeds=args.num_seeds,
        seed_base=args.seed_base,
        replica_seed=args.replica_seed,
        inp_dim=args.inp_dim,
        n_test=args.n_test,
        alpha_values=args.alpha_values,
        lr=args.lr,
        epochs=args.epochs,
        threshold=args.threshold,
        test_every_n_epochs=args.test_every_n_epochs,
        no_tuning=args.no_tuning,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_points=args.alpha_points,
        mc_samples=args.mc_samples,
        output_dir=args.output_dir,
        run_empirical=bool(args.run_empirical),
        run_replica=bool(args.run_replica),
        make_plot=not bool(args.no_plot),
        skip_existing=not bool(args.no_skip_existing),
    )

    print(f"run_dir: {res['run_dir']}")
    if res.get("plot_paths"):
        print(f"plot_png: {res['plot_paths'].get('png')}")
        print(f"plot_pdf: {res['plot_paths'].get('pdf')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


