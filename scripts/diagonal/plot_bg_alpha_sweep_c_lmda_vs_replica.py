#!/usr/bin/env python3
"""
One-figure overlay: empirical BG alpha sweep (varying c and lmda) vs replica theory.

Empirical input:
  experiment_results_bg_alpha_sweep_c_lmda.csv

Replica input:
  cached curves produced by scripts/diagonal/plot_replica_q_bg.py in:
    <output_dir>/replica_cache/

Requested styling (kept consistent with earlier plot):
  - linestyle encodes c: dashed for c=0.001, solid for c=0.5
  - color encodes lmda sign: -0.95c (blue), 0 (black), +0.95c (red)
  - replica curve plotted once per c (same style as c, thicker gray line)

Output:
  figures/diagonal/bg_generalization/empirical_vs_replica__c_lmda_sweep.(png|pdf)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def load_replica_curve(
    output_dir: str,
    rho: float,
    c: float,
    ft_regulariser_scale: float,
    alpha_min: float,
    alpha_max: float,
    alpha_points: int,
    mc_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    # Primary location: <output_dir>/replica_cache (default used by plot_replica_q_bg.py)
    cache_dir = os.path.join(output_dir, "replica_cache")
    cache_filename = (
        f"replica_curve_teacher=bg--"
        f"rho={rho:.6f}--c={c:.6f}--"
        f"ft_reg={ft_regulariser_scale:.6e}--alpha_min={alpha_min:.4f}--"
        f"alpha_max={alpha_max:.4f}--alpha_points={alpha_points}--"
        f"mc_samples={mc_samples}--seed={seed}.csv"
    )
    cache_path = os.path.join(cache_dir, cache_filename)
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        return df["alpha"].values, df["mse"].values

    # Fallback: validation directories sometimes store the exact same cache filename
    # under figures/step2a_validation/replica_cache. This keeps the script robust.
    fallback = os.path.join("figures", "step2a_validation", "replica_cache", cache_filename)
    if os.path.exists(fallback):
        df = pd.read_csv(fallback)
        return df["alpha"].values, df["mse"].values

    return None, None


def nearest_frac(x: float, targets: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(targets - x)))
    return float(targets[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="experiment_results_bg_alpha_sweep_c_lmda.csv")
    p.add_argument("--output_dir", type=str, default="figures/diagonal/bg_generalization")
    p.add_argument("--rho", type=float, default=0.04)
    p.add_argument("--ft_regulariser_scale", type=float, default=1e-6)
    p.add_argument("--alpha_min", type=float, default=0.008)
    p.add_argument("--alpha_max", type=float, default=1.0)
    p.add_argument("--alpha_points", type=int, default=100)
    p.add_argument("--mc_samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--use_db", action="store_true", help="Plot in dB (10log10 MSE). Default is linear MSE.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    for col in ["alpha", "param_mse", "train_pred_mse", "test_pred_mse", "c", "lmda", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["alpha", "c", "lmda", "param_mse", "seed"]).copy()
    if df.empty:
        raise ValueError("No valid rows in CSV after filtering required columns.")

    # Bin lmda by lmda/c into {-0.95, 0, +0.95}
    targets = np.array([-0.95, 0.0, 0.95], dtype=float)
    df["lmda_frac"] = df["lmda"] / df["c"]
    df["lmda_frac_bin"] = df["lmda_frac"].apply(lambda x: nearest_frac(float(x), targets))

    # Aggregate empirical param MSE by (c, lmda_frac_bin, alpha)
    agg = (
        df.groupby(["c", "lmda_frac_bin", "alpha"])
        .agg(
            param_mse_mean=("param_mse", "mean"),
            param_mse_median=("param_mse", "median"),
            count=("param_mse", "count"),
            q25=("param_mse", lambda x: np.percentile(x, 25)),
            q75=("param_mse", lambda x: np.percentile(x, 75)),
        )
        .reset_index()
        .sort_values(["c", "lmda_frac_bin", "alpha"])
    )

    # Styling (improved readability per request):
    # - Color encodes c (same color for replica + empirical at that c)
    # - Replica curves dashed for BOTH c values
    # - Empirical curves solid for BOTH c values
    # - lmda setting distinguished by marker shape
    c_values = sorted(agg["c"].unique().tolist())
    color_by_c = {0.001: "tab:blue", 0.5: "tab:orange"}
    marker_by_frac = {-0.95: "v", 0.0: "o", 0.95: "^"}  # down, circle, up

    def c_key(c_val: float) -> float:
        # snap to the two canonical c values for styling
        return 0.001 if abs(c_val - 0.001) < abs(c_val - 0.5) else 0.5

    fig, ax = plt.subplots(figsize=(10.5, 7.5))

    # Replica curves (one per c): dashed, thicker, same color as c
    for c in c_values:
        ck = c_key(float(c))
        color = color_by_c.get(ck, "0.4")
        a_rep, mse_rep = load_replica_curve(
            output_dir=str(out_dir),
            rho=args.rho,
            c=float(c),
            ft_regulariser_scale=args.ft_regulariser_scale,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_points=args.alpha_points,
            mc_samples=args.mc_samples,
            seed=args.seed,
        )
        if a_rep is None:
            print(
                f"WARNING: replica cache not found for c={c:.6f}. "
                f"Run scripts/diagonal/plot_replica_q_bg.py with matching params to generate it."
            )
            continue
        y = mse_rep
        if args.use_db:
            y = to_db(y)
        ax.plot(
            a_rep,
            y,
            linestyle="--",
            color=color,
            linewidth=3.2,
            label=f"Replica (c={float(c):g})",
            zorder=1,
        )

    # Empirical curves (one per c, frac): solid, same color as c, different markers by lmda_frac
    for (c_val, frac), sub in agg.groupby(["c", "lmda_frac_bin"]):
        c_val = float(c_val)
        frac = float(frac)
        sub = sub.sort_values("alpha")
        ck = c_key(c_val)
        color = color_by_c.get(ck, "0.2")
        marker = marker_by_frac.get(frac, "o")

        y = sub["param_mse_mean"].values
        y_lo = sub["q25"].values
        y_hi = sub["q75"].values
        if args.use_db:
            y = to_db(y)
            y_lo = to_db(y_lo)
            y_hi = to_db(y_hi)

        lmda_val = frac * c_val
        ax.plot(
            sub["alpha"].values,
            y,
            linestyle="-",
            color=color,
            linewidth=2.0,
            marker=marker,
            markersize=5,
            label=rf"Empirical c={c_val:g}, $\lambda$={frac:+.2f}c",
            zorder=2,
        )
        ax.fill_between(
            sub["alpha"].values,
            y_lo,
            y_hi,
            color=color,
            alpha=0.12,
            linewidth=0.0,
            zorder=0,
        )

    ax.set_xlabel(r"$\alpha = n_{\mathrm{train}} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)" if args.use_db else "Parameter MSE", fontsize=14)
    ax.set_title(rf"BG generalization vs replica (single-task), $\rho={args.rho:.2f}$", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Keep legend readable
    ax.legend(fontsize=9, loc="best", ncol=1)
    fig.tight_layout()

    base = "empirical_vs_replica__c_lmda_sweep"
    out_png = out_dir / f"{base}.png"
    out_pdf = out_dir / f"{base}.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()


