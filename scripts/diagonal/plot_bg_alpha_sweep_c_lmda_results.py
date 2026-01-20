#!/usr/bin/env python3
"""
Plot BG alpha sweep results for single-task diagonal network, varying (c, lmda).

Styling requested:
  - dashed lines for c=0.001
  - solid  lines for c=0.5
  - color indicates lmda sign (lmda = -0.95c, 0, +0.95c)

Input:
  experiment_results_bg_alpha_sweep_c_lmda.csv

Output:
  figures/diagonal/bg_generalization/empirical_bg_alpha_sweep__c001_dashed__c05_solid__lmda_pm095c.(png|pdf)
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


def nearest_frac(x: float, targets: np.ndarray) -> float:
    idx = int(np.argmin(np.abs(targets - x)))
    return float(targets[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default="experiment_results_bg_alpha_sweep_c_lmda.csv")
    p.add_argument("--out_dir", type=str, default="figures/diagonal/bg_generalization")
    p.add_argument("--rho", type=float, default=0.04)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    for col in ["alpha", "param_mse", "train_pred_mse", "c", "lmda"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["alpha", "c", "lmda", "param_mse"]).copy()
    if df.empty:
        raise ValueError("No valid rows found in CSV (need alpha,c,lmda,param_mse).")

    # Infer lmda_frac = lmda / c and snap to nearest of {-0.95, 0, +0.95}
    targets = np.array([-0.95, 0.0, 0.95], dtype=float)
    df["lmda_frac"] = df["lmda"] / df["c"]
    df["lmda_frac_bin"] = df["lmda_frac"].apply(lambda x: nearest_frac(float(x), targets))

    # Aggregate by (c, lmda_frac_bin, alpha)
    agg = (
        df.groupby(["c", "lmda_frac_bin", "alpha"])
        .agg(
            param_mse_mean=("param_mse", "mean"),
            param_mse_median=("param_mse", "median"),
            count=("param_mse", "count"),
        )
        .reset_index()
        .sort_values(["c", "lmda_frac_bin", "alpha"])
    )

    # Styling
    # (We key by rounded c to avoid float formatting drift.)
    ls_by_c = {0.001: "--", 0.5: "-"}
    color_by_frac = {-0.95: "tab:blue", 0.0: "k", 0.95: "tab:red"}

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot mean curves; label uses lmda = frac*c
    for (c_val, frac), sub in agg.groupby(["c", "lmda_frac_bin"]):
        c_val = float(c_val)
        frac = float(frac)
        sub = sub.sort_values("alpha")

        # Select linestyle by nearest target c in {0.001, 0.5}
        c_key = 0.001 if abs(c_val - 0.001) < abs(c_val - 0.5) else 0.5
        ls = ls_by_c[c_key]
        color = color_by_frac.get(frac, "gray")

        lmda_val = frac * c_val
        ax.plot(
            sub["alpha"].values,
            to_db(sub["param_mse_mean"].values),
            linestyle=ls,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label=rf"c={c_val:g}, $\lambda$={lmda_val:.3g} (= {frac:+.2f}c)",
        )

    ax.set_xlabel(r"$\alpha = n_{\mathrm{train}} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title(rf"Empirical BG generalization vs $\alpha$ (single-task), $\rho={args.rho:.2f}$", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()

    out_png = out_dir / "empirical_bg_alpha_sweep__c001_dashed__c05_solid__lmda_pm095c.png"
    out_pdf = out_dir / "empirical_bg_alpha_sweep__c001_dashed__c05_solid__lmda_pm095c.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()


