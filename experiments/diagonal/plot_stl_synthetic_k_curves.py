#!/usr/bin/env python3
"""
Make the exact plot you asked for:
  - x-axis: delta = n_train / inp_dim
  - y-axis: final_val_mse
  - multiple curves on the same axes: different imposed k_i scales (k_scale), log-spaced

Reads experiment_results_stl_synthk.csv written by `stl_synthetic_k_q_train.py`.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean_se(df: pd.DataFrame, group_cols, value_col):
    g = df.groupby(group_cols)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", type=str, default="experiment_results_stl_synthk.csv")
    p.add_argument("--filter_contains", type=str, default="data/diagonal/stl_synthk_q/")
    p.add_argument("--out", type=str, default="figures/stl_synthk_q/gen_vs_delta_curves.jpg")
    args = p.parse_args()

    df = pd.read_csv(args.results_csv)
    if args.filter_contains:
        df = df[df["save_folder"].astype(str).str.contains(args.filter_contains, na=False)].copy()
    if df.empty:
        print("No matching rows found.")
        return

    for col in ["delta", "final_val_mse", "k_scale", "seed", "active_dim", "inp_dim", "n_train"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Enforce delta convention: delta = d/n (do NOT rely on stored column from older runs)
    if "inp_dim" in df.columns and "n_train" in df.columns:
        df["delta_dn"] = df["inp_dim"] / df["n_train"]
    else:
        # fall back: use existing delta
        df["delta_dn"] = df["delta"]

    # Aggregate across seeds
    agg = mean_se(df, ["k_scale", "delta_dn"], "final_val_mse")

    fig, ax = plt.subplots(figsize=(7, 5))
    for k_scale, sub in agg.groupby("k_scale"):
        sub = sub.sort_values("delta_dn")
        # Mean curve + explicit error bars (± 2 * SE across seeds)
        ax.errorbar(
            sub["delta_dn"],
            sub["mean"],
            yerr=2 * sub["se"],
            fmt="o-",
            lw=1.8,
            capsize=3,
            alpha=0.95,
            label=f"sqrt(k) scale={k_scale:.0e}",
        )
        # light band as well (helps visibility on dense plots)
        ax.fill_between(
            sub["delta_dn"],
            np.maximum(sub["mean"] - 2 * sub["se"], 0),
            sub["mean"] + 2 * sub["se"],
            alpha=0.08,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta = d/n$")
    ax.set_ylabel("final_val_mse")
    ax.set_title("STL synthetic-k: gen vs delta=d/n (5 k-scales)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as JPG for lab-machine workflows (no interactive display).
    # Matplotlib chooses format from extension; we also pass it explicitly.
    fig.savefig(out_path, dpi=200, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


