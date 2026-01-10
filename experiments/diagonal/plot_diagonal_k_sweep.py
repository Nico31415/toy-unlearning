#!/usr/bin/env python3
"""
Plotting for diagonal k-sweep experiments.

Reads `experiment_results_k.csv` (written by `diagonal_network_finetune_k_runner.py`) and produces:
  - Gen loss vs sample size / delta for each gamma
  - Gen loss vs gamma (fixed n_train2)
  - Gen loss vs lambda (fixed gamma)
  - Histograms of k_i and r_i from `k_r_arrays_path` for representative runs
"""

import argparse
import os
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _maybe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_results(csv_path: str, filter_contains: str | None):
    df = pd.read_csv(csv_path)
    if filter_contains:
        df = df[df["save_path"].astype(str).str.contains(filter_contains, na=False)].copy()
    return df


def mean_se(df: pd.DataFrame, group_cols, value_col):
    g = df.groupby(group_cols)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_gen_vs_delta(df: pd.DataFrame, out_dir: Path, facet_cols=("c", "lmda", "active_dim_2", "pretrain_overlap")):
    """
    For each facet group, plot mean final_val_mse vs delta (n_train2/inp_dim) with lines by gamma.
    """
    needed = {"final_val_mse", "n_train2", "inp_dim", "gamma"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for gen-vs-delta plot: {missing}")

    df = df.copy()
    df["delta"] = df["n_train2"] / df["inp_dim"]
    facet_cols = [c for c in facet_cols if c in df.columns]
    group_cols = facet_cols + ["gamma", "delta"]
    agg = mean_se(df, group_cols, "final_val_mse")

    for keys, sub in agg.groupby(facet_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        title_bits = [f"{k}={v}" for k, v in zip(facet_cols, keys)]
        title = ", ".join(title_bits) if title_bits else "all"
        fig, ax = plt.subplots(figsize=(7, 5))
        for gamma, gsub in sub.groupby("gamma"):
            gsub = gsub.sort_values("delta")
            ax.plot(gsub["delta"], gsub["mean"], marker="o", lw=1.5, label=f"gamma={gamma:g}")
            ax.fill_between(
                gsub["delta"],
                np.maximum(gsub["mean"] - 2 * gsub["se"], 0),
                gsub["mean"] + 2 * gsub["se"],
                alpha=0.15,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\delta = n_{FT}/d$")
        ax.set_ylabel("Final val MSE")
        ax.set_title(f"Gen loss vs delta\n{title}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fname = "gen_vs_delta__" + "__".join([f"{k}={v}" for k, v in zip(facet_cols, keys)]).replace("/", "_") + ".png"
        fig.savefig(out_dir / fname.replace(".png", ".jpg"), dpi=200, bbox_inches="tight", format="jpg")
        plt.close(fig)


def plot_gen_vs_gamma(df: pd.DataFrame, out_dir: Path, n_train2_values=(32, 64, 128), facet_cols=("c", "lmda", "active_dim_2", "pretrain_overlap")):
    needed = {"final_val_mse", "gamma", "n_train2"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for gen-vs-gamma plot: {missing}")

    facet_cols = [c for c in facet_cols if c in df.columns]
    for n_train2 in n_train2_values:
        subdf = df[df["n_train2"] == n_train2].copy()
        if subdf.empty:
            continue
        agg = mean_se(subdf, facet_cols + ["gamma"], "final_val_mse")
        for keys, sub in agg.groupby(facet_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            title_bits = [f"{k}={v}" for k, v in zip(facet_cols, keys)]
            title = ", ".join(title_bits) if title_bits else "all"
            fig, ax = plt.subplots(figsize=(7, 5))
            sub = sub.sort_values("gamma")
            ax.plot(sub["gamma"], sub["mean"], marker="o", lw=1.5)
            ax.fill_between(
                sub["gamma"],
                np.maximum(sub["mean"] - 2 * sub["se"], 0),
                sub["mean"] + 2 * sub["se"],
                alpha=0.15,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$\gamma$")
            ax.set_ylabel("Final val MSE")
            ax.set_title(f"Gen loss vs gamma (n_train2={n_train2})\n{title}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = f"gen_vs_gamma__n_train2={n_train2}__" + "__".join([f"{k}={v}" for k, v in zip(facet_cols, keys)]).replace("/", "_") + ".png"
            fig.savefig(out_dir / fname.replace(".png", ".jpg"), dpi=200, bbox_inches="tight", format="jpg")
            plt.close(fig)


def plot_gen_vs_lmda(df: pd.DataFrame, out_dir: Path, gamma_values=(1e-3, 1e-2), facet_cols=("c", "active_dim_2", "pretrain_overlap")):
    needed = {"final_val_mse", "gamma", "lmda"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for gen-vs-lmda plot: {missing}")

    facet_cols = [c for c in facet_cols if c in df.columns]
    for gamma in gamma_values:
        subdf = df[np.isclose(df["gamma"], gamma)].copy()
        if subdf.empty:
            continue
        agg = mean_se(subdf, facet_cols + ["lmda"], "final_val_mse")
        for keys, sub in agg.groupby(facet_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            title_bits = [f"{k}={v}" for k, v in zip(facet_cols, keys)]
            title = ", ".join(title_bits) if title_bits else "all"
            fig, ax = plt.subplots(figsize=(7, 5))
            sub = sub.sort_values("lmda")
            ax.plot(sub["lmda"], sub["mean"], marker="o", lw=1.5)
            ax.fill_between(
                sub["lmda"],
                np.maximum(sub["mean"] - 2 * sub["se"], 0),
                sub["mean"] + 2 * sub["se"],
                alpha=0.15,
            )
            ax.set_yscale("log")
            ax.set_xlabel(r"$\lambda_{PT}$")
            ax.set_ylabel("Final val MSE")
            ax.set_title(f"Gen loss vs lambda (gamma={gamma:g})\n{title}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = f"gen_vs_lmda__gamma={gamma:g}__" + "__".join([f"{k}={v}" for k, v in zip(facet_cols, keys)]).replace("/", "_") + ".png"
            fig.savefig(out_dir / fname.replace(".png", ".jpg"), dpi=200, bbox_inches="tight", format="jpg")
            plt.close(fig)


def plot_k_r_histograms(df: pd.DataFrame, out_dir: Path, max_plots: int = 24):
    """
    Load representative `k_r_arrays_path` files and plot histograms of k and r.
    """
    if "k_r_arrays_path" not in df.columns:
        return
    df = df.dropna(subset=["k_r_arrays_path"]).copy()
    if df.empty:
        return

    # Prefer larger n_train2 and seed=0 for representativeness
    if "n_train2" in df.columns:
        df = df.sort_values(["n_train2"], ascending=False)
    if "seed" in df.columns:
        df = df.sort_values(["seed"], ascending=True)

    seen = set()
    plots = 0
    for _, row in df.iterrows():
        path = str(row["k_r_arrays_path"])
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            arr = np.load(path)
            k = arr["k"]
            r = arr["r_theory"]
            gamma = float(arr["gamma"])
        except Exception:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(np.log10(k + 1e-300), bins=60, alpha=0.8)
        axes[0].set_title(r"$\log_{10} k_i$")
        axes[0].set_xlabel(r"$\log_{10} k$")
        axes[0].set_ylabel("count")
        axes[0].grid(True, alpha=0.2)

        axes[1].hist(np.log10(r + 1e-300), bins=60, alpha=0.8)
        axes[1].set_title(r"$\log_{10} r_i$ (theory)")
        axes[1].set_xlabel(r"$\log_{10} r$")
        axes[1].set_ylabel("count")
        axes[1].grid(True, alpha=0.2)

        title = f"gamma={gamma:g}, n_train2={row.get('n_train2','NA')}, active_dim_2={row.get('active_dim_2','NA')}, pretrain_overlap={row.get('pretrain_overlap','NA')}, c={row.get('c','NA')}, lmda={row.get('lmda','NA')}"
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        fname = f"hist_k_r__{plots:03d}.png"
        fig.savefig(out_dir / fname.replace(".png", ".jpg"), dpi=200, bbox_inches="tight", format="jpg")
        plt.close(fig)
        plots += 1
        if plots >= max_plots:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", type=str, default="experiment_results_k.csv")
    p.add_argument("--filter_contains", type=str, default="data/diagonal/k_sweep/")
    p.add_argument("--out_dir", type=str, default="figures/diagonal_k_sweep")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_csv, args.filter_contains)
    if df.empty:
        print("No matching rows found. Check --results_csv / --filter_contains.")
        return

    # Make sure numeric columns are numeric
    for col in ["gamma", "c", "lmda", "n_train2", "inp_dim", "active_dim_2", "pretrain_overlap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Loaded {len(df)} rows for plotting.")

    plot_gen_vs_delta(df, out_dir)
    plot_gen_vs_gamma(df, out_dir)
    plot_gen_vs_lmda(df, out_dir)
    plot_k_r_histograms(df, out_dir)
    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()


