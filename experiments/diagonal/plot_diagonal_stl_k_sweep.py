#!/usr/bin/env python3
"""
Plotting for *single-task* diagonal STL k-sweep.

Reads `experiment_results_st_k.csv` (written by diagonal_network_pretrain_k_runner.py postprocess)
and produces:
  - Gen loss vs delta=n/d for each (c,lmda) and each teacher sparsity
  - Gen loss vs c (proxy for sqrt(k)) at fixed n and lmda_frac
  - Histograms of r for representative runs (sanity check)
"""

import argparse
import os
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


def plot_gen_vs_delta(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    # delta = n_train / inp_dim
    if "inp_dim" not in df.columns:
        df["inp_dim"] = 1000
    df["delta"] = df["n_train"] / df["inp_dim"]

    facet_cols = [c for c in ["active_dim", "c", "lmda"] if c in df.columns]
    agg = mean_se(df, facet_cols + ["delta"], "final_val_mse")
    for keys, sub in agg.groupby(facet_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        title = ", ".join([f"{k}={v}" for k, v in zip(facet_cols, keys)])
        sub = sub.sort_values("delta")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sub["delta"], sub["mean"], marker="o", lw=1.5)
        ax.fill_between(
            sub["delta"],
            np.maximum(sub["mean"] - 2 * sub["se"], 0),
            sub["mean"] + 2 * sub["se"],
            alpha=0.15,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\delta = n/d$")
        ax.set_ylabel("Final val MSE")
        ax.set_title(f"STL gen vs delta\n{title}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = "stl_gen_vs_delta__" + "__".join([f"{k}={v}" for k, v in zip(facet_cols, keys)]).replace("/", "_") + ".png"
        # Save as JPG for headless/lab machines
        fig.savefig(out_dir / fname.replace(".png", ".jpg"), dpi=200, bbox_inches="tight", format="jpg")
        plt.close(fig)


def plot_val_vs_c(df: pd.DataFrame, out_dir: Path, n_train_values=(64, 256, 1024)):
    if "c" not in df.columns:
        return
    for n in n_train_values:
        subdf = df[df["n_train"] == n].copy()
        if subdf.empty:
            continue
        facet_cols = [c for c in ["active_dim", "lmda"] if c in subdf.columns]
        agg = mean_se(subdf, facet_cols + ["c"], "final_val_mse")
        for keys, sub in agg.groupby(facet_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            title = ", ".join([f"{k}={v}" for k, v in zip(facet_cols, keys)])
            sub = sub.sort_values("c")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(sub["c"], sub["mean"], marker="o", lw=1.5)
            ax.fill_between(
                sub["c"],
                np.maximum(sub["mean"] - 2 * sub["se"], 0),
                sub["mean"] + 2 * sub["se"],
                alpha=0.15,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("c (proxy for sqrt(k))")
            ax.set_ylabel("Final val MSE")
            ax.set_title(f"STL gen vs c (n_train={n})\n{title}")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = f"stl_gen_vs_c__n_train={n}__" + "__".join([f"{k}={v}" for k, v in zip(facet_cols, keys)]).replace("/", "_") + ".png"
            fig.savefig(out_dir / fname.replace(".png", ".jpg"), dpi=200, bbox_inches="tight", format="jpg")
            plt.close(fig)


def plot_r_histograms(df: pd.DataFrame, out_dir: Path, max_plots: int = 24):
    if "k_r_arrays_path" not in df.columns:
        return
    df = df.dropna(subset=["k_r_arrays_path"]).copy()
    if df.empty:
        return

    # pick representative runs (largest n_train, seed small)
    if "n_train" in df.columns:
        df = df.sort_values(["n_train"], ascending=False)
    if "seed" in df.columns:
        df = df.sort_values(["seed"], ascending=True)

    seen = set()
    plots = 0
    for _, row in df.iterrows():
        p = str(row["k_r_arrays_path"])
        if p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        arr = np.load(p)
        r = arr["r_theory"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log10(r + 1e-300), bins=60, alpha=0.85)
        ax.set_title("STL histogram of log10 r_i")
        ax.set_xlabel("log10 r")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"stl_hist_r__{plots:03d}.jpg", dpi=200, bbox_inches="tight", format="jpg")
        plt.close(fig)
        plots += 1
        if plots >= max_plots:
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_csv", type=str, default="experiment_results_st_k.csv")
    p.add_argument("--filter_contains", type=str, default="data/diagonal/stl_k_sweep/")
    p.add_argument("--out_dir", type=str, default="figures/diagonal_stl_k_sweep")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    if args.filter_contains:
        df = df[df["save_folder"].astype(str).str.contains(args.filter_contains, na=False)].copy()
    if df.empty:
        print("No matching rows found.")
        return

    for col in ["seed", "active_dim", "n_train", "inp_dim", "c", "lmda", "final_val_mse"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_gen_vs_delta(df, out_dir)
    plot_val_vs_c(df, out_dir)
    plot_r_histograms(df, out_dir)
    print(f"Wrote STL plots to {out_dir}")


if __name__ == "__main__":
    main()


