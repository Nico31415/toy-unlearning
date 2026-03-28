#!/usr/bin/env python3
"""
Plot imperfect-pretraining replica PT+FT curves (oracle config grid) as param MSE vs alpha,
overlaying different alpha_pt values on the same axes (linestyle encodes alpha_pt).

Inputs:
  results/replica_imperfect_pt_oraclegrid/master.csv
  plus the per-task CSVs referenced by master.csv "out_csv".
Optionally, include the alpha_pt=1.0 ("oracle PT") baseline from cached *replica-theory*
curves (not empirical training runs). In this repo those are available for the
rho_pt=0.1 overlap-endpoints families via:
  results/replica_ptft_parallel/*.csv

Outputs:
  figures/replica_imperfect_pt_oraclegrid/*.png and *.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def _ensure_float_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _agg_mean_std_safe(df: pd.DataFrame, group_cols: List[str], y_col: str) -> pd.DataFrame:
    g = (
        # NOTE: this repo runs on an older pandas in some environments, where
        # DataFrame.groupby does not support the `dropna=` kwarg.
        df.groupby(group_cols)[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{y_col}_mean", "std": f"{y_col}_std", "count": "n"})
    )
    # for deterministic replica curves, std is NaN (n=1); plot with zero band
    g[f"{y_col}_std"] = pd.to_numeric(g[f"{y_col}_std"], errors="coerce").fillna(0.0)
    return g


def _fmt_label(x: float) -> str:
    return f"{float(x):.6g}".replace("+", "")


def _alpha_pt_style(alpha_pt: float) -> Tuple[str, object]:
    """
    Return (label, linestyle) for alpha_pt. Linestyle can be a string or a dash tuple.
    """
    a = float(alpha_pt)
    if np.isclose(a, 1.0):
        return "alpha_pt=1.0", "-"
    if np.isclose(a, 0.5):
        return "alpha_pt=0.5", "--"
    if np.isclose(a, 0.1):
        return "alpha_pt=0.1", ":"  # dotted
    if np.isclose(a, 0.01):
        # "more dotted": short on/off pattern
        return "alpha_pt=0.01", (0, (0.8, 0.8))
    return f"alpha_pt={_fmt_label(a)}", "-"


def _plot_overlay_alpha_pt(
    *,
    df: pd.DataFrame,
    base_mask: pd.Series,
    hue_col: str,
    title: str,
    out_base: Path,
    metric: str,
    y_label: str,
    yscale: str,
    alpha_pt_order_preferred: List[float],
    drop_hue_values: Optional[List[float]] = None,
) -> None:
    d0 = df[base_mask].copy()
    if d0.empty:
        return

    # Aggregate (handles any duplication / chunking)
    g = (
        d0.groupby(["alpha", "alpha_pt", hue_col])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "y"})
    )

    if drop_hue_values:
        drop = [float(x) for x in drop_hue_values]
        keep = np.ones(len(g), dtype=bool)
        for v in drop:
            keep &= ~np.isclose(g[hue_col].to_numpy(dtype=float), float(v))
        g = g[keep].copy()

    # Orders
    hue_vals = sorted(g[hue_col].dropna().unique().tolist(), key=float)
    alpha_pts_present = sorted(g["alpha_pt"].dropna().unique().tolist(), key=float)
    alpha_pts = [a for a in alpha_pt_order_preferred if any(np.isclose(a, b) for b in alpha_pts_present)]
    # include any unexpected alpha_pt values at the end
    for a in alpha_pts_present:
        if not any(np.isclose(a, b) for b in alpha_pts):
            alpha_pts.append(float(a))

    # Make the plot wide; legends can get large for sweeps.
    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    colors = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    if len(colors) < len(hue_vals):
        colors = colors + ["C%d" % i for i in range(len(hue_vals))]

    color_handles: List[Line2D] = []
    style_handles: List[Line2D] = []

    # Plot: color = hue, linestyle = alpha_pt
    for i, h in enumerate(hue_vals):
        color = colors[i]
        color_label = f"{hue_col}={_fmt_label(h)}" if isinstance(h, (int, float, np.floating)) else str(h)
        color_handles.append(Line2D([0], [0], color=color, linewidth=2.5, linestyle="-", label=color_label))

        for a_pt in alpha_pts:
            m = (np.isclose(g[hue_col], h)) & (np.isclose(g["alpha_pt"], a_pt))
            d = g[m].sort_values("alpha")
            if d.empty:
                continue
            _, ls = _alpha_pt_style(float(a_pt))
            ax.plot(
                d["alpha"].to_numpy(),
                d["y"].to_numpy(),
                color=color,
                linestyle=ls,
                linewidth=2.2,
                alpha=0.95,
            )

    # Linestyle legend (alpha_pt)
    seen = set()
    for a_pt in alpha_pts:
        lbl, ls = _alpha_pt_style(float(a_pt))
        if lbl in seen:
            continue
        seen.add(lbl)
        style_handles.append(Line2D([0], [0], color="black", linewidth=2.2, linestyle=ls, label=lbl))

    ax.set_title(title)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.set_yscale(yscale)

    # Put legends outside on the right so the data region stays wide.
    ax.set_title(title)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.set_yscale(yscale)

    ax.figure.subplots_adjust(right=0.72)

    leg1 = ax.legend(
        handles=color_handles,
        title=hue_col,
        fontsize=9,
        title_fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )
    leg2 = ax.legend(
        handles=style_handles,
        title="alpha_pt (linestyle)",
        fontsize=9,
        title_fontsize=10,
        loc="lower left",
        bbox_to_anchor=(1.01, 0.0),
        frameon=True,
    )
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(str(out_base) + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(Path(str(out_base) + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def _read_master_and_tasks(in_dir: Path) -> pd.DataFrame:
    master_csv = in_dir / "master.csv"
    if not master_csv.exists():
        raise SystemExit(f"Missing master.csv at {master_csv}")

    m = pd.read_csv(master_csv)
    if "status" in m.columns:
        m = m[m["status"] == "ok"].copy()

    if "out_csv" not in m.columns:
        raise SystemExit("master.csv missing required column 'out_csv'")

    dfs = []
    for s in m["out_csv"].astype(str).tolist():
        fp = Path(s)
        if not fp.exists():
            # common case: script invoked from a different cwd
            fp = (Path.cwd() / fp).resolve()
        if not fp.exists():
            # last attempt: interpret relative to in_dir parent (repo root-ish)
            fp2 = (in_dir.parent / Path(s)).resolve()
            if fp2.exists():
                fp = fp2
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        dfs.append(df)

    if not dfs:
        raise SystemExit("No per-task CSVs loaded (check master.csv out_csv paths).")

    return pd.concat(dfs, ignore_index=True)


def _read_oracle_replica_baseline_from_dir(oracle_replica_dir: Path) -> pd.DataFrame:
    """
    Load cached oracle (alpha_pt=1) replica theory curves from a directory of CSVs.
    Expected schema includes: rho_pt,rho_ft,omega,c_pt,lambda_pt,gamma_reinit,alpha,mse_best,...
    """
    if not oracle_replica_dir.exists():
        raise SystemExit(f"Oracle replica dir not found: {oracle_replica_dir}")

    files = sorted([p for p in oracle_replica_dir.glob("*.csv")])
    if not files:
        raise SystemExit(f"No CSVs found in oracle replica dir: {oracle_replica_dir}")

    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.copy()
        df["alpha_pt"] = 1.0
        dfs.append(df)

    if not dfs:
        raise SystemExit(f"Failed to read any oracle replica CSVs from: {oracle_replica_dir}")

    return pd.concat(dfs, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot imperfect-pretraining replica curves (oracle grid) vs alpha, overlay alpha_pt linestyles"
    )
    p.add_argument("--in_dir", type=str, default="results/replica_imperfect_pt_oraclegrid")
    p.add_argument("--out_dir", type=str, default="figures/replica_imperfect_pt_oraclegrid")
    p.add_argument(
        "--oracle_replica_dir",
        type=str,
        default="results/replica_ptft_parallel",
        help="Directory of cached oracle (alpha_pt=1) replica CSV curves to overlay.",
    )
    p.add_argument("--no_oracle", action="store_true", help="Do not overlay alpha_pt=1 oracle baseline.")
    p.add_argument("--metric", type=str, default="mse_best", choices=["mse_best", "mse_fwd", "mse_bwd"])
    p.add_argument("--db", action="store_true", help="Plot in dB: 10*log10(metric + 1e-15).")
    p.add_argument("--yscale", type=str, default="linear", choices=["linear", "log"])
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    metric = args.metric

    df = _read_master_and_tasks(in_dir)
    if not bool(args.no_oracle):
        # IMPORTANT: this is a *replica-theory* oracle baseline (alpha_pt=1), not empirical runs.
        df_orc = _read_oracle_replica_baseline_from_dir(Path(args.oracle_replica_dir))
        df = pd.concat([df, df_orc], ignore_index=True)

    df = _ensure_float_cols(
        df,
        cols=[
            "alpha",
            "alpha_pt",
            metric,
            "rho_pt",
            "rho_ft",
            "omega",
            "c_pt",
            "lambda_pt",
            "gamma_reinit",
        ],
    )

    if metric not in df.columns:
        raise SystemExit(f"Missing metric column '{metric}' in task CSVs.")

    if args.db:
        df = df.copy()
        df[metric] = 10.0 * np.log10(np.maximum(df[metric].to_numpy(), 1e-15))
        y_label = f"{metric} (dB)"
    else:
        y_label = metric

    alpha_pt_order_preferred = [1.0, 0.5, 0.1, 0.01]
    if df["alpha_pt"].dropna().empty:
        raise SystemExit("No alpha_pt values found in loaded CSVs.")

    # -------------------------
    # Families (mirror scripts/diagonal/plot_custom_oracle_sweep_results.py)
    # -------------------------

    # PT+FT dense full overlap: rho_pt=0.9999, omega=1
    base_dense = np.isclose(df["rho_pt"], 0.9999) & np.isclose(df["omega"], 1.0)

    # (2) lambda sweep at rho_ft=0.1, c_pt=0.001, gamma=0
    mask = (
        base_dense
        & np.isclose(df["rho_ft"], 0.1)
        & np.isclose(df["c_pt"], 0.001)
        & np.isclose(df["gamma_reinit"], 0.0)
    )
    _plot_overlay_alpha_pt(
        df=df,
        base_mask=mask,
        hue_col="lambda_pt",
        metric=metric,
        y_label=y_label,
        yscale=args.yscale,
        title=f"Imperfect-PT replica: {y_label} vs alpha — full overlap (rho_pt=0.9999, rho_ft=0.1, omega=1, c_pt=0.001, gamma=0) — lambda sweep",
        out_base=out_dir / "ptft_full__rho_pt=0.9999__rho_ft=0.1__cpt=0.001__gamma=0__lambda_sweep__overlay_alpha_pt",
        alpha_pt_order_preferred=alpha_pt_order_preferred,
        drop_hue_values=[-0.0005, 0.0005],
    )

    # (3) gamma sweep at rho_ft=0.1, c_pt=0.001, lambda=0
    mask = (
        base_dense
        & np.isclose(df["rho_ft"], 0.1)
        & np.isclose(df["c_pt"], 0.001)
        & np.isclose(df["lambda_pt"], 0.0)
    )
    _plot_overlay_alpha_pt(
        df=df,
        base_mask=mask,
        hue_col="gamma_reinit",
        metric=metric,
        y_label=y_label,
        yscale=args.yscale,
        title=f"Imperfect-PT replica: {y_label} vs alpha — full overlap (rho_pt=0.9999, rho_ft=0.1, omega=1, c_pt=0.001, lambda=0) — gamma sweep",
        out_base=out_dir / "ptft_full__rho_pt=0.9999__rho_ft=0.1__cpt=0.001__lambda=0__gamma_sweep__overlay_alpha_pt",
        alpha_pt_order_preferred=alpha_pt_order_preferred,
    )

    # (4) c_pt sweep at rho_ft=0.1, lambda=0, gamma=0
    mask = (
        base_dense
        & np.isclose(df["rho_ft"], 0.1)
        & np.isclose(df["lambda_pt"], 0.0)
        & np.isclose(df["gamma_reinit"], 0.0)
    )
    _plot_overlay_alpha_pt(
        df=df,
        base_mask=mask,
        hue_col="c_pt",
        metric=metric,
        y_label=y_label,
        yscale=args.yscale,
        title=f"Imperfect-PT replica: {y_label} vs alpha — full overlap (rho_pt=0.9999, rho_ft=0.1, omega=1, lambda=0, gamma=0) — c_pt sweep",
        out_base=out_dir / "ptft_full__rho_pt=0.9999__rho_ft=0.1__lambda=0__gamma=0__cpt_sweep__overlay_alpha_pt",
        alpha_pt_order_preferred=alpha_pt_order_preferred,
        drop_hue_values=[0.1],
    )

    # (5) lambda sweep at rho_ft=0.9, c_pt=0.001, gamma=0
    mask = (
        base_dense
        & np.isclose(df["rho_ft"], 0.9)
        & np.isclose(df["c_pt"], 0.001)
        & np.isclose(df["gamma_reinit"], 0.0)
    )
    _plot_overlay_alpha_pt(
        df=df,
        base_mask=mask,
        hue_col="lambda_pt",
        metric=metric,
        y_label=y_label,
        yscale=args.yscale,
        title=f"Imperfect-PT replica: {y_label} vs alpha — full overlap (rho_pt=0.9999, rho_ft=0.9, omega=1, c_pt=0.001, gamma=0) — lambda sweep",
        out_base=out_dir / "ptft_full__rho_pt=0.9999__rho_ft=0.9__cpt=0.001__gamma=0__lambda_sweep__overlay_alpha_pt",
        alpha_pt_order_preferred=alpha_pt_order_preferred,
        drop_hue_values=[-0.0005, 0.0005],
    )

    # PT+FT sparse PT overlap endpoints: rho_pt=0.1, rho_ft=0.1, c_pt=0.001, omega in {0,1}
    base_sparse = np.isclose(df["rho_pt"], 0.1) & np.isclose(df["rho_ft"], 0.1) & np.isclose(df["c_pt"], 0.001)

    # (7) omega endpoints × lambda sweep, gamma=0
    mask = base_sparse & np.isclose(df["gamma_reinit"], 0.0) & df["omega"].isin([0.0, 1.0])
    _plot_overlay_alpha_pt(
        df=df,
        base_mask=mask,
        hue_col="lambda_pt",
        metric=metric,
        y_label=y_label,
        yscale=args.yscale,
        title=f"Imperfect-PT replica: {y_label} vs alpha — overlap endpoints (rho_pt=0.1, rho_ft=0.1, omega∈{{0,1}}, c_pt=0.001, gamma=0) — lambda sweep",
        out_base=out_dir / "ptft_overlap__rho_pt=0.1__rho_ft=0.1__omega01__cpt=0.001__gamma=0__lambda_sweep__overlay_alpha_pt",
        alpha_pt_order_preferred=alpha_pt_order_preferred,
        drop_hue_values=[-0.0005, 0.0005],
    )

    # (8) omega endpoints × gamma sweep, lambda=0
    mask = base_sparse & np.isclose(df["lambda_pt"], 0.0) & df["omega"].isin([0.0, 1.0])
    _plot_overlay_alpha_pt(
        df=df,
        base_mask=mask,
        hue_col="gamma_reinit",
        metric=metric,
        y_label=y_label,
        yscale=args.yscale,
        title=f"Imperfect-PT replica: {y_label} vs alpha — overlap endpoints (rho_pt=0.1, rho_ft=0.1, omega∈{{0,1}}, c_pt=0.001, lambda=0) — gamma sweep",
        out_base=out_dir / "ptft_overlap__rho_pt=0.1__rho_ft=0.1__omega01__cpt=0.001__lambda=0__gamma_sweep__overlay_alpha_pt",
        alpha_pt_order_preferred=alpha_pt_order_preferred,
    )

    print(f"Wrote plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

