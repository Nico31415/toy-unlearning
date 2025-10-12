#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

CSV_PATH = '/home/na658/multi-task2/experiment_results/experiment_results.csv'
OUT_DIR = '/home/na658/multi-task2'

# Targets requested
W_SCALING_TARGETS = [0.01, 1.0]
LMDA_TARGETS = [-1e-05, 0.0]


def _resolve_w_scaling_column(df: pd.DataFrame):
    """Return column name to use for w_scaling-like filtering and a note.
    Preference order: 'w_scaling' -> 'model_scaling' -> 'scaling'.
    """
    for col in ['w_scaling', 'model_scaling', 'scaling']:
        if col in df.columns:
            return col
    return None


def _ensure_w_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a 'w_scaling' column in the dataframe.
    If missing, derive from 'w_scaling'->'model_scaling'->'scaling' in that order.
    Returns a dataframe view with 'w_scaling'.
    """
    if 'w_scaling' in df.columns:
        return df
    if 'model_scaling' in df.columns:
        df = df.copy()
        df['w_scaling'] = df['model_scaling']
        return df
    if 'scaling' in df.columns:
        df = df.copy()
        df['w_scaling'] = df['scaling']
        return df
    # Nothing available; create empty column for compatibility
    df = df.copy()
    df['w_scaling'] = np.nan
    return df


def _group_mean_over_other_params(df: pd.DataFrame) -> pd.DataFrame:
    """Average validation loss across all parameters except n_train2.
    Returns a dataframe indexed by n_train2 with columns mean_val and count.
    """
    if 'n_train2' not in df.columns or 'final_val_loss' not in df.columns:
        return pd.DataFrame(columns=['n_train2', 'mean_val', 'std_val', 'count'])
    df_valid = df[['n_train2', 'final_val_loss']].dropna()
    if df_valid.empty:
        return pd.DataFrame(columns=['n_train2', 'mean_val', 'std_val', 'count'])
    grouped = (df_valid
               .groupby('n_train2')
               .agg(mean_val=('final_val_loss', 'mean'),
                    std_val=('final_val_loss', 'std'),
                    count=('final_val_loss', 'size'))
               .reset_index())
    return grouped.sort_values('n_train2')


def _group_mean_by_active_dim(df: pd.DataFrame) -> dict:
    """Group by active_dim_2 and n_train2, compute mean curves.
    Returns a dict active_dim_2 -> grouped dataframe with n_train2, mean_val.
    """
    curves = {}
    if 'active_dim_2' not in df.columns:
        return curves
    for ad in [5, 40]:
        sub = df[df['active_dim_2'] == ad]
        if sub.empty:
            continue
        curves[ad] = _group_mean_over_other_params(sub)
    return curves


def _compute_shared_limits(dfs):
    """Compute shared log-scale limits across multiple grouped dfs.
    dfs: iterable of dataframes each with columns n_train2 and mean_val
    Returns (x_min, x_max, y_min, y_max) with small safety margins.
    """
    x_vals = []
    y_vals = []
    for d in dfs:
        if d is None or d.empty:
            continue
        x_vals.append(d['n_train2'].values)
        y_vals.append(d['mean_val'].values)
    if not x_vals or not y_vals:
        return None
    x_all = np.concatenate(x_vals)
    y_all = np.concatenate(y_vals)
    # Remove non-positive for log
    x_all = x_all[x_all > 0]
    y_all = y_all[y_all > 0]
    if x_all.size == 0 or y_all.size == 0:
        return None
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    # add small margins on log-scale
    x_min /= 1.1
    x_max *= 1.1
    y_min /= 1.1
    y_max *= 1.1
    # Force x-axis to start at 10^0 for comparability
    x_min = max(1.0, x_min)
    return (x_min, x_max, y_min, y_max)


def _plot_multi_curves(ax, curves: dict, label_fmt: str, shared_limits):
    """Plot multiple active_dim_2 curves on an axis with legend."""
    has_any = False
    for ad, g in sorted(curves.items()):
        if g is None or g.empty:
            continue
        ax.plot(g['n_train2'], g['mean_val'], marker='o', linewidth=2, label=label_fmt.format(active_dim_2=ad))
        has_any = True
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training samples (n_train2)')
    ax.set_ylabel('Validation loss (mean)')
    ax.grid(True, alpha=0.3)
    if shared_limits is not None:
        x_min, x_max, y_min, y_max = shared_limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    if has_any:
        ax.legend(title='active_dim_2')
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')


def _plot_grouped(df_grouped: pd.DataFrame, title: str, filename: str, shared_limits):
    # This function now expects df_grouped to be a dict of active_dim_2 -> grouped df
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_multi_curves(ax, df_grouped, label_fmt='active_dim_2={active_dim_2}', shared_limits=shared_limits)
    ax.set_title(title)
    out_path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _build_curves_from_csv_by_active_dim(df_csv: pd.DataFrame) -> dict:
    """Group csv data by active_dim_2 and n_train2 to produce curves.
    Returns dict active_dim_2 -> grouped df with columns n_train2 and mean_val.
    """
    curves = {}
    if 'active_dim_2' not in df_csv.columns:
        return curves
    for ad in sorted(df_csv['active_dim_2'].dropna().unique().tolist()):
        sub = df_csv[df_csv['active_dim_2'] == ad]
        if sub.empty:
            continue
        grouped = _group_mean_over_other_params(sub)
        if grouped is None or grouped.empty:
            continue
        curves[int(ad)] = grouped
    return curves


def create_per_w_scaling_figures_from_csv(df_csv: pd.DataFrame, out_dir: str) -> list:
    """Create separate figures for each w_scaling value found in CSV.
    Each figure shows mean final_val_loss vs n_train2, with a line per active_dim_2.
    Returns list of output paths.
    """
    outputs = []
    df_csv = _ensure_w_scaling(df_csv)
    unique_ws = sorted([float(x) for x in df_csv['w_scaling'].dropna().astype(float).unique().tolist()])
    if not unique_ws:
        return outputs

    # Pre-compute shared limits across all w_scaling panels for comparability
    all_grouped = []
    per_w_curves = {}
    for w in unique_ws:
        sub = df_csv[np.isclose(df_csv['w_scaling'].astype(float), float(w))]
        curves = _build_curves_from_csv_by_active_dim(sub)
        curves = _normalize_curves_to_first_point(curves)
        per_w_curves[w] = curves
        for g in curves.values():
            if g is not None and not g.empty:
                all_grouped.append(g)
    shared_limits = _compute_shared_limits(all_grouped)

    for w in unique_ws:
        curves = per_w_curves[w]
        fig, ax = plt.subplots(figsize=(8, 6))
        _plot_multi_curves(ax, curves, label_fmt='active_dim_2={active_dim_2}', shared_limits=shared_limits)
        ax.set_title(f'Validation loss vs n_train2 (w_scaling={w}) — normalized')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_path = os.path.join(out_dir, f'val_vs_n_train2_w_scaling_{w}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        outputs.append(out_path)
    return outputs

def load_scaling_feather(path: str) -> pd.DataFrame:
    """Load the processed diagonal scaling feather into a pandas DataFrame."""
    return pd.read_feather(path)


def _normalize_curves_to_first_point(curves: Dict[int, pd.DataFrame]) -> Dict[int, pd.DataFrame]:
    """Normalize each curve so its value at n_train2==16 (or smallest x) becomes 1.0.
    Operates on a dict active_dim_2 -> grouped df with columns n_train2, mean_val.
    Returns a new dict with normalized copies.
    """
    normed: Dict[int, pd.DataFrame] = {}
    for ad, g in curves.items():
        if g is None or g.empty:
            normed[ad] = g
            continue
        g2 = g.copy()
        # Select baseline at n_train2==16 if present, else use the minimum n_train2
        if (g2['n_train2'] == 16).any():
            base_val = g2.loc[g2['n_train2'] == 16, 'mean_val'].iloc[0]
        else:
            idx_min = g2['n_train2'].idxmin()
            base_val = g2.loc[idx_min, 'mean_val']
        if base_val is None or not np.isfinite(base_val) or base_val == 0:
            normed[ad] = g2
            continue
        g2['mean_val'] = g2['mean_val'] / float(base_val)
        normed[ad] = g2
    return normed

# Update builders to apply normalization before returning
def build_curves_from_scaling(df_scale: pd.DataFrame, w_value: float, overlap_filter: str = 'yes') -> Dict[int, pd.DataFrame]:
    """From scaling feather df, build curves for all available active_dims for given w_scaling.
    overlap_filter: 'yes' or 'no'
    Returns dict active_dim -> grouped df with columns n_train2 and mean_val.
    """
    dfv = df_scale.copy()
    if 'split' in dfv.columns:
        dfv = dfv[dfv['split'] == 'val']
    if 'setup' in dfv.columns:
        dfv = dfv[dfv['setup'] == 'PT+FT']
    if 'overlap' in dfv.columns and overlap_filter in ('yes', 'no'):
        dfv = dfv[dfv['overlap'] == overlap_filter]
    dfv = dfv[np.isclose(dfv['w_scaling'].astype(float), float(w_value))]
    curves: Dict[int, pd.DataFrame] = {}
    if dfv.empty:
        return curves
    for ad in sorted(dfv['active_dims'].unique().tolist()):
        sub = dfv[dfv['active_dims'] == ad]
        if sub.empty:
            continue
        grouped = (sub.groupby('n_train', as_index=False)
                     .agg(mean_val=('loss', 'mean'),
                          std_val=('loss', 'std'),
                          count=('loss', 'size')))
        grouped = grouped.rename(columns={'n_train': 'n_train2'})
        grouped = grouped.sort_values('n_train2')
        curves[int(ad)] = grouped
    return _normalize_curves_to_first_point(curves)


def create_combined_3x2_from_sources(df_csv: pd.DataFrame, df_scale: pd.DataFrame, out_path: str) -> str:
    """Create combined 3x2: row1 w_scaling (overlap=yes), row2 w_scaling (overlap=no), row3 lmda."""
    panels = []
    # Row 1: overlap=yes, w_scaling 0.01 and 1.0
    panels.append(("overlap=yes, w_scaling=0.01", build_curves_from_scaling(df_scale, 1e-2, overlap_filter='yes')))
    panels.append(("overlap=yes, w_scaling=1.0", build_curves_from_scaling(df_scale, 1.0, overlap_filter='yes')))
    # Row 2: overlap=no, w_scaling 0.01 and 1.0
    panels.append(("overlap=no, w_scaling=0.01", build_curves_from_scaling(df_scale, 1e-2, overlap_filter='no')))
    panels.append(("overlap=no, w_scaling=1.0", build_curves_from_scaling(df_scale, 1.0, overlap_filter='no')))
    # Row 3: lmda panels from csv, keep overlapping only (>0)
    df_csv2 = _ensure_w_scaling(df_csv)
    if 'overlap' in df_csv2.columns:
        try:
            df_csv2 = df_csv2[df_csv2['overlap'].astype(float) > 0]
        except Exception:
            pass
    for lam in [-1e-5, 0.0]:
        sub = df_csv2[np.isclose(df_csv2['lmda'].astype(float), float(lam))]
        curves = _group_mean_by_active_dim(sub)
        curves = _normalize_curves_to_first_point(curves)
        panels.append((f"overlap=yes, lmda={lam}", curves))

    # Shared limits across all curves
    all_grouped = []
    for (_, curves) in panels:
        for g in curves.values():
            if g is not None and not g.empty:
                all_grouped.append(g)
    shared_limits = _compute_shared_limits(all_grouped)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    for ax, (title, curves) in zip(axes.flat, panels):
        ax.set_title(title)
        _plot_multi_curves(ax, curves, label_fmt='active_dim_2={active_dim_2}', shared_limits=shared_limits)

    fig.suptitle('Validation loss vs training samples (log-log) — normalized at first point', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    Path(os.path.dirname(out_path) or '.').mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)

    required_cols = {'n_train2', 'final_val_loss', 'lmda'}
    if not required_cols.issubset(set(df.columns)):
        print(f"Missing required columns in CSV. Found: {list(df.columns)}")
        return

    # Ensure we have w_scaling available for CSV-derived panels
    df = _ensure_w_scaling(df)

    # Load feather for w_scaling panels
    feather_path = '/home/na658/multi-task2/data/processed/diagonal/df_diagonal_scaling.feather'
    if not os.path.exists(feather_path):
        print(f"Feather not found: {feather_path}")
        return
    df_scale = load_scaling_feather(feather_path)

    # Create combined 3x2 plot from both sources
    combined_out_path = os.path.join(OUT_DIR, 'combined_scaling_lambda_plots.png')
    create_combined_3x2_from_sources(df, df_scale, combined_out_path)
    print(f"Saved combined plot to: {combined_out_path}")

    # Also create separate figures per w_scaling directly from CSV
    per_w_paths = create_per_w_scaling_figures_from_csv(df, OUT_DIR)
    if per_w_paths:
        for p in per_w_paths:
            print(f"Saved per-w_scaling plot to: {p}")
    else:
        print("No per-w_scaling plots created (missing data).")


if __name__ == '__main__':
    main()
