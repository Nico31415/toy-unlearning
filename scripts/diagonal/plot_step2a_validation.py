#!/usr/bin/env python3
"""
Generate Step 2A validation plots from cached replica curves.

This script loads cached CSV results from the validation experiments and creates
combined comparison plots for:
  - Plot A: Omega sweep (main transfer effect)
  - Plot B: PT irrelevance when omega=0
  - Plot C: BG baseline vs ptft_oracle (omega=0)
  - Plot D: a_pt sweep (Cosyne mapping test)
  - Plot E: c_values irrelevance check

Run this after the SLURM array jobs have completed.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_db(x: np.ndarray) -> np.ndarray:
    """Convert MSE to dB: 10*log10(mse + 1e-15)"""
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def load_cache(cache_dir: str, pattern_parts: dict) -> pd.DataFrame:
    """Load a cached CSV file based on pattern matching."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    
    # Build pattern to match
    required_matches = []
    for key, val in pattern_parts.items():
        if isinstance(val, float):
            required_matches.append(f"{key}={val:.4f}")
        else:
            required_matches.append(f"{key}={val}")
    
    # Find matching files
    candidates = list(cache_path.glob("replica_curve_*.csv"))
    for f in candidates:
        name = f.name
        if all(m in name for m in required_matches):
            return pd.read_csv(f)
    
    raise FileNotFoundError(
        f"No cache file found matching: {required_matches}\n"
        f"Available files: {[f.name for f in candidates[:5]]}..."
    )


def find_cache_file(cache_dir: str, pattern_parts: list) -> str:
    """Find cache file matching all pattern parts."""
    cache_path = Path(cache_dir)
    candidates = list(cache_path.glob("replica_curve_*.csv"))
    
    for f in candidates:
        name = f.name
        if all(p in name for p in pattern_parts):
            return str(f)
    
    return None


def plot_omega_sweep(cache_dir: str, output_dir: str):
    """Plot A: Omega sweep showing main transfer effect."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    omegas = [0.0, 0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(omegas)))
    
    for omega, color in zip(omegas, colors):
        # Match pattern for this omega
        pattern = [
            "teacher=ptft_oracle",
            f"om={omega:.4f}",
            "apt=1.00",
            "lpt=0.00",
            "gam=0.00",
        ]
        cache_file = find_cache_file(cache_dir, pattern)
        
        if cache_file is None:
            print(f"  WARNING: No cache file for omega={omega}")
            continue
        
        df = pd.read_csv(cache_file)
        mse_db = to_db(df["mse"].values)
        
        ax.plot(
            df["alpha"], mse_db,
            "-", linewidth=2, color=color,
            label=f"ω = {omega:.2f}"
        )
        print(f"  Loaded omega={omega}: MSE range [{df['mse'].min():.2e}, {df['mse'].max():.2e}]")
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title("Plot A: Omega Sweep (Transfer Effect)\n" + 
                 r"$\rho_{pt}=0.10$, $\rho_{ft}=0.04$, $a_{pt}=1.0$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "step2a_plotA_omega_sweep.png")
    pdf_path = os.path.join(output_dir, "step2a_plotA_omega_sweep.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Saved: {png_path}")
    return True


def plot_pt_irrelevance(cache_dir: str, output_dir: str):
    """Plot B: a_pt irrelevance when omega=0 (only a_pt varies, other PT params fixed)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # B1: baseline (a_pt=1.0, lambda_pt=0.0, gamma_reinit=0.0)
    pattern_b1 = [
        "teacher=ptft_oracle",
        "om=0.0000",
        "apt=1.00",
        "lpt=0.00",
        "gam=0.00",
    ]
    cache_b1 = find_cache_file(cache_dir, pattern_b1)
    
    # B2: only a_pt changed (a_pt=10.0, lambda_pt=0.0, gamma_reinit=0.0)
    pattern_b2 = [
        "teacher=ptft_oracle",
        "om=0.0000",
        "apt=10.00",
        "lpt=0.00",
        "gam=0.00",
    ]
    cache_b2 = find_cache_file(cache_dir, pattern_b2)
    
    if cache_b1:
        df1 = pd.read_csv(cache_b1)
        ax.plot(
            df1["alpha"], to_db(df1["mse"].values),
            "-", linewidth=2.5, color="blue",
            label="$a_{pt}$ = 1.0"
        )
        print(f"  B1 loaded: MSE range [{df1['mse'].min():.2e}, {df1['mse'].max():.2e}]")
    else:
        print("  WARNING: B1 cache not found")
    
    if cache_b2:
        df2 = pd.read_csv(cache_b2)
        ax.plot(
            df2["alpha"], to_db(df2["mse"].values),
            "--", linewidth=2.5, color="red",
            label="$a_{pt}$ = 10.0"
        )
        print(f"  B2 loaded: MSE range [{df2['mse'].min():.2e}, {df2['mse'].max():.2e}]")
    else:
        print("  WARNING: B2 cache not found")
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title("Plot B: $a_{pt}$ Irrelevance when ω=0\n" +
                 "Only $a_{pt}$ varies; curves should be nearly identical", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "step2a_plotB_pt_irrelevance.png")
    pdf_path = os.path.join(output_dir, "step2a_plotB_pt_irrelevance.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Saved: {png_path}")
    
    # Check if curves are similar
    if cache_b1 and cache_b2:
        df1 = pd.read_csv(cache_b1)
        df2 = pd.read_csv(cache_b2)
        max_diff = np.abs(df1["mse"].values - df2["mse"].values).max()
        rel_diff = max_diff / df1["mse"].mean()
        print(f"  Max absolute diff: {max_diff:.2e}, relative: {rel_diff:.2%}")
        if rel_diff > 0.05:
            print("  WARNING: Curves differ by more than 5%!")
            return False
    return True


def plot_bg_comparison(cache_dir: str, output_dir: str):
    """Plot C: BG baseline vs ptft_oracle (omega=0, a_pt=0)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # C1: BG baseline (standard BG teacher)
    pattern_c1 = [
        "teacher=bg",
        "rho=0.040000",
        "c=0.001000",
    ]
    cache_c1 = find_cache_file(cache_dir, pattern_c1)
    
    # C2: ptft_oracle with omega=0, a_pt=0, rho_pt=0.10
    pattern_c2 = [
        "teacher=ptft_oracle",
        "om=0.0000",
        "rpt=0.1000",
        "rft=0.0400",
        "apt=0.00",
        "lpt=0.00",
        "gam=0.00",
    ]
    cache_c2 = find_cache_file(cache_dir, pattern_c2)
    
    # C3: ptft_oracle with omega=0, a_pt=0, rho_pt=0.04 (matches rho_ft)
    pattern_c3 = [
        "teacher=ptft_oracle",
        "om=0.0000",
        "rpt=0.0400",
        "rft=0.0400",
        "apt=0.00",
        "lpt=0.00",
        "gam=0.00",
    ]
    cache_c3 = find_cache_file(cache_dir, pattern_c3)
    
    if cache_c1:
        df1 = pd.read_csv(cache_c1)
        ax.plot(
            df1["alpha"], to_db(df1["mse"].values),
            "-", linewidth=2.5, color="blue",
            label=r"BG baseline ($\rho=0.04$, $c=0.001$)"
        )
        print(f"  C1 (BG) loaded: MSE range [{df1['mse'].min():.2e}, {df1['mse'].max():.2e}]")
    else:
        print("  WARNING: C1 (BG) cache not found")
    
    if cache_c2:
        df2 = pd.read_csv(cache_c2)
        ax.plot(
            df2["alpha"], to_db(df2["mse"].values),
            "--", linewidth=2.5, color="red",
            label=r"ptft_oracle ($\omega=0$, $a_{pt}=0$, $\rho_{pt}=0.10$)"
        )
        print(f"  C2 (ptft rho_pt=0.10) loaded: MSE range [{df2['mse'].min():.2e}, {df2['mse'].max():.2e}]")
    else:
        print("  WARNING: C2 (ptft rho_pt=0.10) cache not found")
    
    if cache_c3:
        df3 = pd.read_csv(cache_c3)
        ax.plot(
            df3["alpha"], to_db(df3["mse"].values),
            ":", linewidth=2.5, color="green",
            label=r"ptft_oracle ($\omega=0$, $a_{pt}=0$, $\rho_{pt}=0.04$)"
        )
        print(f"  C3 (ptft rho_pt=0.04) loaded: MSE range [{df3['mse'].min():.2e}, {df3['mse'].max():.2e}]")
    else:
        print("  WARNING: C3 (ptft rho_pt=0.04) cache not found")
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title("Plot C: BG Baseline vs ptft_oracle (ω=0, $a_{pt}$=0)\n" +
                 "With $a_{pt}$=0 and matching params, curves should align", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "step2a_plotC_bg_comparison.png")
    pdf_path = os.path.join(output_dir, "step2a_plotC_bg_comparison.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Saved: {png_path}")
    return True


def plot_apt_sweep(cache_dir: str, output_dir: str):
    """Plot D: a_pt sweep (Cosyne mapping test)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    a_pts = [0.0, 0.5, 1.0, 2.0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(a_pts)))
    
    for a_pt, color in zip(a_pts, colors):
        # Match pattern for this a_pt
        pattern = [
            "teacher=ptft_oracle",
            "om=1.0000",
            f"apt={a_pt:.2f}",
            "lpt=0.00",
            "gam=0.00",
        ]
        cache_file = find_cache_file(cache_dir, pattern)
        
        if cache_file is None:
            print(f"  WARNING: No cache file for a_pt={a_pt}")
            continue
        
        df = pd.read_csv(cache_file)
        mse_db = to_db(df["mse"].values)
        
        ax.plot(
            df["alpha"], mse_db,
            "-", linewidth=2, color=color,
            label=f"$a_{{pt}}$ = {a_pt:.1f}"
        )
        print(f"  Loaded a_pt={a_pt}: MSE range [{df['mse'].min():.2e}, {df['mse'].max():.2e}]")
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title("Plot D: $a_{pt}$ Sweep (ω=1.0)\n" +
                 r"$a_{pt}=0$ should match no-transfer baseline", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "step2a_plotD_apt_sweep.png")
    pdf_path = os.path.join(output_dir, "step2a_plotD_apt_sweep.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Saved: {png_path}")
    return True


def plot_cvalues_irrelevance(cache_dir: str, output_dir: str):
    """Plot E: c_values irrelevance check."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    c_values = [0.001, 0.5]
    colors = ["blue", "red"]
    linestyles = ["-", "--"]
    
    mse_arrays = []
    for c_val, color, ls in zip(c_values, colors, linestyles):
        # Match pattern for this c value
        pattern = [
            "teacher=ptft_oracle",
            "om=0.5000",
            "apt=1.00",
            f"c={c_val:.6f}",
        ]
        cache_file = find_cache_file(cache_dir, pattern)
        
        if cache_file is None:
            print(f"  WARNING: No cache file for c={c_val}")
            continue
        
        df = pd.read_csv(cache_file)
        mse_db = to_db(df["mse"].values)
        mse_arrays.append(df["mse"].values)
        
        ax.plot(
            df["alpha"], mse_db,
            ls, linewidth=2.5, color=color,
            label=f"c_values = {c_val}"
        )
        print(f"  Loaded c={c_val}: MSE range [{df['mse'].min():.2e}, {df['mse'].max():.2e}]")
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=14)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=14)
    ax.set_title("Plot E: c_values Irrelevance Check\n" +
                 "Curves should be identical (c_values ignored in ptft_oracle)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "step2a_plotE_cvalues_irrelevance.png")
    pdf_path = os.path.join(output_dir, "step2a_plotE_cvalues_irrelevance.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Saved: {png_path}")
    
    # Check if curves are identical
    if len(mse_arrays) == 2:
        max_diff = np.abs(mse_arrays[0] - mse_arrays[1]).max()
        if max_diff > 1e-10:
            print(f"  WARNING: Curves differ! Max diff = {max_diff:.2e}")
            return False
        else:
            print(f"  PASS: Curves are identical (max diff = {max_diff:.2e})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate Step 2A validation plots")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="figures/step2a_validation/replica_cache",
        help="Directory containing cached replica curve CSVs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="figures/step2a_validation",
        help="Output directory for plots",
    )
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("STEP 2A VALIDATION PLOTS")
    print("=" * 80)
    print(f"Cache dir: {args.cache_dir}")
    print(f"Output dir: {args.output_dir}")
    
    all_passed = True
    
    print("\n--- Plot A: Omega Sweep ---")
    try:
        plot_omega_sweep(args.cache_dir, args.output_dir)
    except Exception as e:
        print(f"  ERROR: {e}")
        all_passed = False
    
    print("\n--- Plot B: PT Irrelevance (omega=0) ---")
    try:
        if not plot_pt_irrelevance(args.cache_dir, args.output_dir):
            all_passed = False
    except Exception as e:
        print(f"  ERROR: {e}")
        all_passed = False
    
    print("\n--- Plot C: BG Baseline vs ptft_oracle ---")
    try:
        plot_bg_comparison(args.cache_dir, args.output_dir)
    except Exception as e:
        print(f"  ERROR: {e}")
        all_passed = False
    
    print("\n--- Plot D: a_pt Sweep ---")
    try:
        plot_apt_sweep(args.cache_dir, args.output_dir)
    except Exception as e:
        print(f"  ERROR: {e}")
        all_passed = False
    
    print("\n--- Plot E: c_values Irrelevance ---")
    try:
        if not plot_cvalues_irrelevance(args.cache_dir, args.output_dir):
            all_passed = False
    except Exception as e:
        print(f"  ERROR: {e}")
        all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL VALIDATION CHECKS PASSED")
    else:
        print("SOME VALIDATION CHECKS FAILED - review output above")
    print("=" * 80)


if __name__ == "__main__":
    main()

