#!/usr/bin/env python3
"""
Step 0 Validation: Overlay empirical vs replica curves for golden baseline.

This script:
1. Loads existing empirical results for c=0.001 and c=0.5 at rho=0.04
2. Generates/loads replica curves with matching parameters
3. Creates overlay plots for validation
4. Outputs acceptance criteria checks

Usage:
    python scripts/diagonal/plot_step0_validation.py --output_dir figures/validation_step0
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

# Import replica curve generation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ReplicaExperiments.fixed_lambda_all import (
    Config,
    gamma_ext_for_q_small,
    gamma_ext_for_q_big,
    solve_rspmap_qk_curve_best_of_forward_backward,
    sample_bg,
)


def to_db(x: np.ndarray) -> np.ndarray:
    """Convert MSE to dB: 10*log10(mse + 1e-15)"""
    return 10.0 * np.log10(np.maximum(x, 1e-15))


def build_config(
    rho: float,
    sigma0_2: float,
    beta_min: float,
    beta_max: float,
    beta_points: int,
    max_fp_iters: int = 900,
    tol_fp: float = 1e-10,
    damp: float = 0.25,
) -> Config:
    """Build Config for replica theory computation."""
    var_nonzero = 1.0 / rho
    betas = np.linspace(beta_min, beta_max, beta_points)
    
    return Config(
        rho=rho,
        var_nonzero=var_nonzero,
        sigma0_2=sigma0_2,
        betas=betas,
        max_fp_iters=max_fp_iters,
        tol_fp=tol_fp,
        damp=damp,
    )


def compute_replica_curve(
    c: float,
    ft_regulariser_scale: float,
    alpha_range: np.ndarray,
    cfg: Config,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
) -> tuple:
    """Compute replica theory curve for given c value."""
    # Compute k_q from c: k = (2c)^2
    k_q = (2.0 * c) ** 2
    
    # Compute gamma_ext
    if k_q < 1.0:
        gamma_ext = gamma_ext_for_q_small(ft_regulariser_scale, k_q)
    else:
        gamma_ext = gamma_ext_for_q_big(ft_regulariser_scale, k_q)
    
    print(f"  c={c:.6f}, k_q={k_q:.6e}, gamma_ext={gamma_ext:.6e}")
    
    # Convert alpha to beta (beta = 1/alpha)
    alpha_reversed = alpha_range[::-1]
    beta_range = 1.0 / alpha_reversed
    
    # Compute replica curve
    mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
        beta_range, gamma_ext, k_q, x_mc, v_mc, cfg
    )
    
    # Reverse to match alpha ordering
    mse_alpha = mse_beta[::-1]
    
    return alpha_range, mse_alpha


def load_empirical_results(csv_path: str) -> pd.DataFrame:
    """Load aggregated empirical results."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Empirical results not found: {csv_path}")
    return pd.read_csv(csv_path)


def check_acceptance_criteria(empirical_dfs: dict, replica_curves: dict, rho: float) -> dict:
    """Check Step 0 acceptance criteria."""
    results = {
        "stable_across_seeds": {},
        "qualitative_match": {},
        "no_undertraining": {},
    }
    
    for c_str, df in empirical_dfs.items():
        c_val = float(c_str)
        
        # 1. Check stability: IQR should be reasonable compared to mean
        if "param_mse_q25" in df.columns and "param_mse_q75" in df.columns:
            iqr = df["param_mse_q75"] - df["param_mse_q25"]
            mean_val = df["param_mse_mean"]
            # Coefficient of variation proxy
            cv_proxy = (iqr / mean_val.replace(0, np.nan)).mean()
            stable = cv_proxy < 1.0  # IQR less than mean on average
            results["stable_across_seeds"][c_str] = {
                "pass": bool(stable),
                "cv_proxy": float(cv_proxy) if np.isfinite(cv_proxy) else None
            }
        
        # 2. Check no undertraining: train error << test error
        if "train_pred_mse_median" in df.columns and "test_pred_mse_median" in df.columns:
            # For alpha >= 0.2, train MSE should be << test MSE
            mask = df["alpha"] >= 0.2
            if mask.sum() > 0:
                train_mse = df.loc[mask, "train_pred_mse_median"].values
                test_mse = df.loc[mask, "test_pred_mse_median"].values
                # Train should be at least 100x smaller than test for well-trained models
                ratio = train_mse / test_mse
                max_ratio = ratio.max() if len(ratio) > 0 else np.nan
                no_undertraining = max_ratio < 0.01 if np.isfinite(max_ratio) else False
                results["no_undertraining"][c_str] = {
                    "pass": bool(no_undertraining),
                    "max_train_test_ratio": float(max_ratio) if np.isfinite(max_ratio) else None
                }
        
        # 3. Check qualitative match with replica
        if c_str in replica_curves:
            alpha_rep, mse_rep = replica_curves[c_str]
            # Interpolate replica to empirical alpha values
            emp_alpha = df["alpha"].values
            emp_mse = df["param_mse_mean"].values
            
            # Simple check: correlation of log MSE
            valid = np.isfinite(emp_mse) & (emp_mse > 0)
            if valid.sum() > 3:
                emp_log = np.log10(emp_mse[valid])
                rep_interp = np.interp(emp_alpha[valid], alpha_rep, mse_rep)
                rep_log = np.log10(rep_interp)
                
                corr = np.corrcoef(emp_log, rep_log)[0, 1]
                qualitative_match = corr > 0.9  # High correlation
                results["qualitative_match"][c_str] = {
                    "pass": bool(qualitative_match),
                    "correlation": float(corr) if np.isfinite(corr) else None
                }
    
    return results


def plot_step0_overlay(
    empirical_dfs: dict,
    replica_curves: dict,
    output_dir: str,
    rho: float,
    acceptance_results: dict,
):
    """Create overlay plot for Step 0 validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {"0.001": ("blue", "red"), "0.5": ("green", "orange")}
    
    for idx, (c_str, df) in enumerate(sorted(empirical_dfs.items())):
        ax = axes[idx]
        c_val = float(c_str)
        emp_color, rep_color = colors.get(c_str, ("blue", "red"))
        
        df = df.sort_values("alpha")
        
        # Filter valid data
        valid_mask = df["param_mse_mean"].notna() & (df["param_mse_mean"] > 0)
        df_valid = df[valid_mask].copy()
        
        if len(df_valid) > 0:
            # Plot empirical mean
            ax.plot(
                df_valid["alpha"],
                to_db(df_valid["param_mse_mean"].values),
                "o-",
                label=f"Empirical (mean)",
                linewidth=2,
                markersize=5,
                color=emp_color,
            )
            
            # Plot empirical median
            ax.plot(
                df_valid["alpha"],
                to_db(df_valid["param_mse_median"].values),
                "s--",
                label=f"Empirical (median)",
                linewidth=1.5,
                markersize=4,
                color=emp_color,
                alpha=0.7,
            )
            
            # Fill IQR
            if "param_mse_q25" in df_valid.columns:
                ax.fill_between(
                    df_valid["alpha"],
                    to_db(df_valid["param_mse_q25"].values),
                    to_db(df_valid["param_mse_q75"].values),
                    alpha=0.2,
                    color=emp_color,
                    label="IQR",
                )
        
        # Plot replica curve
        if c_str in replica_curves:
            alpha_rep, mse_rep = replica_curves[c_str]
            ax.plot(
                alpha_rep,
                to_db(mse_rep),
                "-",
                label="Replica Theory",
                linewidth=2.5,
                color=rep_color,
            )
        
        # Build status string for title
        status_parts = []
        if c_str in acceptance_results.get("stable_across_seeds", {}):
            s = acceptance_results["stable_across_seeds"][c_str]
            status_parts.append(f"Stable: {'✓' if s['pass'] else '✗'}")
        if c_str in acceptance_results.get("qualitative_match", {}):
            q = acceptance_results["qualitative_match"][c_str]
            status_parts.append(f"Match: {'✓' if q['pass'] else '✗'} (r={q['correlation']:.3f})")
        if c_str in acceptance_results.get("no_undertraining", {}):
            u = acceptance_results["no_undertraining"][c_str]
            status_parts.append(f"Converged: {'✓' if u['pass'] else '✗'}")
        
        status_str = ", ".join(status_parts) if status_parts else ""
        
        ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=12)
        ax.set_ylabel("Parameter MSE (dB)", fontsize=12)
        ax.set_title(
            f"Step 0 Validation: c = {c_val}\n"
            f"$\\rho$ = {rho}, {status_str}",
            fontsize=11
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    
    fig.tight_layout()
    
    # Save
    png_path = os.path.join(output_dir, "step0_validation_overlay.png")
    pdf_path = os.path.join(output_dir, "step0_validation_overlay.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nOverlay plot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 0 Validation: Golden Baseline")
    parser.add_argument("--rho", type=float, default=0.04, help="Sparsity parameter")
    parser.add_argument("--c_values", type=float, nargs="+", default=[0.001, 0.5],
                        help="C values to validate")
    parser.add_argument("--ft_regulariser_scale", type=float, default=1e-6,
                        help="FT regularization strength")
    parser.add_argument("--alpha_min", type=float, default=0.008,
                        help="Minimum alpha for replica")
    parser.add_argument("--alpha_max", type=float, default=1.0,
                        help="Maximum alpha for replica")
    parser.add_argument("--alpha_points", type=int, default=100,
                        help="Number of alpha points for replica")
    parser.add_argument("--mc_samples", type=int, default=50000,
                        help="Monte Carlo samples for replica")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--empirical_dir", type=str, 
                        default="figures/diagonal/bg_generalization",
                        help="Directory with empirical results")
    parser.add_argument("--output_dir", type=str, 
                        default="figures/validation_step0",
                        help="Output directory")
    parser.add_argument("--skip_replica", action="store_true",
                        help="Skip replica computation, use cached")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "replica_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("STEP 0 VALIDATION: GOLDEN BASELINE")
    print("="*80)
    print(f"rho = {args.rho}")
    print(f"c values = {args.c_values}")
    print(f"ft_regulariser_scale = {args.ft_regulariser_scale:.6e}")
    print(f"alpha range: [{args.alpha_min}, {args.alpha_max}] ({args.alpha_points} points)")
    print(f"MC samples = {args.mc_samples}")
    print("="*80)
    
    # Build replica config
    sigma0_2 = 0.0  # Noiseless
    beta_min = 1.0 / args.alpha_max
    beta_max = 1.0 / args.alpha_min
    
    cfg = build_config(
        rho=args.rho,
        sigma0_2=sigma0_2,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_points=args.alpha_points,
    )
    
    # Generate MC samples for replica
    print("\nGenerating Monte Carlo samples...")
    rng = np.random.default_rng(args.seed)
    x_mc = sample_bg(args.mc_samples, rng, args.rho, cfg.var_nonzero, return_mask=False)
    v_mc = rng.normal(size=args.mc_samples)
    print(f"Generated {args.mc_samples} MC samples")
    
    # Alpha range
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    
    # Compute replica curves
    print("\nComputing replica theory curves...")
    replica_curves = {}
    for c in args.c_values:
        c_str = str(c)
        cache_file = os.path.join(
            cache_dir,
            f"replica_step0_rho={args.rho:.6f}_c={c:.6f}_ft_reg={args.ft_regulariser_scale:.0e}.csv"
        )
        
        if args.skip_replica and os.path.exists(cache_file):
            print(f"\nLoading cached replica for c = {c:.6f}...")
            df_cache = pd.read_csv(cache_file)
            replica_curves[c_str] = (df_cache["alpha"].values, df_cache["mse"].values)
        else:
            print(f"\nComputing replica for c = {c:.6f}...")
            alpha_vals, mse_vals = compute_replica_curve(
                c, args.ft_regulariser_scale, alpha_range, cfg, x_mc, v_mc
            )
            replica_curves[c_str] = (alpha_vals, mse_vals)
            
            # Save cache
            df_cache = pd.DataFrame({"alpha": alpha_vals, "mse": mse_vals})
            df_cache.to_csv(cache_file, index=False)
            print(f"  Cached to {cache_file}")
        
        alpha_vals, mse_vals = replica_curves[c_str]
        print(f"  MSE range: [{mse_vals.min():.6e}, {mse_vals.max():.6e}]")
    
    # Load empirical results
    print("\nLoading empirical results...")
    empirical_dfs = {}
    for c in args.c_values:
        c_str = str(c)
        # Try ALL.csv first (includes all runs, not just successful ones)
        csv_path = os.path.join(
            args.empirical_dir,
            f"aggregated_results_rho={args.rho:.6f}--c={c:.6f}--ALL.csv"
        )
        if not os.path.exists(csv_path):
            csv_path = os.path.join(
                args.empirical_dir,
                f"aggregated_results_rho={args.rho:.6f}--c={c:.6f}.csv"
            )
        
        if os.path.exists(csv_path):
            df = load_empirical_results(csv_path)
            empirical_dfs[c_str] = df
            print(f"  Loaded c={c:.6f} from {csv_path}")
            print(f"    Alpha range: [{df['alpha'].min():.4f}, {df['alpha'].max():.4f}]")
            print(f"    Seeds per alpha: {df['total_count'].iloc[0] if 'total_count' in df.columns else 'unknown'}")
        else:
            print(f"  WARNING: No empirical data found for c={c:.6f}")
    
    if not empirical_dfs:
        print("\nERROR: No empirical data found. Cannot proceed with validation.")
        return 1
    
    # Check acceptance criteria
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    
    acceptance_results = check_acceptance_criteria(empirical_dfs, replica_curves, args.rho)
    
    all_pass = True
    
    print("\n1. Stability across seeds:")
    for c_str, result in acceptance_results.get("stable_across_seeds", {}).items():
        status = "PASS" if result["pass"] else "FAIL"
        cv = result.get("cv_proxy", "N/A")
        print(f"   c={c_str}: {status} (CV proxy = {cv:.3f})" if cv != "N/A" else f"   c={c_str}: {status}")
        if not result["pass"]:
            all_pass = False
    
    print("\n2. No undertraining:")
    for c_str, result in acceptance_results.get("no_undertraining", {}).items():
        status = "PASS" if result["pass"] else "FAIL"
        ratio = result.get("max_train_test_ratio", "N/A")
        print(f"   c={c_str}: {status} (max train/test ratio = {ratio:.2e})" if ratio != "N/A" else f"   c={c_str}: {status}")
        if not result["pass"]:
            all_pass = False
    
    print("\n3. Qualitative match with replica:")
    for c_str, result in acceptance_results.get("qualitative_match", {}).items():
        status = "PASS" if result["pass"] else "FAIL"
        corr = result.get("correlation", "N/A")
        print(f"   c={c_str}: {status} (correlation = {corr:.4f})" if corr != "N/A" else f"   c={c_str}: {status}")
        if not result["pass"]:
            all_pass = False
    
    print("\n" + "="*80)
    if all_pass:
        print("STEP 0 VALIDATION: ALL CRITERIA PASS ✓")
    else:
        print("STEP 0 VALIDATION: SOME CRITERIA FAIL ✗")
        print("Review the plots and diagnose issues before proceeding to Step 1.")
    print("="*80)
    
    # Create overlay plot
    print("\nCreating overlay plot...")
    plot_step0_overlay(empirical_dfs, replica_curves, args.output_dir, args.rho, acceptance_results)
    
    # Save acceptance results - convert numpy bools to Python bools for JSON serialization
    import json
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(x) for x in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj
    
    results_path = os.path.join(args.output_dir, "acceptance_results.json")
    with open(results_path, "w") as f:
        json.dump(convert_to_serializable(acceptance_results), f, indent=2)
    print(f"\nAcceptance results saved to: {results_path}")
    
    print("\nStep 0 validation complete!")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

