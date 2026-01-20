#!/usr/bin/env python3
"""
Step 2 Validation: Overlay empirical support-conditioned k vs replica curves.

This script:
1. Aggregates empirical results from step2_support experiments
2. Generates replica curves with matching support-k parameters
3. Creates overlay plots for validation
4. Checks acceptance criteria

Usage:
    python scripts/diagonal/plot_step2_validation.py --output_dir figures/validation_step2
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
    KModeConfig,
    solve_rspmap_qk_curve_best_of_forward_backward,
    sample_bg,
    sample_k_mc,
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


def gamma_ext_for_hetero_k(ft_regulariser_scale: float) -> float:
    """For heterogeneous-k, gamma_ext = ft_regulariser_scale directly."""
    return float(ft_regulariser_scale)


def compute_replica_curve_support(
    k_nz: float,
    k_z: float,
    ft_regulariser_scale: float,
    alpha_range: np.ndarray,
    cfg: Config,
    x_mc: np.ndarray,
    v_mc: np.ndarray,
    mask_bg: np.ndarray,
    rng: np.random.Generator,
) -> tuple:
    """Compute replica theory curve for support-k mode."""
    # Sample k_mc using support mode
    k_cfg = KModeConfig(
        mode="support",
        k_nz=k_nz,
        k_z=k_z,
    )
    k_mc, g_mc = sample_k_mc(k_cfg, x_mc, rng, mask_bg=mask_bg)
    
    # Gamma_ext for hetero-k
    gamma_ext = gamma_ext_for_hetero_k(ft_regulariser_scale)
    
    print(f"  [support] k_nz={k_nz:.6e}, k_z={k_z:.6e}")
    print(f"  gamma_ext={gamma_ext:.6e}")
    
    # Convert alpha to beta
    alpha_reversed = alpha_range[::-1]
    beta_range = 1.0 / alpha_reversed
    
    # Use hetero-k path
    k_q = 1.0  # Placeholder
    mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
        beta_range, gamma_ext, k_q, x_mc, v_mc, cfg, k_mc=k_mc, g_mc=g_mc
    )
    
    mse_alpha = mse_beta[::-1]
    return alpha_range, mse_alpha


def aggregate_empirical_results(csv_path: str) -> pd.DataFrame:
    """Load and aggregate empirical results by (case_name, alpha)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Aggregate by (case_name, alpha)
    agg = df.groupby(['case_name', 'alpha']).agg({
        'param_mse': ['mean', 'median', 'std', 'count',
                      lambda x: np.percentile(x, 25),
                      lambda x: np.percentile(x, 75)],
        'test_pred_mse': ['mean', 'median'],
        'train_pred_mse': ['mean', 'median'],
    }).reset_index()
    
    # Flatten columns
    agg.columns = [
        'case_name', 'alpha',
        'param_mse_mean', 'param_mse_median', 'param_mse_std', 'count',
        'param_mse_q25', 'param_mse_q75',
        'test_pred_mse_mean', 'test_pred_mse_median',
        'train_pred_mse_mean', 'train_pred_mse_median',
    ]
    
    return agg


def check_acceptance_criteria(agg_df: pd.DataFrame, replica_curves: dict) -> dict:
    """Check Step 2 acceptance criteria."""
    results = {
        "directionality_correct": False,
        "collapse_sanity": None,  # Would need c_nz == c_z run
    }
    
    cases = agg_df['case_name'].unique()
    
    if 'good' not in cases or 'bad' not in cases:
        print("WARNING: Need both 'good' and 'bad' cases to check directionality")
        return results
    
    # Check directionality: 'good' case should have lower MSE at low alpha
    # (where support-conditioned init matters most)
    directionality_checks = []
    for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
        good_mask = (agg_df['case_name'] == 'good') & (np.abs(agg_df['alpha'] - alpha) < 0.01)
        bad_mask = (agg_df['case_name'] == 'bad') & (np.abs(agg_df['alpha'] - alpha) < 0.01)
        
        if good_mask.sum() > 0 and bad_mask.sum() > 0:
            good_mse = agg_df.loc[good_mask, 'param_mse_mean'].values[0]
            bad_mse = agg_df.loc[bad_mask, 'param_mse_mean'].values[0]
            # Good case should have lower MSE
            directionality_checks.append(good_mse < bad_mse)
    
    results["directionality_correct"] = all(directionality_checks) if directionality_checks else False
    
    return results


def plot_step2_overlay(
    agg_df: pd.DataFrame,
    replica_curves: dict,
    output_dir: str,
    acceptance_results: dict,
):
    """Create overlay plot for Step 2 validation."""
    cases = sorted(agg_df['case_name'].unique())
    
    colors = {'good': ('green', 'darkgreen'), 'bad': ('red', 'darkred')}
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for case_name in cases:
        emp_color, rep_color = colors.get(case_name, ('gray', 'black'))
        
        # Empirical data
        mask = agg_df['case_name'] == case_name
        df_case = agg_df[mask].sort_values('alpha')
        
        if len(df_case) > 0:
            label_suffix = "(c_nz<c_z)" if case_name == 'good' else "(c_nz>c_z)"
            ax.plot(
                df_case['alpha'],
                to_db(df_case['param_mse_mean'].values),
                'o-',
                label=f'Empirical {case_name} {label_suffix}',
                linewidth=2,
                markersize=5,
                color=emp_color,
            )
            
            # IQR fill
            if 'param_mse_q25' in df_case.columns:
                ax.fill_between(
                    df_case['alpha'],
                    to_db(df_case['param_mse_q25'].values),
                    to_db(df_case['param_mse_q75'].values),
                    alpha=0.15,
                    color=emp_color,
                )
        
        # Replica curve
        if case_name in replica_curves:
            alpha_rep, mse_rep = replica_curves[case_name]
            ax.plot(
                alpha_rep,
                to_db(mse_rep),
                '--',
                label=f'Replica {case_name}',
                linewidth=2.5,
                color=rep_color,
            )
    
    # Status string
    status = "Directionality: ✓" if acceptance_results.get("directionality_correct") else "Directionality: ✗"
    
    ax.set_xlabel(r"$\alpha = n_{train} / d$", fontsize=12)
    ax.set_ylabel("Parameter MSE (dB)", fontsize=12)
    ax.set_title(
        f"Step 2 Validation: Support-Conditioned k Init\n"
        f"ρ=0.04 | {status}",
        fontsize=11
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    
    fig.tight_layout()
    
    png_path = os.path.join(output_dir, "step2_validation_overlay.png")
    pdf_path = os.path.join(output_dir, "step2_validation_overlay.pdf")
    
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nOverlay plot saved to:")
    print(f"  {png_path}")
    print(f"  {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 2 Validation: Support-k")
    parser.add_argument("--rho", type=float, default=0.04)
    parser.add_argument("--ft_regulariser_scale", type=float, default=1e-6)
    parser.add_argument("--alpha_min", type=float, default=0.008)
    parser.add_argument("--alpha_max", type=float, default=1.0)
    parser.add_argument("--alpha_points", type=int, default=100)
    parser.add_argument("--mc_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--empirical_csv", type=str, 
                        default="experiment_results_step2_support.csv")
    parser.add_argument("--output_dir", type=str, 
                        default="figures/validation_step2")
    parser.add_argument("--skip_replica", action="store_true")
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "replica_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Cases: good = c_nz < c_z, bad = c_nz > c_z
    cases = {
        'good': {'c_nz': 0.001, 'c_z': 0.5},
        'bad': {'c_nz': 0.5, 'c_z': 0.001},
    }
    
    print("="*80)
    print("STEP 2 VALIDATION: SUPPORT-CONDITIONED k INIT")
    print("="*80)
    print(f"rho = {args.rho}")
    for name, params in cases.items():
        k_nz = (2.0 * params['c_nz']) ** 2
        k_z = (2.0 * params['c_z']) ** 2
        print(f"Case {name}: c_nz={params['c_nz']}, c_z={params['c_z']} -> k_nz={k_nz:.6e}, k_z={k_z:.6e}")
    print("="*80)
    
    # Build config
    sigma0_2 = 0.0
    beta_min = 1.0 / args.alpha_max
    beta_max = 1.0 / args.alpha_min
    
    cfg = build_config(
        rho=args.rho,
        sigma0_2=sigma0_2,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_points=args.alpha_points,
    )
    
    # Generate MC samples with mask for support mode
    print("\nGenerating Monte Carlo samples...")
    rng = np.random.default_rng(args.seed)
    x_mc, mask_bg = sample_bg(args.mc_samples, rng, args.rho, cfg.var_nonzero, return_mask=True)
    v_mc = rng.normal(size=args.mc_samples)
    
    alpha_range = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
    
    # Compute replica curves for each case
    print("\nComputing replica curves...")
    replica_curves = {}
    for case_name, params in cases.items():
        k_nz = (2.0 * params['c_nz']) ** 2
        k_z = (2.0 * params['c_z']) ** 2
        
        cache_file = os.path.join(
            cache_dir,
            f"replica_support_{case_name}_k_nz={k_nz:.6e}_k_z={k_z:.6e}.csv"
        )
        
        if args.skip_replica and os.path.exists(cache_file):
            print(f"\nLoading cached replica for case {case_name}...")
            df_cache = pd.read_csv(cache_file)
            replica_curves[case_name] = (df_cache["alpha"].values, df_cache["mse"].values)
        else:
            print(f"\nComputing replica for case {case_name}...")
            rng_case = np.random.default_rng(args.seed + hash(case_name) % 10000)
            alpha_vals, mse_vals = compute_replica_curve_support(
                k_nz, k_z,
                args.ft_regulariser_scale,
                alpha_range, cfg, x_mc, v_mc, mask_bg, rng_case
            )
            replica_curves[case_name] = (alpha_vals, mse_vals)
            
            df_cache = pd.DataFrame({"alpha": alpha_vals, "mse": mse_vals})
            df_cache.to_csv(cache_file, index=False)
        
        alpha_vals, mse_vals = replica_curves[case_name]
        print(f"  MSE range: [{mse_vals.min():.6e}, {mse_vals.max():.6e}]")
    
    # Load empirical results
    print("\nLoading empirical results...")
    if os.path.exists(args.empirical_csv):
        agg_df = aggregate_empirical_results(args.empirical_csv)
        print(f"  Loaded {len(agg_df)} aggregated rows")
        print(f"  Cases: {sorted(agg_df['case_name'].unique())}")
    else:
        print(f"  WARNING: Empirical results not found at {args.empirical_csv}")
        print("  Run step2_support_sweep.sh first to generate empirical data")
        agg_df = pd.DataFrame()
    
    # Check acceptance criteria
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    
    if len(agg_df) > 0:
        acceptance_results = check_acceptance_criteria(agg_df, replica_curves)
        
        print(f"\n1. Directionality correct ('good' < 'bad' at low alpha): "
              f"{'PASS' if acceptance_results['directionality_correct'] else 'FAIL'}")
        
        all_pass = acceptance_results['directionality_correct']
        
        print("\n" + "="*80)
        if all_pass:
            print("STEP 2 VALIDATION: ALL CRITERIA PASS ✓")
        else:
            print("STEP 2 VALIDATION: SOME CRITERIA FAIL ✗")
        print("="*80)
        
        # Create overlay plot
        print("\nCreating overlay plot...")
        plot_step2_overlay(agg_df, replica_curves, args.output_dir, acceptance_results)
        
        # Save results - convert numpy bools to Python bools for JSON serialization
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
        
        return 0 if all_pass else 1
    else:
        print("\nNo empirical data available. Run experiments first.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


