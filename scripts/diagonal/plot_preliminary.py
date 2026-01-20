#!/usr/bin/env python3
"""Quick script to plot preliminary results from partial experiments."""

import pandas as pd
import numpy as np
import os
import sys
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import replica theory functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from ReplicaExperiments.fixed_lambda_all import (
    Config,
    gamma_ext_for_q_small,
    gamma_ext_for_q_big,
    solve_rspmap_qk_curve_best_of_forward_backward,
    sample_bg,
)

# Collect all current results
results = []
for n_train in [200, 300, 400, 600, 800, 1024, 1500]:
    alpha = n_train / 1000.0
    for seed in range(10):
        save_folder = f'results/diagonal/bg_experiments/alpha={alpha:.6f}--n_train={n_train}--seed={seed}--rho=0.040000--c=0.001000/'
        df_path = os.path.join(save_folder, 'df.feather')
        if os.path.exists(df_path):
            try:
                df = pd.read_feather(df_path)
                test_df = df[df['split'] == 'test']
                train_df = df[df['split'] == 'train']
                if not test_df.empty and not train_df.empty:
                    final_epoch = test_df['epoch'].max()
                    final_test = test_df[test_df['epoch'] == final_epoch].iloc[0]
                    train_at_final = train_df[train_df['epoch'] == final_epoch]
                    if train_at_final.empty:
                        train_at_final = train_df.loc[train_df['epoch'].idxmax()]
                    else:
                        train_at_final = train_at_final.iloc[0]
                    
                    results.append({
                        'alpha': alpha,
                        'n_train': n_train,
                        'seed': seed,
                        'test_pred_mse': final_test['pred_mse'],
                        'param_mse': final_test['param_mse'],
                        'train_pred_mse': train_at_final['pred_mse'],
                        'epoch': final_epoch,
                    })
            except Exception as e:
                print(f'Error reading {df_path}: {e}')

if results:
    df_results = pd.DataFrame(results)
    print(f'Collected {len(df_results)} results')
    print(f'Alpha values: {sorted(df_results["alpha"].unique())}')
    print(f'\nTrain pred MSE stats:')
    print(f'  Mean: {df_results["train_pred_mse"].mean():.6e}')
    print(f'  Median: {df_results["train_pred_mse"].median():.6e}')
    print(f'  Min: {df_results["train_pred_mse"].min():.6e}')
    print(f'  Max: {df_results["train_pred_mse"].max():.6e}')
    print(f'  < 1e-10: {(df_results["train_pred_mse"] < 1e-10).sum()} / {len(df_results)}')
    print(f'  < 1e-8: {(df_results["train_pred_mse"] < 1e-8).sum()} / {len(df_results)}')
    print(f'\nEpoch stats:')
    print(f'  Mean: {df_results["epoch"].mean():.0f}')
    print(f'  Median: {df_results["epoch"].median():.0f}')
    print(f'  Min: {df_results["epoch"].min()}')
    print(f'  Max: {df_results["epoch"].max()}')
    
    # Aggregate by alpha
    agg = df_results.groupby('alpha').agg({
        'param_mse': ['mean', 'median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)],
        'train_pred_mse': 'mean',
    }).reset_index()
    agg.columns = ['alpha', 'param_mse_mean', 'param_mse_median', 'param_mse_q25', 'param_mse_q75', 'train_pred_mse_mean']
    
    # Convert to dB
    agg['param_mse_mean_db'] = 10 * np.log10(agg['param_mse_mean'] + 1e-15)
    agg['param_mse_median_db'] = 10 * np.log10(agg['param_mse_median'] + 1e-15)
    agg['param_mse_q25_db'] = 10 * np.log10(agg['param_mse_q25'] + 1e-15)
    agg['param_mse_q75_db'] = 10 * np.log10(agg['param_mse_q75'] + 1e-15)
    
    # Load replica curve from cache
    print('\nLoading replica theory curve for c=0.001 from cache...')
    c = 0.001
    rho = 0.04
    ft_regulariser_scale = 1e-6
    alpha_min = 0.008
    alpha_max = 1.0
    alpha_points = 100
    mc_samples = 50000
    seed = 12345
    
    cache_dir = 'figures/diagonal/bg_generalization/replica_cache'
    cache_filename = (
        f"replica_curve_teacher=bg--rho={rho:.6f}--c={c:.6f}--"
        f"ft_reg={ft_regulariser_scale:.6e}--alpha_min={alpha_min:.4f}--"
        f"alpha_max={alpha_max:.4f}--alpha_points={alpha_points}--"
        f"mc_samples={mc_samples}--seed={seed}.csv"
    )
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if os.path.exists(cache_path):
        print(f'  Loading from cache: {cache_path}')
        df_cache = pd.read_csv(cache_path)
        alpha_range = df_cache['alpha'].values
        mse_alpha = df_cache['mse'].values
        mse_alpha_db = 10 * np.log10(mse_alpha + 1e-15)
        print(f'  Loaded. MSE range: [{mse_alpha.min():.6e}, {mse_alpha.max():.6e}]')
    else:
        print(f'  Cache not found at {cache_path}')
        print('  Computing replica curve (this will take a few minutes)...')
        
        # Build config
        sigma0_2 = 0.0
        beta_min = 1.0 / alpha_max
        beta_max = 1.0 / alpha_min
        var_nonzero = 1.0 / rho
        betas = np.linspace(beta_min, beta_max, alpha_points)
        
        cfg = Config(
            rho=rho,
            var_nonzero=var_nonzero,
            sigma0_2=sigma0_2,
            betas=betas,
            max_fp_iters=900,
            tol_fp=1e-10,
            damp=0.25,
        )
        
        # Generate MC samples
        rng = np.random.default_rng(seed)
        x_mc = sample_bg(mc_samples, rng, rho, var_nonzero)
        v_mc = rng.normal(size=mc_samples)
        
        # Compute k_q from c
        k_q = (2.0 * c) ** 2
        
        # Compute gamma_ext
        if k_q < 1.0:
            gamma_ext = gamma_ext_for_q_small(ft_regulariser_scale, k_q)
        else:
            gamma_ext = gamma_ext_for_q_big(ft_regulariser_scale, k_q)
        
        print(f'  c={c:.6f}, sqrt_k={math.sqrt(k_q):.6e}, k_q={k_q:.6e}, ft_reg={ft_regulariser_scale:.6e}, gamma_ext={gamma_ext:.6e}')
        
        # Create alpha range
        alpha_range = np.linspace(alpha_min, alpha_max, alpha_points)
        alpha_reversed = alpha_range[::-1]
        beta_range = 1.0 / alpha_reversed
        
        # Compute replica curve
        mse_beta = solve_rspmap_qk_curve_best_of_forward_backward(
            beta_range, gamma_ext, k_q, x_mc, v_mc, cfg
        )
        mse_alpha = mse_beta[::-1]
        
        # Convert to dB
        mse_alpha_db = 10 * np.log10(mse_alpha + 1e-15)
        
        print(f'  Replica MSE range: [{mse_alpha.min():.6e}, {mse_alpha.max():.6e}]')
        
        # Save to cache
        os.makedirs(cache_dir, exist_ok=True)
        df_cache = pd.DataFrame({'alpha': alpha_range, 'mse': mse_alpha})
        df_cache.to_csv(cache_path, index=False)
        print(f'  Saved to cache: {cache_path}')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    agg = agg.sort_values('alpha')
    
    # Plot empirical
    ax.plot(agg['alpha'], agg['param_mse_mean_db'], 'o-', label='Empirical mean (preliminary)', linewidth=2, markersize=6, color='blue')
    ax.plot(agg['alpha'], agg['param_mse_median_db'], 's--', label='Empirical median (preliminary)', linewidth=2, markersize=5, color='red', alpha=0.7)
    ax.fill_between(agg['alpha'], agg['param_mse_q25_db'], agg['param_mse_q75_db'], alpha=0.2, color='blue', label='IQR')
    
    # Plot replica theory
    ax.plot(alpha_range, mse_alpha_db, '-', label='Replica theory (c=0.001)', linewidth=2.5, color='orange', alpha=0.9)
    
    ax.set_xlabel(r'$\alpha = n_{\text{train}} / d$', fontsize=12)
    ax.set_ylabel('Parameter MSE (dB)', fontsize=12)
    n_success = (df_results['train_pred_mse'] < 1e-10).sum()
    ax.set_title(f'Preliminary Results: c=0.001 ({len(df_results)}/70 experiments)\nTrain MSE < 1e-10: {n_success}/{len(df_results)}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    output_dir = 'figures/diagonal/bg_generalization'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/preliminary_plot_c=0.001.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{output_dir}/preliminary_plot_c=0.001.pdf', bbox_inches='tight')
    print(f'\nPreliminary plot saved to: {output_dir}/preliminary_plot_c=0.001.png')
    
    # Also save CSV
    df_results.to_csv(f'{output_dir}/preliminary_results_c=0.001.csv', index=False)
    print(f'Results saved to: {output_dir}/preliminary_results_c=0.001.csv')
else:
    print('No results found')

