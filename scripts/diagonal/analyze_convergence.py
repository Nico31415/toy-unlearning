#!/usr/bin/env python3
"""Analyze convergence behavior of training experiments."""

import pandas as pd
import numpy as np

# Check convergence rate across all experiments
print('Analyzing convergence across all experiments:')
print('='*80)

results = []
for n_train in [200, 300, 400, 600, 800]:
    alpha = n_train / 1000.0
    for seed in range(10):
        df_path = f'results/diagonal/bg_experiments/alpha={alpha:.6f}--n_train={n_train}--seed={seed}--rho=0.040000--c=0.001000/df.feather'
        try:
            df = pd.read_feather(df_path)
            train_df = df[df['split'] == 'train'].sort_values('epoch')
            
            if len(train_df) < 2:
                continue
                
            epochs = train_df['epoch'].values
            train_mse = train_df['pred_mse'].values
            
            # Check final value
            final_mse = train_mse[-1]
            final_epoch = epochs[-1]
            
            # Check if still decreasing (last 10% of epochs)
            if len(train_mse) > 10:
                n_recent = max(10, len(train_mse) // 10)
                recent_mse = train_mse[-n_recent:]
                recent_epochs = epochs[-n_recent:]
                
                # Linear fit in log space
                log_mse = np.log10(recent_mse + 1e-20)
                if len(recent_epochs) > 1:
                    slope = np.polyfit(recent_epochs, log_mse, 1)[0]
                else:
                    slope = 0
                
                # Check improvement in last 100 epochs
                if len(train_mse) > 100:
                    improvement = (train_mse[-100] - train_mse[-1]) / train_mse[-100]
                else:
                    improvement = 0
                
                # Estimate epochs needed to reach 1e-12
                if slope < 0 and final_mse > 1e-12:
                    epochs_to_target = (np.log10(final_mse) - np.log10(1e-12)) / (-slope)
                else:
                    epochs_to_target = np.inf
                
                results.append({
                    'alpha': alpha,
                    'seed': seed,
                    'final_mse': final_mse,
                    'final_epoch': final_epoch,
                    'slope': slope,
                    'improvement': improvement,
                    'epochs_to_1e12': epochs_to_target,
                })
        except Exception as e:
            pass

if results:
    df_results = pd.DataFrame(results)
    
    print('\nFinal MSE by alpha:')
    for alpha in sorted(df_results['alpha'].unique()):
        subset = df_results[df_results['alpha'] == alpha]
        print(f'  Alpha={alpha:.3f}: Mean={subset["final_mse"].mean():.6e}, Median={subset["final_mse"].median():.6e}')
    
    print('\nConvergence rate (slope in log10 space, per epoch):')
    for alpha in sorted(df_results['alpha'].unique()):
        subset = df_results[df_results['alpha'] == alpha]
        mean_slope = subset['slope'].mean()
        print(f'  Alpha={alpha:.3f}: Mean slope={mean_slope:.6e} (negative = still decreasing)')
        if mean_slope > -1e-7:
            print(f'    WARNING: Loss appears to have plateaued!')
        else:
            # Estimate time to converge
            mean_epochs_needed = subset[subset['epochs_to_1e12'] < np.inf]['epochs_to_1e12'].mean()
            if not np.isnan(mean_epochs_needed):
                print(f'    Estimated epochs to reach 1e-12: {mean_epochs_needed:.0f}')
    
    print('\nImprovement in last 100 epochs:')
    for alpha in sorted(df_results['alpha'].unique()):
        subset = df_results[df_results['alpha'] == alpha]
        mean_improvement = subset['improvement'].mean()
        print(f'  Alpha={alpha:.3f}: Mean improvement={mean_improvement*100:.2f}%')
    
    print('\nCONCLUSION:')
    print('='*80)
    mean_slope_all = df_results['slope'].mean()
    if mean_slope_all < -1e-6:
        print('✓ Loss is STILL DECREASING (not stuck)')
        print(f'  Average convergence rate: {mean_slope_all:.6e} per epoch (log10)')
        print('  This means loss decreases by ~10^({:.2e}) per epoch'.format(mean_slope_all))
        print('\n  The learning rate (lr=0.05) might be too LOW for fine convergence.')
        print('  With very small gradients near the minimum, steps are tiny.')
        print('  Options:')
        print('    1. Increase learning rate (e.g., lr=0.1 or 0.2)')
        print('    2. Use learning rate schedule (start high, decay later)')
        print('    3. Increase epochs significantly (500k-1M)')
    else:
        print('✗ Loss appears to have PLATEAUED')
        print('  This suggests the learning rate is too high or optimization is stuck')







