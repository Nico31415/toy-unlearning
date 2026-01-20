#!/usr/bin/env python3
"""Check convergence and compare empirical vs replica theory."""

import pandas as pd
import numpy as np

# Load preliminary results
df = pd.read_csv('figures/diagonal/bg_generalization/preliminary_results_c=0.001.csv')

# Load replica curve
df_replica = pd.read_csv('figures/diagonal/bg_generalization/replica_cache/replica_curve_rho=0.040000--c=0.001000--lambda=1.000000e-06--alpha_min=0.0080--alpha_max=1.0000--alpha_points=100--mc_samples=50000--seed=12345.csv')

# Aggregate empirical by alpha
agg = df.groupby('alpha')['param_mse'].agg(['mean', 'median', 'min', 'max']).reset_index()

print('Comparison: Empirical vs Replica Theory')
print('='*80)
print(f"{'Alpha':<10} {'Emp Mean (dB)':<15} {'Emp Min (dB)':<15} {'Replica (dB)':<15} {'Diff (dB)':<15}")
print('-'*80)

for _, row in agg.iterrows():
    alpha = row['alpha']
    emp_mean = 10 * np.log10(row['mean'] + 1e-15)
    emp_min = 10 * np.log10(row['min'] + 1e-15)
    
    # Find closest replica value
    idx = (df_replica['alpha'] - alpha).abs().idxmin()
    replica_mse = df_replica.loc[idx, 'mse']
    replica_db = 10 * np.log10(replica_mse + 1e-15)
    
    diff = emp_mean - replica_db
    
    print(f"{alpha:<10.3f} {emp_mean:<15.2f} {emp_min:<15.2f} {replica_db:<15.2f} {diff:<15.2f}")

print('\nNote: Negative diff means empirical is WORSE than theory')
print('='*80)

print('\nReplica theory predictions:')
print('='*60)
print(f"{'Alpha':<10} {'MSE':<20} {'MSE (dB)':<15}")
print('-'*60)

for alpha in [0.2, 0.3, 0.4, 0.6, 0.8]:
    idx = (df_replica['alpha'] - alpha).abs().idxmin()
    mse = df_replica.loc[idx, 'mse']
    mse_db = 10 * np.log10(mse + 1e-15)
    print(f"{alpha:<10.3f} {mse:<20.6e} {mse_db:<15.2f}")

print('\nReference:')
print('-40 dB = 1e-4 = 0.0001')
print('-50 dB = 1e-5 = 0.00001')
print('-60 dB = 1e-6 = 0.000001')

print('\nConvergence check:')
print('='*60)
print(f"All experiments hit epoch limit: {(df['epoch'] == 199999).all()}")
print(f"Train MSE < 1e-10: {(df['train_pred_mse'] < 1e-10).sum()} / {len(df)}")
print(f"Train MSE < 1e-8: {(df['train_pred_mse'] < 1e-8).sum()} / {len(df)}")
print(f"Train MSE median: {df['train_pred_mse'].median():.6e}")
print(f"Train MSE mean: {df['train_pred_mse'].mean():.6e}")







