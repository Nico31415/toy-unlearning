#!/usr/bin/env python3
"""
Plot test MSE vs alpha for each optimizer/lr combo.

Usage:
    python experiments/diagonal/plot_optimizer_sweep.py \
        --csv /tmp/opt_sweep/optimizer_comparison.csv \
        --out /tmp/opt_sweep/optimizer_sweep.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

STYLE = {
    'full_batch':    dict(color='black',  linestyle='-',  marker='o', label='Full-batch GD'),
    'sgd':           dict(color='blue',   linestyle='-',  marker='s', label='SGD (mom=0.9)'),
    'adam_lr=0.01':  dict(color='red',    linestyle='--', marker='^', label='Adam lr=1e-2'),
    'adam_lr=0.001': dict(color='orange', linestyle='--', marker='v', label='Adam lr=1e-3'),
    'adam_lr=0.0001':dict(color='green',  linestyle='--', marker='D', label='Adam lr=1e-4'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    summary = df.groupby(['label', 'alpha'])['final_test_mse'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, grp in summary.groupby('label'):
        grp = grp.sort_values('alpha')
        style = STYLE.get(label, dict(marker='x'))
        ax.plot(grp['alpha'], grp['mean'], **style)
        ax.fill_between(
            grp['alpha'],
            grp['mean'] - grp['std'],
            grp['mean'] + grp['std'],
            alpha=0.15,
            color=style.get('color', 'grey'),
        )

    ax.set_xlabel(r'$\alpha = n_{\rm train} / d$', fontsize=13)
    ax.set_ylabel('Test MSE', fontsize=13)
    ax.set_title('Generalisation vs. $\\alpha$ — oracle FT (full overlap)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()
