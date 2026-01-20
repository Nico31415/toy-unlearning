#!/usr/bin/env python3
"""
Plot showing support-conditioned k effects.

Explains how different regularization strategies perform when you know 
which coordinates are signal vs noise in the teacher.

Configurations:
1. Homogeneous: All coordinates use k=4e-6 (baseline, no knowledge of support)
2. Collapse test: k_signal=k_noise=4e-6 (should match homogeneous - sanity check)
3. Correct strategy: k_signal=1.0 (weak reg), k_noise=4e-6 (strong reg)
4. Pathological: k_signal=4e-6 (strong reg), k_noise=1.0 (weak reg) - backwards!

Remember: small k → L1-like → strong reg; large k → L2-like → weak reg
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RHO = 0.04
LAMBDA_SMALL = 1e-6
C = 0.001
K_SIGNAL_WEAK = 1.0      # Large k = weak regularization (L2-like)
K_SIGNAL_STRONG = 4e-6   # Small k = strong regularization (L1-like)
K_NOISE_WEAK = 1.0       # Large k = weak regularization
K_NOISE_STRONG = 4e-6    # Small k = strong regularization

DATA_DIR = Path("figures/diagonal/bg_generalization")
OUTPUT_DIR = DATA_DIR

def to_db(x):
    """Convert MSE to dB."""
    return 10.0 * np.log10(np.maximum(x, 1e-15))

def load_npz(filename):
    """Load replica curve from .npz file."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    data = np.load(path)
    return data['alpha_vals'], data['mse_vals']

# Load data
print("Loading replica curve data...")

# Homogeneous baseline: all coordinates use same k (no knowledge of teacher support)
alpha_homo, mse_homo = load_npz(f"replica_curve_lambda={LAMBDA_SMALL:.0e}--c={C:.6f}.npz")

# Collapse test: k_signal = k_noise (should match homogeneous - sanity check)
alpha_collapse, mse_collapse = load_npz(
    f"replica_curve_lambda={LAMBDA_SMALL:.0e}--c={C:.6f}--"
    f"k_mode=support--k_nz={K_SIGNAL_STRONG:.6e}--k_z={K_NOISE_STRONG:.6e}.npz"
)

# CORRECT strategy: weak reg on signal, strong reg on noise
alpha_correct, mse_correct = load_npz(
    f"replica_curve_lambda={LAMBDA_SMALL:.0e}--c={C:.6f}--"
    f"k_mode=support--k_nz={K_SIGNAL_WEAK:.6e}--k_z={K_NOISE_STRONG:.6e}.npz"
)

# PATHOLOGICAL strategy: strong reg on signal, weak reg on noise (backwards!)
alpha_patho, mse_patho = load_npz(
    f"replica_curve_lambda={LAMBDA_SMALL:.0e}--c={C:.6f}--"
    f"k_mode=support--k_nz={K_SIGNAL_STRONG:.6e}--k_z={K_NOISE_WEAK:.6e}.npz"
)

print("Data loaded successfully.")

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors
colors = {
    'homo': '#7f7f7f',       # gray (baseline)
    'collapse': '#bcbd22',   # yellow-green (should match homo)
    'correct': '#2ca02c',    # green (good configuration)
    'patho': '#d62728',      # red (bad configuration)
}

# Plot curves
ax.plot(alpha_homo, to_db(mse_homo), 
        linewidth=3.0, color=colors['homo'], linestyle='-',
        label=r'Homogeneous: all coordinates use $k=4 \times 10^{-6}$ (baseline, no support knowledge)',
        zorder=4)

ax.plot(alpha_collapse, to_db(mse_collapse), 
        linewidth=2.0, color=colors['collapse'], linestyle='--',
        label=r'Collapse test: $k_{\mathrm{signal}} = k_{\mathrm{noise}} = 4 \times 10^{-6}$ (should match baseline)',
        zorder=3, alpha=0.8)

ax.plot(alpha_correct, to_db(mse_correct), 
        linewidth=3.0, color=colors['correct'], linestyle='-',
        label=r'Correct: $k_{\mathrm{signal}} = 1.0$ (weak reg), $k_{\mathrm{noise}} = 4 \times 10^{-6}$ (strong reg) ✓',
        zorder=5)

ax.plot(alpha_patho, to_db(mse_patho), 
        linewidth=2.5, color=colors['patho'], linestyle='-.',
        label=r'Pathological: $k_{\mathrm{signal}} = 4 \times 10^{-6}$ (strong reg), $k_{\mathrm{noise}} = 1.0$ (weak reg) ✗',
        zorder=2)

# Formatting
ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=16)
ax.set_ylabel('Parameter MSE (dB)', fontsize=16)
ax.set_title(
    f'Support-Conditioned Regularization Strategies\n'
    f'BG teacher: $\\rho = {RHO}$ ({RHO*100:.0f}% signal, {(1-RHO)*100:.0f}% noise), $\\lambda = {LAMBDA_SMALL:.0e}$',
    fontsize=14
)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10.5, loc='upper right', framealpha=0.95)

# Add annotation explaining the strategy
ax.text(0.02, 0.18, 
        r'Key insight: $q_k(x) = \sqrt{k + x^2}$ behavior:' + '\n' +
        r'  • Small $k$ (e.g., $4 \times 10^{-6}$) → L1-like → strong regularization' + '\n' +
        r'  • Large $k$ (e.g., $1.0$) → L2-like → weak regularization' + '\n\n' +
        r'Strategy: weak reg on signal (allow learning) + strong reg on noise (suppress)',
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

ax.set_xlim(alpha_homo.min(), alpha_homo.max())
ax.tick_params(labelsize=12)

fig.tight_layout()

# Save
output_png = OUTPUT_DIR / f"support_mode_comparison_rho={RHO:.2f}.png"
output_pdf = OUTPUT_DIR / f"support_mode_comparison_rho={RHO:.2f}.pdf"

fig.savefig(output_png, dpi=300, bbox_inches='tight')
fig.savefig(output_pdf, bbox_inches='tight')

print(f"\nPlot saved:")
print(f"  {output_png}")
print(f"  {output_pdf}")

plt.close(fig)
print("\nDone!")

