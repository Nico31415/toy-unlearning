#!/usr/bin/env python3
"""
Plot showing how mixture mode interpolates as we vary pi_A.

Shows the smooth transition from all k_B (pi_A→0, strong regularization)
to all k_A (pi_A→1, weak regularization).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RHO = 0.04
LAMBDA_SMALL = 1e-6
C = 0.001
K_A = 4e-6
K_B = 1.0

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

# Homogeneous k=4e-6 (effectively pi_A=1, all k_A)
alpha_homo, mse_homo = load_npz(f"replica_curve_lambda={LAMBDA_SMALL:.0e}--c={C:.6f}.npz")

# Mixture mode: k_A=4e-6, k_B=1.0
alpha_09, mse_09 = load_npz(
    f"replica_curve_lambda={LAMBDA_SMALL:.0e}--c={C:.6f}--"
    f"k_mode=mixture--k_A={K_A:.6e}--k_B={K_B:.6e}--pi_A=9.000000e-01.npz"
)

alpha_05, mse_05 = load_npz(
    f"replica_curve_lambda={LAMBDA_SMALL:.0e}--c={C:.6f}--"
    f"k_mode=mixture--k_A={K_A:.6e}--k_B={K_B:.6e}--pi_A=5.000000e-01.npz"
)

alpha_01, mse_01 = load_npz(
    f"replica_curve_lambda={LAMBDA_SMALL:.0e}--c={C:.6f}--"
    f"k_mode=mixture--k_A={K_A:.6e}--k_B={K_B:.6e}--pi_A=1.000000e-01.npz"
)

print("Data loaded successfully.")

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors (gradient from blue to red)
colors = {
    'homo': '#d73027',      # red (all k_A, weak reg)
    '0.9': '#fc8d59',       # orange-red
    '0.5': '#fee090',       # yellow
    '0.1': '#91bfdb',       # light blue
    # '0.0': '#4575b4',     # blue (all k_B, strong reg) - if we had it
}

# Plot curves (in order from weak to strong regularization)
ax.plot(alpha_homo, to_db(mse_homo), 
        linewidth=3.0, color=colors['homo'], 
        label=r'Homogeneous $k=4 \times 10^{-6}$ (effectively $\pi_A \approx 1$)',
        zorder=5)

ax.plot(alpha_09, to_db(mse_09), 
        linewidth=2.5, color=colors['0.9'], 
        label=r'Mixture: $\pi_A = 0.9$ (mostly small $k$)',
        zorder=4)

ax.plot(alpha_05, to_db(mse_05), 
        linewidth=2.5, color=colors['0.5'], 
        label=r'Mixture: $\pi_A = 0.5$ (balanced)',
        zorder=3)

ax.plot(alpha_01, to_db(mse_01), 
        linewidth=2.5, color=colors['0.1'], 
        label=r'Mixture: $\pi_A = 0.1$ (mostly large $k$)',
        zorder=2)

# Formatting
ax.set_xlabel(r'$\alpha = n_{\mathrm{train}} / d$', fontsize=16)
ax.set_ylabel('Parameter MSE (dB)', fontsize=16)
ax.set_title(
    f'Mixture Mode Interpolation: Varying $\\pi_A$\n'
    f'$k_A = {K_A:.0e}$ (weak reg) vs $k_B = {K_B:.1f}$ (strong reg), '
    f'$\\rho = {RHO}$, $\\lambda = {LAMBDA_SMALL:.0e}$',
    fontsize=14
)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)

# Add annotation explaining the interpolation
ax.text(0.02, 0.05, 
        r'As $\pi_A$ increases: more coordinates have small $k_A$ → earlier learning',
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

ax.set_xlim(alpha_homo.min(), alpha_homo.max())
ax.tick_params(labelsize=12)

fig.tight_layout()

# Save
output_png = OUTPUT_DIR / f"mixture_interpolation_pi_A_sweep_rho={RHO:.2f}.png"
output_pdf = OUTPUT_DIR / f"mixture_interpolation_pi_A_sweep_rho={RHO:.2f}.pdf"

fig.savefig(output_png, dpi=300, bbox_inches='tight')
fig.savefig(output_pdf, bbox_inches='tight')

print(f"\nPlot saved:")
print(f"  {output_png}")
print(f"  {output_pdf}")

plt.close(fig)
print("\nDone!")




