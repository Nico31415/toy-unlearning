"""
functions/bregman.py
====================
Bregman implicit-bias quantities for diagonal linear networks.

Reference: "Nonzero Initialization, Bregman Implicit Bias, and a Toy Theory
of Relearning-Resistant Unlearning"

All public functions are NumPy-vectorised and accept arbitrary broadcastable
array shapes.

Mathematical objects
--------------------
q_k(z)                  coordinate potential  (eq. 11)
q_k_prime(z)            = (1/2) arcsinh(2z/√k)               (eq. 12)
q_k_double_prime(z)     = 1 / √(k + 4z²)                     (eq. 12)
bregman_D(z, b, k)      Bregman divergence D_{q_k}(z, b)      (eq. 14)
C_rel(u, b, k)          local relearning cost = D_{q_k}(b+u, b)  (eq. 99)
ell_order(u, b, k)      update ℓ-order ∈ [1, 2]               (eq. 56-57)
PD_b(u, b, k)           centre elasticity d log D / d log|b|  (eq. 102)
canonical_PD(...)       full PD d log D / d log a             (eq. 101)
layer_rescaling_bk(...) b and k^FT from (a, c_PT, λ_PT, s, ρ)  (App. B)
compute_grid_quantities aggregate helper returning all four panel values
"""

from __future__ import annotations

import numpy as np

_EPS_K: float = 1e-30   # guard k = 0
_EPS_D: float = 1e-30   # guard D = 0 in log / ratio


# ---------------------------------------------------------------------------
# Potential and its derivatives
# ---------------------------------------------------------------------------

def q_k(z: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Coordinate potential q_k(z)  (eq. 11).

        q_k(z) = (√k / 4) * (1 - √(1 + 4z²/k) + (2z/√k) arcsinh(2z/√k))

    Limits:
        k → 0   :  q_k(z) → |z|              (L1 / rich regime)
        k → ∞   :  q_k(z) → z² / (2√k)      (quadratic / lazy regime)
    """
    k = np.maximum(k, _EPS_K)
    sqrt_k = np.sqrt(k)
    t = 2.0 * z / sqrt_k            # = 2z/√k
    return (sqrt_k / 4.0) * (1.0 - np.sqrt(1.0 + t * t) + t * np.arcsinh(t))


def q_k_prime(z: np.ndarray, k: np.ndarray) -> np.ndarray:
    """First derivative q_k'(z) = (1/2) arcsinh(2z/√k)  (eq. 12)."""
    k = np.maximum(k, _EPS_K)
    return 0.5 * np.arcsinh(2.0 * z / np.sqrt(k))


def q_k_double_prime(z: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Second derivative q_k''(z) = 1 / √(k + 4z²)  (eq. 12)."""
    k = np.maximum(k, _EPS_K)
    return 1.0 / np.sqrt(k + 4.0 * z * z)


# ---------------------------------------------------------------------------
# Bregman divergence and relearning cost
# ---------------------------------------------------------------------------

def bregman_D(z: np.ndarray, b: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Bregman divergence  D_{q_k}(z, b)  (eq. 14).

        D_{q_k}(z, b) = q_k(z) - q_k(b) - q_k'(b) · (z - b)

    Strictly non-negative for k > 0 (strict convexity of q_k).
    """
    return q_k(z, k) - q_k(b, k) - q_k_prime(b, k) * (z - b)


def C_rel(u: np.ndarray, b: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Local relearning cost  C_rel(u) = D_{q_k}(b + u, b)  (eq. 99).

    Cost of an update u starting from inherited centre b under geometry k.
    Smaller C_rel → cheaper relearning (dangerous for unlearning resistance).

    Dangerous centre-reuse regime (|b|² ≫ k, u small):
        C_rel ≈ u² / (4|b|)
    """
    return bregman_D(b + u, b, k)


# ---------------------------------------------------------------------------
# ℓ-order and pretraining dependence
# ---------------------------------------------------------------------------

def ell_order(u: np.ndarray, b: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Update ℓ-order  (eq. 56-57).

        ℓ = u · [q_k'(b+u) - q_k'(b)] / D_{q_k}(b+u, b)

    Measures the effective power-law of the update penalty:
        ℓ ≈ 1  :  sparse / rich-like  (cost ∝ |u|)
        ℓ ≈ 2  :  quadratic / lazy-like  (cost ∝ u²)

    Always ≥ 0; approaches 2 as u → 0  (eq. 59).
    """
    D = C_rel(u, b, k)
    numerator = u * (q_k_prime(b + u, k) - q_k_prime(b, k))
    return np.where(np.abs(D) > _EPS_D, numerator / D, 2.0)


def PD_b(u: np.ndarray, b: np.ndarray, k: np.ndarray,
         eps_fd: float = 1e-4) -> np.ndarray:
    """
    Centre pretraining dependence  PD_b  (eq. 102).

        PD_b = ∂ log D_{q_k}(b+u, b) / ∂ log|b|    [u, k fixed]

    Computed by multiplicative finite differences in |b|.

    Interpretation:
        PD_b ≈  0  :  pretraining-independent
        PD_b ≈ -1  :  strongly pretraining-dependent (large centre makes
                       update cheaper; dangerous centre-reuse regime)

    Dangerous limit (|b|² ≫ k, small u):
        PD_b → -4b² / (k + 4b²) → -1   (eq. 73, centre-channel version)
    """
    b_p  = b * (1.0 + eps_fd)
    D0   = C_rel(u, b,   k)
    D1   = C_rel(u, b_p, k)
    log0 = np.log(np.maximum(np.abs(D0), _EPS_D))
    log1 = np.log(np.maximum(np.abs(D1), _EPS_D))
    return (log1 - log0) / np.log(1.0 + eps_fd)


def canonical_PD(
    u: float,
    a: np.ndarray,
    c_PT: float,
    lambda_PT: float,
    s: np.ndarray,
    rho: np.ndarray,
    eps_fd: float = 1e-4,
) -> np.ndarray:
    """
    Canonical pretraining dependence  PD  (eq. 101) for layer-rescaling heatmaps.

        PD = ∂ log D_{q_{k(a)}}(b(a)+u, b(a)) / ∂ log a    [s, ρ, u fixed]

    Both b(a) = s·a and k^FT(a) change with a, so this captures both the
    centre channel and the geometry channel (eq. 62-63):

        PD = PD_centre + PD_geometry

    Computed by multiplicative finite differences in a.

    Args:
        u          : scalar update size
        a          : pretrained feature magnitude |β̂_PT|, array
        c_PT       : pretrained geometry scale (c > 0)
        lambda_PT  : pretrained layer asymmetry |λ| < 1
        s          : function scale grid (same shape as a)
        rho        : layer imbalance grid (same shape as a)

    Returns:
        PD values with the same shape as a / s / rho.
    """
    a1        = a * (1.0 + eps_fd)
    b0, k0    = layer_rescaling_bk(a,  c_PT, lambda_PT, s, rho)
    b1, k1    = layer_rescaling_bk(a1, c_PT, lambda_PT, s, rho)
    D0        = C_rel(u, b0, k0)
    D1        = C_rel(u, b1, k1)
    log0      = np.log(np.maximum(np.abs(D0), _EPS_D))
    log1      = np.log(np.maximum(np.abs(D1), _EPS_D))
    return (log1 - log0) / np.log(1.0 + eps_fd)


# ---------------------------------------------------------------------------
# Layer-rescaling formulas  (Appendix B, eqs. 94-98, 133)
# ---------------------------------------------------------------------------

def layer_rescaling_bk(
    a: np.ndarray,
    c_PT: float,
    lambda_PT: float,
    s: np.ndarray,
    rho: np.ndarray,
) -> tuple:
    """
    Inherited Bregman centre b and fine-tuning geometry k^FT from layer-
    rescaling parameters, under the balanced-pathway gauge  (Appendix B).

    Parameters
    ----------
    a          : pretrained feature magnitude |β̂_PT|  (≥ 0)
    c_PT       : pretrained geometry scale  c > 0
    lambda_PT  : pretrained layer asymmetry  |λ| < 1
    s          : function scale  s = α_in · α_out  (s > 0)
    rho        : layer imbalance  ρ = α_in / α_out  (ρ > 0)

    Returns
    -------
    b     : inherited centre  b = s · a              (eq. 95)
    k_FT  : fine-tuning geometry  k^FT  ≥ 0         (eq. 97 / 98)

    Formulas
    --------
    H       = √(c_PT² + a²)                                      (eq. 133)
    Σ_α     = s · (ρ + 1/ρ)                                      (eq. 94)
    Δ_α     = s · (ρ - 1/ρ)
    b       = s · a                                               (eq. 95)
    k^FT    = (Σ_α·c_PT + Δ_α·λ_PT·H)²
              + Δ_α²·a²·(1 - λ_PT²)                             (eq. 97)

    Special cases
    -------------
    ρ = 1  (equal layer scaling) :  Δ_α = 0,
        k^FT = (2·s·c_PT)²   — independent of a and λ_PT.
    a = 0  (inactive coordinate) :  b = 0,
        k^FT = (Σ_α·c_PT)²  — no centre-reuse basin.
    """
    H       = np.sqrt(c_PT**2 + np.asarray(a)**2)
    b       = np.asarray(s) * np.asarray(a)
    Sigma   = np.asarray(s) * (np.asarray(rho) + 1.0 / np.asarray(rho))
    Delta   = np.asarray(s) * (np.asarray(rho) - 1.0 / np.asarray(rho))
    k_FT    = (Sigma * c_PT + Delta * lambda_PT * H)**2 \
              + Delta**2 * np.asarray(a)**2 * (1.0 - lambda_PT**2)
    return b, np.maximum(k_FT, _EPS_K)


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def compute_grid_quantities(
    u: float,
    b: np.ndarray,
    k: np.ndarray,
    pd: np.ndarray,
) -> dict:
    """
    Compute the four standard regime-diagnostic quantities on a grid.

    The caller supplies ``pd`` (pretraining dependence) because its
    definition differs across heatmap types:
      - Reduced Bregman diagram : PD_b  (eq. 102)
      - Layer-rescaling diagram  : canonical_PD  (eq. 101)
      - Gauge diagram            : canonical_PD  (eq. 101)

    Parameters
    ----------
    u    : scalar update size
    b    : inherited centre grid
    k    : geometry grid
    pd   : pretraining dependence grid (same shape as b, k)

    Returns
    -------
    dict with keys:
        'ell'         : ℓ-order
        'pd'          : pretraining dependence (passed through)
        'ell_plus_pd' : ℓ + PD
        'log_C_rel'   : log C_rel  (log local relearning cost)
    """
    ell   = ell_order(u, b, k)
    C     = C_rel(u, b, k)
    log_C = np.log(np.maximum(C, _EPS_D))
    return {
        'ell':         ell,
        'pd':          pd,
        'ell_plus_pd': ell + pd,
        'log_C_rel':   log_C,
    }
