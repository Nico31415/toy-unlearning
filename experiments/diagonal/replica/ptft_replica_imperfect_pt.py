# ================================
# File: ptft_replica_imperfect_pt.py
# ================================
# Extension of the PT+FT replica curve to imperfect pretraining (alpha_pt < 1).
#
# Theory (replica_derivation.pdf):
#   When pretraining is imperfect (alpha_pt != 1), the pretrained predictor
#   beta_hat_PT is random, following a scalar pretraining channel:
#
#     B_hat_PT = eta_est_PT(B*_PT + sqrt(tau_PT) * Z),  Z ~ N(0,1)
#     K = Psi(B_hat_PT)
#
#   The FT replica equations are unchanged in form; only the law of K changes.
#   alpha_pt = 1 is the oracle limit (square system, unique solution = beta*_PT).
#
# Public API:
#   - ptft_qk_curve_imperfect_pt(...)
#
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ptft_replica_qk import (
    PTFTOracleParams,
    _batch_means_se,
    _default_gamma_schedule,
    compute_c_ft_from_pt,
    reliability_score,
    sample_ptft_oracle_mc,
    to_db,
)


# -------------------------
# Vectorized prox + local variance (k may be scalar or array)
# -------------------------

def _prox_qk_vec(
    z: np.ndarray,
    gp: float,
    k,
    iters: int = 20,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Solve x - z + 0.5*gp*asinh(2x/sqrt(k)) = 0 by safeguarded Newton/bisection.
    k may be a scalar or an array broadcastable with z.
    """
    z = np.asarray(z, float)
    gp = float(gp)
    k = np.asarray(k, dtype=float)
    if gp <= 0.0:
        return z.copy()

    sk = np.sqrt(k)
    lo = np.minimum(z, 0.0)
    hi = np.maximum(z, 0.0)
    x = 0.5 * (lo + hi)

    for _ in range(iters):
        Fx = x - z + 0.5 * gp * np.arcsinh(2.0 * x / sk)
        if float(np.max(np.abs(Fx))) < tol:
            break
        Fpx = 1.0 + gp / np.sqrt(k + 4.0 * x * x)
        x_new = x - Fx / Fpx

        neg = Fx < 0
        lo = np.where(neg, x, lo)
        hi = np.where(~neg, x, hi)

        bad = (x_new < lo) | (x_new > hi) | (~np.isfinite(x_new))
        x = np.where(bad, 0.5 * (lo + hi), x_new)

    return x


def _sigma2_qk_vec(xhat: np.ndarray, gp: float, k) -> np.ndarray:
    gp = float(max(gp, 1e-14))
    k = np.asarray(k, dtype=float)
    qpp = 1.0 / np.sqrt(k + 4.0 * xhat * xhat)
    return 1.0 / (1.0 / gp + qpp)


# -------------------------
# FP solver using vectorized prox (heterogeneous k support)
# -------------------------

def _solve_fp_qk_vec(
    *,
    beta: float,
    x: np.ndarray,
    v: np.ndarray,
    k_mc: np.ndarray,
    g_mc: Optional[np.ndarray],
    sigma0_2: float,
    gamma_ext: float,
    init_state: Optional[Tuple[float, float]] = None,
    max_iters: int = 900,
    tol: float = 1e-10,
    damp: float = 0.25,
    eps_active: float = 1e-6,
    use_grouped_k: bool = True,
    n_batches_se: int = 50,
) -> Tuple[float, float, Tuple[float, float], float, float]:
    """
    Identical to solve_fp_qk_one from ptft_replica_qk, except the per-sample-k
    fallback path uses vectorized _prox_qk_vec / _sigma2_qk_vec instead of a
    Python loop over coordinates.
    """
    beta = float(beta)
    sigma0_2 = float(sigma0_2)
    gamma_ext = float(gamma_ext)

    s2 = float(max(sigma0_2, init_state[0])) if init_state else float(max(sigma0_2, sigma0_2))
    gp = float(max(gamma_ext, init_state[1], 1e-14)) if init_state else float(max(gamma_ext, 1e-14))

    x = np.asarray(x, float)
    v = np.asarray(v, float)
    k_mc = np.asarray(k_mc, float)
    if k_mc.shape != x.shape:
        raise ValueError("k_mc must match x shape")

    if g_mc is not None:
        g = np.asarray(g_mc)
        if g.shape != x.shape:
            raise ValueError("g_mc must match x shape")
        uniq_g = np.unique(g)
    else:
        g = None
        uniq_g = None

    last_res = float("inf")
    for _ in range(int(max_iters)):
        z = x + math.sqrt(max(s2, 1e-15)) * v

        if use_grouped_k and g is not None:
            xhat = np.empty_like(z)
            sig2 = np.empty_like(z)
            for lab in uniq_g:
                idx = (g == lab)
                if not np.any(idx):
                    continue
                k_lab = float(np.mean(k_mc[idx]))
                xhat[idx] = _prox_qk_vec(z[idx], gp, k_lab)
                sig2[idx] = _sigma2_qk_vec(xhat[idx], gp, k_lab)
        else:
            uniq_k = np.unique(k_mc)
            xhat = np.empty_like(z)
            sig2 = np.empty_like(z)
            if uniq_k.size <= 64:
                for k_val in uniq_k:
                    idx = (k_mc == k_val)
                    if not np.any(idx):
                        continue
                    k_val = float(k_val)
                    xhat[idx] = _prox_qk_vec(z[idx], gp, k_val)
                    sig2[idx] = _sigma2_qk_vec(xhat[idx], gp, k_val)
            else:
                # vectorized path: _prox_qk_vec / _sigma2_qk_vec accept array k
                xhat = _prox_qk_vec(z, gp, k_mc)
                sig2 = _sigma2_qk_vec(xhat, gp, k_mc)

        err2 = (x - xhat) ** 2
        mse = float(err2.mean())
        mean_sig2 = float(sig2.mean())

        s2_new = float(sigma0_2 + beta * mse)
        gp_new = float(gamma_ext + beta * mean_sig2)

        res = max(abs(s2_new - s2), abs(gp_new - gp))
        last_res = res
        if res < float(tol):
            active = float(np.mean(np.abs(xhat) > float(eps_active)))
            mse_se = _batch_means_se(err2, n_batches=int(n_batches_se))
            return mse, active, (s2_new, gp_new), res, mse_se

        s2 = (1.0 - damp) * s2 + damp * s2_new
        gp = (1.0 - damp) * gp + damp * gp_new
        s2 = float(max(s2, sigma0_2))
        gp = float(max(gp, gamma_ext, 1e-14))

    active = float(np.mean(np.abs(xhat) > float(eps_active)))
    mse_se = _batch_means_se(err2, n_batches=int(n_batches_se))
    return float(mse), active, (float(s2), float(gp)), float(last_res), float(mse_se)


# -------------------------
# Homotopy solver using _solve_fp_qk_vec
# -------------------------

def _solve_curve_vec(
    *,
    alphas: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    k_mc: np.ndarray,
    g_mc: Optional[np.ndarray],
    sigma0_2: float,
    gamma_target: float,
    gamma_schedule: Optional[List[float]],
    max_iters: int,
    tol: float,
    damp: float,
    eps_active: float,
    use_grouped_k: bool,
    n_batches_se: int,
    mse_floor_for_db_se: float,
) -> Dict[str, np.ndarray]:
    """
    Identical to solve_curve_with_gamma_homotopy_best_of_fwd_bwd from ptft_replica_qk,
    except each call to solve_fp_qk_one is replaced by _solve_fp_qk_vec.
    """
    alphas = np.asarray(alphas, float)
    betas = 1.0 / alphas

    gt = float(gamma_target)
    if gamma_schedule is None:
        gamma_schedule = _default_gamma_schedule(gt)
    else:
        gamma_schedule = [float(g) for g in gamma_schedule]
        if gamma_schedule[-1] != gt:
            raise ValueError("gamma_schedule must end at gamma_target")
        if any(g < 0 for g in gamma_schedule):
            raise ValueError("gamma_schedule must have nonnegative gammas")

    fwd_states: List[Optional[Tuple[float, float]]] = [None] * betas.size
    bwd_states: List[Optional[Tuple[float, float]]] = [None] * betas.size

    mse_fwd = act_fwd = res_fwd = se_fwd = None
    mse_bwd = act_bwd = res_bwd = se_bwd = None

    for gamma_ext in gamma_schedule:
        mse_fwd = np.empty_like(betas)
        act_fwd = np.empty_like(betas)
        res_fwd = np.empty_like(betas)
        se_fwd = np.empty_like(betas)

        prev_state = None
        for i, b in enumerate(betas):
            init = fwd_states[i] if fwd_states[i] is not None else prev_state
            mse, act, st, res, mse_se = _solve_fp_qk_vec(
                beta=float(b),
                x=x, v=v,
                k_mc=k_mc, g_mc=g_mc,
                sigma0_2=float(sigma0_2),
                gamma_ext=float(gamma_ext),
                init_state=init,
                max_iters=int(max_iters),
                tol=float(tol),
                damp=float(damp),
                eps_active=float(eps_active),
                use_grouped_k=bool(use_grouped_k),
                n_batches_se=int(n_batches_se),
            )
            mse_fwd[i] = mse
            act_fwd[i] = act
            res_fwd[i] = res
            se_fwd[i] = mse_se
            fwd_states[i] = st
            prev_state = st

        mse_bwd_r = np.empty_like(betas)
        act_bwd_r = np.empty_like(betas)
        res_bwd_r = np.empty_like(betas)
        se_bwd_r = np.empty_like(betas)

        prev_state = None
        for j, b in enumerate(betas[::-1]):
            i = betas.size - 1 - j
            init = bwd_states[i] if bwd_states[i] is not None else prev_state
            mse, act, st, res, mse_se = _solve_fp_qk_vec(
                beta=float(b),
                x=x, v=v,
                k_mc=k_mc, g_mc=g_mc,
                sigma0_2=float(sigma0_2),
                gamma_ext=float(gamma_ext),
                init_state=init,
                max_iters=int(max_iters),
                tol=float(tol),
                damp=float(damp),
                eps_active=float(eps_active),
                use_grouped_k=bool(use_grouped_k),
                n_batches_se=int(n_batches_se),
            )
            mse_bwd_r[j] = mse
            act_bwd_r[j] = act
            res_bwd_r[j] = res
            se_bwd_r[j] = mse_se
            bwd_states[i] = st
            prev_state = st

        mse_bwd = mse_bwd_r[::-1]
        act_bwd = act_bwd_r[::-1]
        res_bwd = res_bwd_r[::-1]
        se_bwd = se_bwd_r[::-1]

    diff_db = np.abs(to_db(mse_fwd) - to_db(mse_bwd))
    choose_fwd = mse_fwd <= mse_bwd
    mse_best = np.where(choose_fwd, mse_fwd, mse_bwd)
    act_best = np.where(choose_fwd, act_fwd, act_bwd)
    res_best = np.where(choose_fwd, res_fwd, res_bwd)
    se_best = np.where(choose_fwd, se_fwd, se_bwd)

    mse_floor = float(max(mse_floor_for_db_se, 1e-300))
    rel_se = se_best / np.maximum(mse_best, mse_floor)

    slope = (10.0 / math.log(10.0)) / np.maximum(mse_best, mse_floor)
    se_db = slope * np.nan_to_num(se_best, nan=np.nan)

    return {
        "alpha": alphas,
        "mse_best": mse_best,
        "active_best": act_best,
        "mse_fwd": mse_fwd,
        "mse_bwd": mse_bwd,
        "diff_db": diff_db,
        "fp_residual": res_best,
        "mse_se": se_best,
        "mse_rel_se": rel_se,
        "mse_se_db": se_db,
        "gamma_schedule": np.array(gamma_schedule, dtype=float),
    }


# -------------------------
# Public API
# -------------------------

def ptft_qk_curve_imperfect_pt(
    *,
    # Teacher support / overlap parameters
    rho_pt: float = 0.10,
    rho_ft: float = 0.04,
    omega: float = 1.00,
    # NEW: pretraining quality parameters
    alpha_pt: float,              # pretraining sample ratio; alpha_pt=1 → oracle
    sigma0_pt: float = 0.0,       # PT label noise std dev  (variance = sigma0_pt**2; default noiseless)
    s2_pt_manual: Optional[float] = None,  # if set, override AWGN bypass s2_pt (use empirical pt_param_mse)
    # FT noise
    gamma_ext: float = 1e-6,
    sigma0_2: float = 0.0,
    # Alpha sweep for FT
    alphas: Optional[np.ndarray] = None,
    alpha_min: float = 0.2,
    alpha_max: float = 2.0,
    n_alpha: int = 61,
    # MC parameters
    mc: int = 80_000,
    seed: int = 0,
    # PT parameters controlling k heterogeneity (same as ptft_qk_curve)
    a_pt: float = 1.0,
    c_pt: float = 0.001,
    lambda_pt: float = 0.0,
    gamma_reinit: float = 0.0,
    # Solver parameters
    max_iters: int = 900,
    tol: float = 1e-10,
    damp: float = 0.25,
    eps_active: float = 1e-6,
    use_grouped_k: bool = True,
    n_batches_se: int = 50,
    # Warm start controls (gamma homotopy)
    gamma_schedule: Optional[List[float]] = None,
    mse_floor_for_db_se: float = 1e-12,
    # Reliability score parameters
    score_agg: str = "p95",
    score_z: float = 3.0,
    score_sfp_db: float = 1.0,
    # FT teacher convention
    ft_teacher_norm: str = "unit_total_var",
) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
    """
    PT+FT replica curve with imperfect pretraining (alpha_pt != 1).

    Extends ptft_qk_curve by modelling the pretraining stage as a scalar channel:

        B_hat_PT = prox_{q_{k_PT}/theta_PT}(B*_PT + sqrt(tau_PT) * Z),  Z ~ N(0,1)
        K = Psi(B_hat_PT)

    where (tau_PT, theta_PT) are solved from the PT fixed-point equations with
    sample ratio alpha_pt and noise sigma0_pt.

    When alpha_pt >= 1.0 the system has a unique interpolating solution equal to
    beta*_PT, recovering the oracle (same as ptft_qk_curve).

    Parameters
    ----------
    alpha_pt : float
        Pretraining sample ratio n_PT / D. Use alpha_pt=1.0 for oracle recovery.
    sigma0_pt : float
        Pretraining label noise variance (default 0 = noiseless labels).

    All other parameters are identical to ptft_qk_curve.

    Returns
    -------
    curve, reliability, info
        Same structure as ptft_qk_curve. info is augmented with PT diagnostics:
        s2_pt, gp_pt, res_pt, k_pt, alpha_pt.
    """
    if alphas is None:
        alphas = np.linspace(alpha_min, alpha_max, int(n_alpha), dtype=float)
    else:
        alphas = np.asarray(alphas, float)

    params = PTFTOracleParams(
        rho_pt=float(rho_pt), rho_ft=float(rho_ft), omega=float(omega),
        a_pt=float(a_pt), c_pt=float(c_pt),
        lambda_pt=float(lambda_pt), gamma_reinit=float(gamma_reinit),
    )

    # Step 1: Sample joint FT+PT teacher distribution
    x_ft, _, g_mc, v_ft, info = sample_ptft_oracle_mc(
        int(mc), int(seed), params, ft_teacher_norm=ft_teacher_norm
    )

    # Step 2: Reconstruct beta*_PT from group labels
    beta_pt = np.zeros(int(mc), dtype=np.float64)
    beta_pt[(g_mc == 0) | (g_mc == 2)] = float(a_pt)

    # PT diagnostics defaults (oracle case)
    s2_pt = 0.0
    gp_pt = 0.0
    res_pt = 0.0
    k_pt = 4.0 * float(c_pt) ** 2

    if float(alpha_pt) >= 1.0 and float(sigma0_pt) == 0.0:
        # Oracle shortcut: noiseless square/overdetermined system → exact recovery
        beta_hat_pt = beta_pt.copy()
    elif float(alpha_pt) >= 1.0:
        # Step 3: Independent PT noise Z
        rng_pt = np.random.default_rng(int(seed) ^ 0xDEADBEEF)
        v_pt = rng_pt.normal(size=int(mc))

        # Step 4 (alpha_pt >= 1, noisy): Direct AWGN model — bypass FP solver.
        #
        # When alpha_pt >= 1 (square or overdetermined system), the interpolating
        # PT fixed-point equations s2 = sigma0_pt^2 + (1/alpha_pt)*mse are
        # DEGENERATE at alpha_pt = 1 (no finite fixed point for sigma0_pt > 0),
        # and poorly conditioned for alpha_pt just above 1.
        #
        # Physical picture: with alpha_pt >= 1 samples and small label noise
        # sigma0_pt, the min-norm diagonal-network estimator achieves near-oracle
        # recovery. The residual channel noise is dominated by the label noise
        # directly, i.e. s2_pt = sigma0_pt^2. This matches empirical pt_param_mse
        # which converges to sigma0_pt^2 at alpha_pt = 1.
        # s2_pt_manual overrides this: pass empirical pt_param_mse to test if the
        # FT equations are correct and only the s2_pt assumption is wrong.
        s2_pt = float(s2_pt_manual) if s2_pt_manual is not None else float(sigma0_pt) ** 2
        gp_pt = 1e-6   # near-zero: prox is near-identity
        res_pt = 0.0
        z_pt = beta_pt + math.sqrt(s2_pt) * v_pt
        beta_hat_pt = _prox_qk_vec(z_pt, gp_pt, k_pt)
    else:
        # Step 3: Independent PT noise Z (must be independent of FT noise U = v_ft)
        rng_pt = np.random.default_rng(int(seed) ^ 0xDEADBEEF)
        v_pt = rng_pt.normal(size=int(mc))

        # Step 4 (alpha_pt < 1): Solve PT fixed-point equations
        # Pretraining estimator: prox_{q_{k_PT}/theta_PT}, k_PT = 4*c_pt^2 (homogeneous)
        # gamma_ext_PT = 0 (interpolating PT, no external regularization)
        #
        # Initialization: (s2=0, gp=0) is a spurious trivial fixed point.
        # Warm-start from the prior-variance formula for underdetermined min-norm:
        #   tau_PT_init = (1/alpha_pt - 1) * rho_pt * a_pt^2
        _alpha_pt = float(alpha_pt)
        _rho_pt   = float(rho_pt)
        _a_pt     = float(a_pt)
        # sigma0_pt is std dev; the FP equations take variance = sigma0_pt**2
        s2_pt_init  = (1.0 / _alpha_pt - 1.0) * _rho_pt * (_a_pt ** 2) + float(sigma0_pt) ** 2
        gp_pt_init  = 0.01
        k_pt_mc = np.full(int(mc), k_pt)
        _, _, (s2_pt, gp_pt), res_pt, _ = _solve_fp_qk_vec(
            beta=1.0 / _alpha_pt,
            x=beta_pt,
            v=v_pt,
            k_mc=k_pt_mc,
            g_mc=None,
            sigma0_2=float(sigma0_pt) ** 2,
            gamma_ext=0.0,
            init_state=(s2_pt_init, gp_pt_init),
            use_grouped_k=False,
            max_iters=int(max_iters),
            tol=float(tol),
            damp=float(damp),
            eps_active=float(eps_active),
            n_batches_se=int(n_batches_se),
        )

        # Step 5: Compute B_hat_PT = eta_est_PT(B*_PT + sqrt(tau_PT) * Z)
        z_pt = beta_pt + math.sqrt(max(s2_pt, 1e-15)) * v_pt
        beta_hat_pt = _prox_qk_vec(z_pt, float(gp_pt), k_pt)

    # Step 6: Compute heterogeneous FT penalty K = Psi(B_hat_PT)
    c_ft = compute_c_ft_from_pt(beta_hat_pt, float(c_pt), float(lambda_pt), float(gamma_reinit))
    k_mc = 4.0 * (c_ft ** 2)

    # Step 7: Run FT curve (vectorized path handles continuous k distribution)
    curve = _solve_curve_vec(
        alphas=alphas,
        x=x_ft,
        v=v_ft,
        k_mc=k_mc,
        g_mc=None,           # continuous k: grouping by g_mc would be approximate
        sigma0_2=float(sigma0_2),
        gamma_target=float(gamma_ext),
        gamma_schedule=gamma_schedule,
        max_iters=int(max_iters),
        tol=float(tol),
        damp=float(damp),
        eps_active=float(eps_active),
        use_grouped_k=False,  # use per-sample vectorized path
        n_batches_se=int(n_batches_se),
        mse_floor_for_db_se=float(mse_floor_for_db_se),
    )

    # Step 8: Reliability score (same as ptft_qk_curve)
    score_curve, score_alpha, parts = reliability_score(
        {
            "diff_db": curve["diff_db"],
            "mse_se_db": curve["mse_se_db"],
            "fp_residual": curve["fp_residual"],
        },
        tol=float(tol),
        z=float(score_z),
        s_fp_db=float(score_sfp_db),
        agg=str(score_agg),
    )
    reliability = {
        "score_db": float(score_curve),
        "score_alpha_db": score_alpha,
        "score_parts": parts,
        "agg": score_agg,
        "z": score_z,
        "s_fp_db": score_sfp_db,
        "tol": float(tol),
        "mse_floor_for_db_se": float(mse_floor_for_db_se),
    }

    # Augment info with PT diagnostics
    info["alpha_pt"] = float(alpha_pt)
    info["sigma0_pt"] = float(sigma0_pt)
    info["k_pt"] = float(k_pt)
    info["s2_pt"] = float(s2_pt)
    info["gp_pt"] = float(gp_pt)
    info["res_pt"] = float(res_pt)
    info["oracle"] = float(alpha_pt) >= 1.0 and float(sigma0_pt) == 0.0

    return curve, reliability, info
