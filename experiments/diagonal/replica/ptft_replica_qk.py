# =========================
# File: ptft_replica_qk.py
# =========================
# Compact RS fixed-point solver for q_k with PT->FT hetero-k (PT+FT oracle) + single-task option.
#
# Implements requested fixes:
#  (1) Reliability score no longer explodes when MSE→0:
#      - caps the dB delta-method slope with mse_floor
#      - also tracks an additional "relative" uncertainty channel internally
#  (2) Proper gamma-homotopy warm-start ("option 4") with state reuse:
#      - for gamma_target == 0: run schedule down to 0
#      - for 0 < gamma_target < min(schedule): also run schedule ending at gamma_target
#      - carries (s2,gp) states across gamma steps AND across beta continuation
#  (3) Forward/backward continuation kept; best-of branch selection still used.
#
# Public API:
#   - ptft_qk_curve(...)
#   - single_task_qk_curve(...)
#
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import itertools
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union


# -------------------------
# Utilities
# -------------------------

def to_db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, float), 1e-15))


def _batch_means_se(x: np.ndarray, n_batches: int = 50) -> float:
    """
    Batch-means SE for mean(x). Reasonable for MC expectations with nonlinear prox.
    """
    x = np.asarray(x, float)
    n = x.size
    n_batches = int(max(5, min(n_batches, n // 200)))  # aim for batch size >= ~200
    if n_batches < 5:
        return float("nan")
    m = n // n_batches
    xb = x[:m * n_batches].reshape(n_batches, m).mean(axis=1)
    return float(xb.std(ddof=1) / math.sqrt(n_batches))


def reliability_score(
    curve: Dict[str, np.ndarray],
    *,
    tol: float,
    z: float = 3.0,
    s_fp_db: float = 1.0,
    agg: str = "p95",
) -> Tuple[float, np.ndarray, Dict[str, np.ndarray]]:
    """
    One-number reliability *penalty* in dB (bigger = worse).

    Uses three components:
      - branch mismatch: diff_db
      - MC uncertainty (conservative): z * mse_se_db  (mse_se_db is slope-capped)
      - FP convergence penalty: s_fp_db * log10(1 + fp_residual/tol)

    Aggregation across alpha: p95 / max / median.
    """
    diff_db = np.asarray(curve["diff_db"], float)
    se_db = np.asarray(curve["mse_se_db"], float)      # slope-capped in curve builder
    fp_res = np.asarray(curve["fp_residual"], float)

    c_branch = diff_db
    c_mc = z * np.nan_to_num(se_db, nan=0.0, posinf=np.inf, neginf=np.inf)
    c_fp = s_fp_db * np.log10(1.0 + np.maximum(fp_res, 0.0) / max(float(tol), 1e-300))

    score_alpha = np.maximum.reduce([c_branch, c_mc, c_fp])

    if agg == "max":
        score_curve = float(np.max(score_alpha))
    elif agg == "p95":
        score_curve = float(np.percentile(score_alpha, 95))
    elif agg == "median":
        score_curve = float(np.median(score_alpha))
    else:
        raise ValueError("agg must be one of {'max','p95','median'}")

    parts = {"branch_db": c_branch, "mc_db": c_mc, "fp_db": c_fp}
    return score_curve, score_alpha, parts


# -------------------------
# q_k prox + local variance
# -------------------------

def prox_qk(z: np.ndarray, gp: float, k: float, iters: int = 80, tol: float = 1e-12) -> np.ndarray:
    """
    Solve x - z + 0.5*gp*asinh(2x/sqrt(k)) = 0 by safeguarded Newton/bisection.
    """
    z = np.asarray(z, float)
    gp = float(gp)
    k = float(k)
    if gp <= 0.0:
        return z.copy()

    sk = math.sqrt(k)
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


def sigma2_qk(xhat: np.ndarray, gp: float, k: float) -> np.ndarray:
    gp = float(max(gp, 1e-14))
    qpp = 1.0 / np.sqrt(float(k) + 4.0 * xhat * xhat)
    return 1.0 / (1.0 / gp + qpp)


# -------------------------
# PT -> k_d mapping (Cosyne)
# -------------------------

def compute_c_ft_from_pt(beta_pt: np.ndarray, c_pt: float, lambda_pt: float, gamma_reinit: float) -> np.ndarray:
    beta_pt = np.asarray(beta_pt, dtype=np.float64)
    c_pt = float(c_pt)
    lambda_pt = float(lambda_pt)
    gamma_reinit = float(gamma_reinit)
    if c_pt <= 0.0:
        raise ValueError("c_pt must be > 0")
    ratio_sq = (beta_pt / c_pt) ** 2
    return (lambda_pt + c_pt) * (1.0 + np.sqrt(1.0 + ratio_sq)) + 0.5 * (gamma_reinit ** 2)


# -------------------------
# PT+FT oracle sampler (4 groups)
# -------------------------

@dataclass(frozen=True)
class PTFTOracleParams:
    rho_pt: float
    rho_ft: float
    omega: float
    a_pt: float = 1.0
    c_pt: float = 0.001
    lambda_pt: float = 0.0
    gamma_reinit: float = 0.0


def sample_ptft_oracle_mc(
    mc: int,
    seed: int,
    p: PTFTOracleParams,
    ft_teacher_norm: str = "unit_total_var",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Returns:
      x    : FT ground truth (teacher) [mc]
      k_mc : per-coordinate k [mc]
      g_mc : group labels in {0,1,2,3} [mc]
      v    : N(0,1) noise [mc]
      info : diagnostics
    """
    rho_pt = float(p.rho_pt)
    rho_ft = float(p.rho_ft)
    omega = float(p.omega)
    if not (0.0 < rho_pt < 1.0):
        raise ValueError("rho_pt must be in (0,1)")
    if not (0.0 < rho_ft < 1.0):
        raise ValueError("rho_ft must be in (0,1)")
    if not (0.0 <= omega <= 1.0):
        raise ValueError("omega must be in [0,1]")

    p_ov = omega * rho_ft
    p_new = (1.0 - omega) * rho_ft
    p_ptonly = rho_pt - p_ov
    p_none = 1.0 - rho_pt - p_new
    if min(p_ov, p_new, p_ptonly, p_none) < -1e-12:
        raise ValueError(
            "Infeasible (rho_pt, rho_ft, omega). Need:\n"
            "  omega*rho_ft <= rho_pt\n"
            "  rho_pt + (1-omega)*rho_ft <= 1\n"
            f"Got: p_ov={p_ov:.6g}, p_new={p_new:.6g}, p_ptonly={p_ptonly:.6g}, p_none={p_none:.6g}"
        )

    rng = np.random.default_rng(int(seed))
    probs = np.array([p_ov, p_new, p_ptonly, p_none], dtype=np.float64)
    probs /= probs.sum()

    g_mc = rng.choice(4, size=int(mc), p=probs).astype(np.int32)

    beta_pt = np.zeros(int(mc), dtype=np.float64)
    pt_active = (g_mc == 0) | (g_mc == 2)
    beta_pt[pt_active] = float(p.a_pt)

    x = np.zeros(int(mc), dtype=np.float64)
    ft_nonzero = (g_mc == 0) | (g_mc == 1)
    n_ft = int(ft_nonzero.sum())
    if n_ft > 0:
        # FT teacher amplitude convention (default preserves existing behavior):
        # - "unit_total_var": Var(nonzero)=1/rho_ft so that E[x_i^2]=1 regardless of rho_ft
        # - "unit_nonzero_var": Var(nonzero)=1 so that typical active feature has O(1) magnitude
        ft_teacher_norm = str(ft_teacher_norm)
        if ft_teacher_norm == "unit_total_var":
            sigma_ft = 1.0 / math.sqrt(rho_ft)
        elif ft_teacher_norm == "unit_nonzero_var":
            sigma_ft = 1.0
        else:
            raise ValueError(f"Unknown ft_teacher_norm={ft_teacher_norm!r}")
        x[ft_nonzero] = rng.normal(0.0, sigma_ft, size=n_ft)

    c_ft = compute_c_ft_from_pt(beta_pt, p.c_pt, p.lambda_pt, p.gamma_reinit)
    k_mc = 4.0 * (c_ft ** 2)

    v = rng.normal(size=int(mc))

    info = {
        "group_probs_target": probs,
        "group_fracs_emp": np.array([np.mean(g_mc == i) for i in range(4)], dtype=float),
        "rho_ft_emp": float(np.mean(ft_nonzero)),
        "rho_pt_emp": float(np.mean(pt_active)),
        "k_by_group": {int(i): float(np.mean(k_mc[g_mc == i])) for i in range(4) if np.any(g_mc == i)},
    }
    return x, k_mc, g_mc, v, info


# -------------------------
# Fixed-point solver (one beta) with explicit init_state
# -------------------------

def solve_fp_qk_one(
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
    Returns:
      mse, active_frac, (s2, gp), fp_residual, mse_mc_se
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
                xhat[idx] = prox_qk(z[idx], gp, k_lab)
                sig2[idx] = sigma2_qk(xhat[idx], gp, k_lab)
        else:
            # constant-k or per-sample-k path
            uniq_k = np.unique(k_mc)
            xhat = np.empty_like(z)
            sig2 = np.empty_like(z)
            if uniq_k.size <= 64:
                for k_val in uniq_k:
                    idx = (k_mc == k_val)
                    if not np.any(idx):
                        continue
                    k_val = float(k_val)
                    xhat[idx] = prox_qk(z[idx], gp, k_val)
                    sig2[idx] = sigma2_qk(xhat[idx], gp, k_val)
            else:
                # slow fallback
                for i in range(z.size):
                    k_val = float(k_mc[i])
                    xhat[i] = prox_qk(np.array([z[i]]), gp, k_val)[0]
                    sig2[i] = sigma2_qk(np.array([xhat[i]]), gp, k_val)[0]

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

    # not converged: return last iter stats
    active = float(np.mean(np.abs(xhat) > float(eps_active)))
    mse_se = _batch_means_se(err2, n_batches=int(n_batches_se))
    return float(mse), active, (float(s2), float(gp)), float(last_res), float(mse_se)


# -------------------------
# Gamma schedule + homotopy with state reuse
# -------------------------

def _default_gamma_schedule(gamma_target: float) -> List[float]:
    """
    Geometric schedule down to gamma_target (possibly 0) that stabilizes tiny gammas.

    For gamma_target == 0:
      [..., 1e-8, 1e-10, 1e-12, 0]
    For 0 < gamma_target < 1e-8:
      [..., 1e-8, 1e-10, 1e-12, gamma_target] (strictly decreasing, ends at target)
    For gamma_target >= 1e-8:
      [gamma_target] (no need)
    """
    gt = float(gamma_target)
    if gt < 0:
        raise ValueError("gamma_ext must be >= 0")
    # tune these if desired
    ladder = [1e-6, 1e-8, 1e-10, 1e-12]
    if gt == 0.0:
        return [g for g in ladder] + [0.0]
    if gt >= ladder[0]:
        return [gt]
    # include ladder values above gt, then end at gt
    sched = [g for g in ladder if g > gt]
    sched.append(gt)
    # ensure monotone decreasing (except equal if user passes exactly a ladder point)
    out = []
    for g in sched:
        if not out or g < out[-1] or abs(g - out[-1]) < 1e-30:
            out.append(g)
    return out


def solve_curve_with_gamma_homotopy_best_of_fwd_bwd(
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
    Runs forward/backward continuation at each gamma in schedule, *reusing per-index states*
    across gamma steps. Final output corresponds to gamma_target.

    Returns curve dict with:
      alpha, mse_best, active_best, mse_fwd, mse_bwd, diff_db,
      fp_residual, mse_se, mse_se_db, mse_rel_se
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

    # State arrays: one (s2,gp) per beta index, for fwd and bwd paths
    fwd_states: List[Optional[Tuple[float, float]]] = [None] * betas.size
    bwd_states: List[Optional[Tuple[float, float]]] = [None] * betas.size

    # We will store only the final gamma's curves, but we need intermediate states.
    mse_fwd = act_fwd = res_fwd = se_fwd = None
    mse_bwd = act_bwd = res_bwd = se_bwd = None

    for gamma_ext in gamma_schedule:
        # --- forward ---
        mse_fwd = np.empty_like(betas)
        act_fwd = np.empty_like(betas)
        res_fwd = np.empty_like(betas)
        se_fwd = np.empty_like(betas)

        prev_state = None
        for i, b in enumerate(betas):
            # Prefer the per-index warm state from previous gamma; fallback to continuation prev_state.
            init = fwd_states[i] if fwd_states[i] is not None else prev_state
            mse, act, st, res, mse_se = solve_fp_qk_one(
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

        # --- backward ---
        mse_bwd_r = np.empty_like(betas)
        act_bwd_r = np.empty_like(betas)
        res_bwd_r = np.empty_like(betas)
        se_bwd_r = np.empty_like(betas)

        prev_state = None
        for j, b in enumerate(betas[::-1]):
            i = betas.size - 1 - j
            init = bwd_states[i] if bwd_states[i] is not None else prev_state
            mse, act, st, res, mse_se = solve_fp_qk_one(
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

        # loop continues: next gamma reuses states arrays

    # Now choose best-of branches at final gamma
    diff_db = np.abs(to_db(mse_fwd) - to_db(mse_bwd))
    choose_fwd = mse_fwd <= mse_bwd
    mse_best = np.where(choose_fwd, mse_fwd, mse_bwd)
    act_best = np.where(choose_fwd, act_fwd, act_bwd)
    res_best = np.where(choose_fwd, res_fwd, res_bwd)
    se_best = np.where(choose_fwd, se_fwd, se_bwd)

    # Relative SE in linear space (robust near MSE→0 if floored)
    mse_floor = float(max(mse_floor_for_db_se, 1e-300))
    rel_se = se_best / np.maximum(mse_best, mse_floor)

    # SE in dB (delta method) with slope cap by mse_floor
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
# Public API: PTFT curve + reliability
# -------------------------

def ptft_qk_curve(
    *,
    rho_pt: float = 0.10,
    rho_ft: float = 0.04,
    omega: float = 1.00,
    gamma_ext: float = 1e-6,
    sigma0_2: float = 0.0,
    alphas: Optional[np.ndarray] = None,
    alpha_min: float = 0.2,
    alpha_max: float = 2.0,
    n_alpha: int = 61,
    mc: int = 80_000,
    seed: int = 0,
    # PT parameters controlling k heterogeneity
    a_pt: float = 1.0,
    c_pt: float = 0.001,
    lambda_pt: float = 0.0,
    gamma_reinit: float = 0.0,
    # solver parameters
    max_iters: int = 900,
    tol: float = 1e-10,
    damp: float = 0.25,
    eps_active: float = 1e-6,
    use_grouped_k: bool = True,
    n_batches_se: int = 50,
    # warm start controls (gamma homotopy)
    gamma_schedule: Optional[List[float]] = None,
    mse_floor_for_db_se: float = 1e-12,
    # reliability score parameters
    score_agg: str = "p95",
    score_z: float = 3.0,
    score_sfp_db: float = 1.0,
    # FT teacher convention (default preserves existing behavior)
    ft_teacher_norm: str = "unit_total_var",
) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
    """
    Returns:
      curve: dict of arrays
      reliability: dict with score + components
      sampling_info: dict
    """
    if alphas is None:
        alphas = np.linspace(alpha_min, alpha_max, int(n_alpha), dtype=float)
    else:
        alphas = np.asarray(alphas, float)

    params = PTFTOracleParams(
        rho_pt=float(rho_pt), rho_ft=float(rho_ft), omega=float(omega),
        a_pt=float(a_pt), c_pt=float(c_pt), lambda_pt=float(lambda_pt), gamma_reinit=float(gamma_reinit)
    )
    x, k_mc, g_mc, v, info = sample_ptft_oracle_mc(int(mc), int(seed), params, ft_teacher_norm=ft_teacher_norm)

    curve = solve_curve_with_gamma_homotopy_best_of_fwd_bwd(
        alphas=alphas,
        x=x, v=v,
        k_mc=k_mc,
        g_mc=g_mc if use_grouped_k else None,
        sigma0_2=float(sigma0_2),
        gamma_target=float(gamma_ext),
        gamma_schedule=gamma_schedule,
        max_iters=int(max_iters),
        tol=float(tol),
        damp=float(damp),
        eps_active=float(eps_active),
        use_grouped_k=bool(use_grouped_k),
        n_batches_se=int(n_batches_se),
        mse_floor_for_db_se=float(mse_floor_for_db_se),
    )

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

    return curve, reliability, info


# -------------------------
# Public API: single-task BG with constant k = (2c)^2 = 4c^2
# -------------------------

def single_task_qk_curve(
    *,
    rho: float = 0.1,
    c: float = 0.001,
    gamma_ext: float = 1e-6,
    sigma0_2: float = 0.0,
    alphas: Optional[np.ndarray] = None,
    alpha_min: float = 0.2,
    alpha_max: float = 2.0,
    n_alpha: int = 61,
    mc: int = 80_000,
    seed: int = 0,
    # solver parameters
    max_iters: int = 900,
    tol: float = 1e-10,
    damp: float = 0.25,
    eps_active: float = 1e-6,
    n_batches_se: int = 50,
    # warm start controls (gamma homotopy)
    gamma_schedule: Optional[List[float]] = None,
    mse_floor_for_db_se: float = 1e-12,
    # reliability score parameters
    score_agg: str = "p95",
    score_z: float = 3.0,
    score_sfp_db: float = 1.0,
) -> Tuple[Dict[str, np.ndarray], Dict, Dict]:
    if not (0.0 < float(rho) < 1.0):
        raise ValueError("rho must be in (0,1)")
    c = float(c)
    if c <= 0.0:
        raise ValueError("c must be > 0")

    if alphas is None:
        alphas = np.linspace(alpha_min, alpha_max, int(n_alpha), dtype=float)
    else:
        alphas = np.asarray(alphas, float)

    rng = np.random.default_rng(int(seed))
    x = np.zeros(int(mc), dtype=np.float64)
    active = rng.random(int(mc)) < float(rho)
    if active.any():
        x[active] = rng.normal(0.0, 1.0 / math.sqrt(float(rho)), size=int(active.sum()))
    # external noise for scalar channel
    if float(sigma0_2) > 0:
        v = rng.normal(size=int(mc))
    else:
        v = rng.normal(size=int(mc))  # still needed, multiplied by sqrt(s2) during FP

    k0 = 4.0 * (c ** 2)
    k_mc = np.full(int(mc), k0, dtype=np.float64)

    curve = solve_curve_with_gamma_homotopy_best_of_fwd_bwd(
        alphas=alphas,
        x=x, v=v,
        k_mc=k_mc,
        g_mc=None,
        sigma0_2=float(sigma0_2),
        gamma_target=float(gamma_ext),
        gamma_schedule=gamma_schedule,
        max_iters=int(max_iters),
        tol=float(tol),
        damp=float(damp),
        eps_active=float(eps_active),
        use_grouped_k=False,
        n_batches_se=int(n_batches_se),
        mse_floor_for_db_se=float(mse_floor_for_db_se),
    )

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

    info = {"rho": float(rho), "rho_emp": float(np.mean(active)), "c": float(c), "k0": float(k0)}
    return curve, reliability, info


# -------------------------
# Utility: build dataframe for parameter sweep
# -------------------------

def build_ptft_curves_dataframe(
    *,
    rho_pt: Union[float, List[float]] = 0.10,
    rho_ft: Union[float, List[float]] = 0.04,
    omega: Union[float, List[float]] = 1.00,
    c_pt: Union[float, List[float]] = 0.001,
    lambda_pt: Union[float, List[float]] = 0.0,
    gamma_reinit: Union[float, List[float]] = 0.0,
    # other parameters that remain fixed across all runs
    gamma_ext: float = 1e-6,
    sigma0_2: float = 0.0,
    alphas: Optional[np.ndarray] = None,
    alpha_min: float = 0.2,
    alpha_max: float = 2.0,
    n_alpha: int = 61,
    mc: int = 80_000,
    seed: Union[int, List[int]] = 0,
    a_pt: float = 1.0,
    max_iters: int = 900,
    tol: float = 1e-10,
    damp: float = 0.25,
    eps_active: float = 1e-6,
    use_grouped_k: bool = True,
    n_batches_se: int = 50,
    gamma_schedule: Optional[List[float]] = None,
    mse_floor_for_db_se: float = 1e-12,
    score_agg: str = "p95",
    score_z: float = 3.0,
    score_sfp_db: float = 1.0,
    ft_teacher_norm: str = "unit_total_var",
) -> pd.DataFrame:
    """
    Build a dataframe of curves by running ptft_qk_curve for the Cartesian product
    of all parameter combinations.
    
    Parameters that can be lists or single values:
      - rho_pt: PT sparsity level(s)
      - rho_ft: FT sparsity level(s)
      - omega: overlap parameter(s)
      - c_pt: PT regularization scale(s)
      - lambda_pt: PT lambda parameter(s)
      - gamma_reinit: PT reinitialization noise scale(s)
    
    All other parameters remain fixed across runs.
    
    Returns:
      DataFrame with columns:
        - rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit: parameter values for this run
        - alpha: sample efficiency (one row per (param_combo, alpha) pair)
        - mse_best, active_best, mse_fwd, mse_bwd, diff_db, fp_residual, 
          mse_se, mse_rel_se, mse_se_db: curve outputs
        - reliability_score_db: overall reliability score for this parameter combo
        - plus additional metadata from reliability and sampling_info
    """
    # Convert single values to lists
    def to_list(x):
        return x if isinstance(x, (list, tuple, np.ndarray)) else [x]
    
    rho_pt_list = to_list(rho_pt)
    rho_ft_list = to_list(rho_ft)
    omega_list = to_list(omega)
    c_pt_list = to_list(c_pt)
    lambda_pt_list = to_list(lambda_pt)
    gamma_reinit_list = to_list(gamma_reinit)
    seed_list = to_list(seed)
    
    # Generate all combinations
    param_combinations = list(itertools.product(
        rho_pt_list, rho_ft_list, omega_list, c_pt_list, lambda_pt_list, gamma_reinit_list, seed_list
    ))
    
    results = []
    
    for rho_pt_val, rho_ft_val, omega_val, c_pt_val, lambda_pt_val, gamma_reinit_val, seed_val in param_combinations:
        try:
            curve, reliability, sampling_info = ptft_qk_curve(
                rho_pt=rho_pt_val,
                rho_ft=rho_ft_val,
                omega=omega_val,
                gamma_ext=gamma_ext,
                sigma0_2=sigma0_2,
                alphas=alphas,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                n_alpha=n_alpha,
                mc=mc,
                seed=int(seed_val),
                a_pt=a_pt,
                c_pt=c_pt_val,
                lambda_pt=lambda_pt_val,
                gamma_reinit=gamma_reinit_val,
                max_iters=max_iters,
                tol=tol,
                damp=damp,
                eps_active=eps_active,
                use_grouped_k=use_grouped_k,
                n_batches_se=n_batches_se,
                gamma_schedule=gamma_schedule,
                mse_floor_for_db_se=mse_floor_for_db_se,
                score_agg=score_agg,
                score_z=score_z,
                score_sfp_db=score_sfp_db,
                ft_teacher_norm=ft_teacher_norm,
            )
            
            # Extract curve arrays
            alphas_arr = curve["alpha"]
            n = len(alphas_arr)
            
            # Create one row per (parameter combo, alpha) pair
            for i in range(n):
                row = {
                    # Parameter values
                    "rho_pt": rho_pt_val,
                    "rho_ft": rho_ft_val,
                    "omega": omega_val,
                    "c_pt": c_pt_val,
                    "lambda_pt": lambda_pt_val,
                    "gamma_reinit": gamma_reinit_val,
                    "seed": int(seed_val),
                    "ft_teacher_norm": str(ft_teacher_norm),
                    # Alpha value
                    "alpha": alphas_arr[i],
                    # Curve outputs
                    "mse_best": curve["mse_best"][i],
                    "active_best": curve["active_best"][i],
                    "mse_fwd": curve["mse_fwd"][i],
                    "mse_bwd": curve["mse_bwd"][i],
                    "diff_db": curve["diff_db"][i],
                    "fp_residual": curve["fp_residual"][i],
                    "mse_se": curve["mse_se"][i],
                    "mse_rel_se": curve["mse_rel_se"][i],
                    "mse_se_db": curve["mse_se_db"][i],
                    # Reliability score (same for all alphas in this combo)
                    "reliability_score_db": reliability["score_db"],
                    # Sampling info
                    "rho_pt_emp": sampling_info["rho_pt_emp"],
                    "rho_ft_emp": sampling_info["rho_ft_emp"],
                }
                results.append(row)
                
        except Exception as e:
            print(f"Warning: Failed for params (rho_pt={rho_pt_val}, rho_ft={rho_ft_val}, "
                  f"omega={omega_val}, c_pt={c_pt_val}, lambda_pt={lambda_pt_val}, "
                  f"gamma_reinit={gamma_reinit_val}, seed={seed_val}): {e}")
            continue
    
    return pd.DataFrame(results)


def build_single_task_curves_dataframe(
    *,
    rho: Union[float, List[float]] = 0.1,
    c: Union[float, List[float]] = 0.001,
    # other parameters that remain fixed across all runs
    gamma_ext: float = 1e-6,
    sigma0_2: float = 0.0,
    alphas: Optional[np.ndarray] = None,
    alpha_min: float = 0.2,
    alpha_max: float = 2.0,
    n_alpha: int = 61,
    mc: int = 80_000,
    seed: Union[int, List[int]] = 0,
    max_iters: int = 900,
    tol: float = 1e-10,
    damp: float = 0.25,
    eps_active: float = 1e-6,
    n_batches_se: int = 50,
    gamma_schedule: Optional[List[float]] = None,
    mse_floor_for_db_se: float = 1e-12,
    score_agg: str = "p95",
    score_z: float = 3.0,
    score_sfp_db: float = 1.0,
) -> pd.DataFrame:
    """
    Build a dataframe of curves by running single_task_qk_curve for the Cartesian product
    of all parameter combinations.
    
    Parameters that can be lists or single values:
      - rho: sparsity level(s)
      - c: regularization scale(s)
    
    All other parameters remain fixed across runs.
    
    Returns:
      DataFrame with columns:
        - rho, c: parameter values for this run
        - alpha: sample efficiency (one row per (param_combo, alpha) pair)
        - mse_best, active_best, mse_fwd, mse_bwd, diff_db, fp_residual, 
          mse_se, mse_rel_se, mse_se_db: curve outputs
        - reliability_score_db: overall reliability score for this parameter combo
        - plus additional metadata from reliability and sampling_info
    """
    # Convert single values to lists
    def to_list(x):
        return x if isinstance(x, (list, tuple, np.ndarray)) else [x]
    
    rho_list = to_list(rho)
    c_list = to_list(c)
    seed_list = to_list(seed)
    
    # Generate all combinations
    param_combinations = list(itertools.product(rho_list, c_list, seed_list))
    
    results = []
    
    for rho_val, c_val, seed_val in param_combinations:
        try:
            curve, reliability, sampling_info = single_task_qk_curve(
                rho=rho_val,
                c=c_val,
                gamma_ext=gamma_ext,
                sigma0_2=sigma0_2,
                alphas=alphas,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                n_alpha=n_alpha,
                mc=mc,
                seed=int(seed_val),
                max_iters=max_iters,
                tol=tol,
                damp=damp,
                eps_active=eps_active,
                n_batches_se=n_batches_se,
                gamma_schedule=gamma_schedule,
                mse_floor_for_db_se=mse_floor_for_db_se,
                score_agg=score_agg,
                score_z=score_z,
                score_sfp_db=score_sfp_db,
            )
            
            # Extract curve arrays
            alphas_arr = curve["alpha"]
            n = len(alphas_arr)
            
            # Create one row per (parameter combo, alpha) pair
            for i in range(n):
                row = {
                    # Parameter values
                    "rho": rho_val,
                    "c": c_val,
                    "seed": int(seed_val),
                    # Alpha value
                    "alpha": alphas_arr[i],
                    # Curve outputs
                    "mse_best": curve["mse_best"][i],
                    "active_best": curve["active_best"][i],
                    "mse_fwd": curve["mse_fwd"][i],
                    "mse_bwd": curve["mse_bwd"][i],
                    "diff_db": curve["diff_db"][i],
                    "fp_residual": curve["fp_residual"][i],
                    "mse_se": curve["mse_se"][i],
                    "mse_rel_se": curve["mse_rel_se"][i],
                    "mse_se_db": curve["mse_se_db"][i],
                    # Reliability score (same for all alphas in this combo)
                    "reliability_score_db": reliability["score_db"],
                    # Sampling info
                    "rho_emp": sampling_info["rho_emp"],
                    "k0": sampling_info["k0"],
                }
                results.append(row)
                
        except Exception as e:
            print(f"Warning: Failed for params (rho={rho_val}, c={c_val}): {e}")
            continue
    
    return pd.DataFrame(results)
