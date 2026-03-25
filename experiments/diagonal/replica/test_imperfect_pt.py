#!/usr/bin/env python3
"""
Tests for ptft_replica_imperfect_pt.py

Run from the replica/ directory:
    python test_imperfect_pt.py

Tests:
1. Import and basic shapes
2. Vectorized prox: scalar k matches array k (backward compat)
3. Vectorized prox: array k gives correct per-element result
4. FP solver: constant k matches oracle solve_fp_qk_one
5. Oracle recovery: alpha_pt=1.0 matches ptft_qk_curve exactly
6. PT FP convergence: res_pt < tol for alpha_pt < 1
7. Imperfect PT shifts the curve (k_mc becomes heterogeneous)
8. Speed: vectorized path is not catastrophically slow
"""
import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from ptft_replica_qk import (
    prox_qk,
    sigma2_qk,
    solve_fp_qk_one,
    ptft_qk_curve,
)
from ptft_replica_imperfect_pt import (
    _prox_qk_vec,
    _sigma2_qk_vec,
    _solve_fp_qk_vec,
    ptft_qk_curve_imperfect_pt,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        _failures.append(name)


# -----------------------------------------------------------------------
# Test 1: imports and module-level sanity
# -----------------------------------------------------------------------
print("\n--- Test 1: imports ---")
check("_prox_qk_vec callable", callable(_prox_qk_vec))
check("_sigma2_qk_vec callable", callable(_sigma2_qk_vec))
check("_solve_fp_qk_vec callable", callable(_solve_fp_qk_vec))
check("ptft_qk_curve_imperfect_pt callable", callable(ptft_qk_curve_imperfect_pt))


# -----------------------------------------------------------------------
# Test 2: vectorized prox — scalar k gives same result as original prox_qk
# -----------------------------------------------------------------------
print("\n--- Test 2: _prox_qk_vec scalar k == prox_qk ---")
rng = np.random.default_rng(42)
z_test = rng.normal(size=500)
gp_test = 0.5
k_scalar = 0.01

ref = prox_qk(z_test, gp_test, k_scalar)
got = _prox_qk_vec(z_test, gp_test, k_scalar)
max_err = float(np.max(np.abs(ref - got)))
check("prox scalar k: max_err < 1e-10", max_err < 1e-10, f"max_err={max_err:.2e}")

ref_s = sigma2_qk(ref, gp_test, k_scalar)
got_s = _sigma2_qk_vec(ref, gp_test, k_scalar)
max_err_s = float(np.max(np.abs(ref_s - got_s)))
check("sigma2 scalar k: max_err < 1e-10", max_err_s < 1e-10, f"max_err={max_err_s:.2e}")


# -----------------------------------------------------------------------
# Test 3: vectorized prox — array k: per-element matches scalar calls
# -----------------------------------------------------------------------
print("\n--- Test 3: _prox_qk_vec array k ---")
k_arr = np.abs(rng.normal(0.01, 0.005, size=500)) + 1e-4  # positive k values
expected = np.array([float(prox_qk(np.array([z_test[i]]), gp_test, k_arr[i])[0])
                     for i in range(len(z_test))])
got_arr = _prox_qk_vec(z_test, gp_test, k_arr)
max_err_arr = float(np.max(np.abs(expected - got_arr)))
check("prox array k: max_err < 1e-8", max_err_arr < 1e-8, f"max_err={max_err_arr:.2e}")

expected_s = np.array([float(sigma2_qk(np.array([expected[i]]), gp_test, k_arr[i])[0])
                        for i in range(len(z_test))])
got_s_arr = _sigma2_qk_vec(expected, gp_test, k_arr)
max_err_s2 = float(np.max(np.abs(expected_s - got_s_arr)))
check("sigma2 array k: max_err < 1e-8", max_err_s2 < 1e-8, f"max_err={max_err_s2:.2e}")


# -----------------------------------------------------------------------
# Test 4: _solve_fp_qk_vec with constant k matches solve_fp_qk_one
# -----------------------------------------------------------------------
print("\n--- Test 4: _solve_fp_qk_vec constant k matches solve_fp_qk_one ---")
mc = 20_000
rng2 = np.random.default_rng(7)
x_fp = rng2.choice([0.0, 1.0], size=mc, p=[0.9, 0.1])
v_fp = rng2.normal(size=mc)
k_const = 4.0 * 0.001**2
k_fp_mc = np.full(mc, k_const)

ref_mse, ref_act, (ref_s2, ref_gp), ref_res, _ = solve_fp_qk_one(
    beta=2.0, x=x_fp, v=v_fp, k_mc=k_fp_mc, g_mc=None,
    sigma0_2=0.0, gamma_ext=1e-6, use_grouped_k=False,
)
got_mse, got_act, (got_s2, got_gp), got_res, _ = _solve_fp_qk_vec(
    beta=2.0, x=x_fp, v=v_fp, k_mc=k_fp_mc, g_mc=None,
    sigma0_2=0.0, gamma_ext=1e-6, use_grouped_k=False,
)
check("FP mse match", abs(ref_mse - got_mse) < 1e-8,
      f"ref={ref_mse:.6g} got={got_mse:.6g}")
check("FP s2 match", abs(ref_s2 - got_s2) < 1e-8,
      f"ref={ref_s2:.6g} got={got_s2:.6g}")
check("FP gp match", abs(ref_gp - got_gp) < 1e-8,
      f"ref={ref_gp:.6g} got={got_gp:.6g}")


# -----------------------------------------------------------------------
# Test 5: oracle recovery — alpha_pt=1.0 matches ptft_qk_curve
# -----------------------------------------------------------------------
print("\n--- Test 5: oracle recovery (alpha_pt=1.0 == ptft_qk_curve) ---")
shared_kwargs = dict(
    rho_pt=0.10, rho_ft=0.04, omega=1.0,
    gamma_ext=1e-6, sigma0_2=0.0,
    alpha_min=0.5, alpha_max=1.5, n_alpha=11,
    mc=30_000, seed=0,
    a_pt=1.0, c_pt=0.001, lambda_pt=0.0, gamma_reinit=0.0,
)

curve_oracle, _, info_oracle = ptft_qk_curve(**shared_kwargs)
curve_impf1, _, info_impf1 = ptft_qk_curve_imperfect_pt(alpha_pt=1.0, **shared_kwargs)

max_mse_diff = float(np.max(np.abs(curve_oracle["mse_best"] - curve_impf1["mse_best"])))
check("oracle: mse_best identical", max_mse_diff < 1e-12,
      f"max_diff={max_mse_diff:.2e}")
check("oracle: info['oracle'] == True", info_impf1["oracle"] is True or info_impf1["oracle"] == True)
check("oracle: s2_pt == 0", info_impf1["s2_pt"] == 0.0)


# -----------------------------------------------------------------------
# Test 6: PT FP convergence for alpha_pt < 1
# -----------------------------------------------------------------------
print("\n--- Test 6: PT FP convergence for alpha_pt < 1 ---")
for alpha_test in [0.8, 0.5, 0.2]:
    _, _, info_test = ptft_qk_curve_imperfect_pt(
        alpha_pt=alpha_test,
        rho_pt=0.10, rho_ft=0.04, omega=1.0,
        gamma_ext=1e-6, sigma0_2=0.0,
        alpha_min=0.5, alpha_max=1.5, n_alpha=5,
        mc=20_000, seed=1,
        a_pt=1.0, c_pt=0.001, lambda_pt=0.0, gamma_reinit=0.0,
    )
    res = info_test["res_pt"]
    check(f"PT FP converged (alpha_pt={alpha_test}): res_pt={res:.2e} < 1e-8",
          res < 1e-8, f"res_pt={res:.2e}")


# -----------------------------------------------------------------------
# Test 7: imperfect PT makes k_mc heterogeneous and shifts curve
# -----------------------------------------------------------------------
print("\n--- Test 7: imperfect PT makes k_mc heterogeneous ---")
# We verify this by directly computing beta_hat_pt and checking it differs from
# beta_pt, and that k_mc has more than 2 unique values (oracle has exactly 2).
from ptft_replica_qk import sample_ptft_oracle_mc, PTFTOracleParams, compute_c_ft_from_pt
_pvec = _prox_qk_vec  # alias already imported above

mc_test = 30_000
rng_t7 = np.random.default_rng(99)
params_t7 = PTFTOracleParams(
    rho_pt=0.10, rho_ft=0.04, omega=1.0,
    a_pt=1.0, c_pt=0.001, lambda_pt=0.0, gamma_reinit=0.0,
)
_, _, g_mc_t7, _, _ = sample_ptft_oracle_mc(mc_test, 0, params_t7)
beta_pt_t7 = np.zeros(mc_test)
beta_pt_t7[(g_mc_t7 == 0) | (g_mc_t7 == 2)] = 1.0

# Oracle k_mc has at most 2 distinct values
c_ft_oracle = compute_c_ft_from_pt(beta_pt_t7, 0.001, 0.0, 0.0)
k_mc_oracle = 4.0 * c_ft_oracle**2
n_unique_oracle = len(np.unique(np.round(k_mc_oracle, 10)))
check("oracle k_mc has <= 2 unique values", n_unique_oracle <= 2,
      f"n_unique={n_unique_oracle}")

# With PT noise (sigma0_pt > 0), beta_hat_pt is genuinely random → many unique k values
# Use sigma0_pt=0.5 to force visible PT noise regardless of alpha_pt
alpha_t7 = 0.5
sigma0_pt_t7 = 0.5
v_pt_t7 = rng_t7.normal(size=mc_test)
k_pt_t7 = 4.0 * 0.001**2
k_pt_mc_t7 = np.full(mc_test, k_pt_t7)
_, _, (s2_pt_t7, gp_pt_t7), res_t7, _ = _solve_fp_qk_vec(
    beta=1.0/alpha_t7,
    x=beta_pt_t7, v=v_pt_t7,
    k_mc=k_pt_mc_t7, g_mc=None,
    sigma0_2=sigma0_pt_t7, gamma_ext=0.0,
    use_grouped_k=False,
)
z_pt_t7 = beta_pt_t7 + math.sqrt(max(s2_pt_t7, 1e-15)) * v_pt_t7
beta_hat_pt_t7 = _pvec(z_pt_t7, float(gp_pt_t7), k_pt_t7)

# beta_hat_pt should differ from beta_pt due to PT noise
diff_beta = float(np.std(beta_hat_pt_t7 - beta_pt_t7))
check(f"PT noise (sigma0_pt={sigma0_pt_t7}): beta_hat_pt has nonzero std",
      diff_beta > 1e-6, f"std(beta_hat_pt - beta_pt) = {diff_beta:.4g}")

# k_mc from noisy beta_hat_pt should have many unique values (heterogeneous)
c_ft_noisy = compute_c_ft_from_pt(beta_hat_pt_t7, 0.001, 0.0, 0.0)
k_mc_noisy = 4.0 * c_ft_noisy**2
n_unique_noisy = len(np.unique(np.round(k_mc_noisy, 8)))
check(f"noisy k_mc has many unique values", n_unique_noisy > 100,
      f"n_unique={n_unique_noisy}")

# MSE curve should differ from oracle when PT labels are noisy
curve_noisy, _, info_noisy = ptft_qk_curve_imperfect_pt(
    alpha_pt=alpha_t7, sigma0_pt=sigma0_pt_t7,
    rho_pt=0.10, rho_ft=0.04, omega=1.0,
    gamma_ext=1e-6, sigma0_2=0.0,
    alpha_min=0.5, alpha_max=1.5, n_alpha=11,
    mc=mc_test, seed=0,
    a_pt=1.0, c_pt=0.001, lambda_pt=0.0, gamma_reinit=0.0,
)
mse_diff_noisy = float(np.mean(np.abs(
    curve_oracle["mse_best"] - curve_noisy["mse_best"]
)))
check("noisy PT (sigma0_pt=0.5) shifts MSE curve vs oracle",
      mse_diff_noisy > 1e-6,
      f"mean |Δmse| = {mse_diff_noisy:.2e}")
print(f"    s2_pt={info_noisy['s2_pt']:.4g}, gp_pt={info_noisy['gp_pt']:.4g}, "
      f"res_pt={info_noisy['res_pt']:.2e}")


# -----------------------------------------------------------------------
# Test 8: speed — vectorized path not catastrophically slow
# -----------------------------------------------------------------------
print("\n--- Test 8: speed with heterogeneous k (mc=80_000) ---")
t0 = time.perf_counter()
ptft_qk_curve_imperfect_pt(
    alpha_pt=0.5,
    rho_pt=0.10, rho_ft=0.04, omega=1.0,
    gamma_ext=1e-6, sigma0_2=0.0,
    alpha_min=0.5, alpha_max=1.5, n_alpha=11,
    mc=80_000, seed=0,
    a_pt=1.0, c_pt=0.001, lambda_pt=0.0, gamma_reinit=0.0,
)
elapsed = time.perf_counter() - t0
# Rough threshold: should complete in under 120s (vectorized), vs minutes for loop
check(f"speed: completed in {elapsed:.1f}s (< 120s)", elapsed < 120,
      f"elapsed={elapsed:.1f}s")
print(f"    (wall time: {elapsed:.1f}s)")


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
print()
if _failures:
    print(f"FAILED: {len(_failures)} test(s): {_failures}")
    sys.exit(1)
else:
    n = 8  # number of test sections
    print(f"All tests passed.")
    sys.exit(0)
