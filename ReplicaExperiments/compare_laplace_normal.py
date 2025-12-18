"""
Two-prior comparison (NO abslog, NO optimal MMSE):
Replica-style RS-PMAP tuned curves + empirical markers for:
- Linear (ridge / LMMSE with Var(x)=1)
- LASSO
- l0 (hard-threshold proxy path)
- q-regulariser (two k values)

Two subplots side-by-side:
- Left:  x ~ Normal(0,1)
- Right: x ~ Laplace(0,b) with Var=1 (b = 1/sqrt(2))

Output:
- replica_two_priors_gauss_laplace.png
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================
# Global configuration
# =============================
BASE_SEED = 12345

# Noise: SNR0 = 10 dB with Var(X)=1 => sigma0^2 = 1/SNR
SNR0_dB = 10.0
SNR0 = 10 ** (SNR0_dB / 10.0)
sigma0_2 = 1.0 / SNR0
sigma0 = float(np.sqrt(sigma0_2))

# Measurement ratios beta = n/m (ONLY 10 points now)
betas = np.linspace(0.5, 3.0, 10)

# Monte Carlo for replica expectations
# mc_samples_full = 40_000
mc_samples_full = 10_000

# Fixed point knobs
max_fp_iters = 220
tol_fp = 1e-9
damp = 0.35

# Gamma grids (GLOBAL gamma)
gamma_grid_coarse = np.logspace(-10, 1.0, 120)
refine_half_decades = 0.7
gamma_refine_points = 120

# l0 smoothness weight: larger => smoother gamma(beta) path
l0_smooth_w = 0.12

# q regulariser settings
k_q_list = [25.0, 0.05]

# Empirical simulation knobs
sim_n = 240
sim_trials = 50

# Coordinate descent knobs (lasso, q)
max_cd_iters_lasso = 6000
max_cd_iters_q = 2500
cd_tol = 1e-7

# Logging knobs
PRINT_EVERY_FP_ITERS = 40   # print within solve_rspmap every N iterations when verbose=True


# =============================
# Prior sampling
# =============================
def sample_gaussian(n, rng):
    # Var = 1
    return rng.normal(0.0, 1.0, size=n)

def sample_laplace_var1(n, rng):
    # Laplace(0,b) has Var = 2 b^2, so choose b = 1/sqrt(2)
    b = 1.0 / np.sqrt(2.0)
    return rng.laplace(0.0, b, size=n)


# =============================
# Denoisers / thresholds
# =============================
def soft_threshold(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

def hard_threshold(z, t):
    out = np.zeros_like(z)
    m = np.abs(z) > t
    out[m] = z[m]
    return out


# =============================
# q-regulariser: prox + local variance sigma^2
# =============================
def prox_q_exact_safeguarded(z, lam, k, tol=1e-10, max_iters=80):
    z = np.asarray(z, dtype=float)
    if lam <= 0:
        return z.copy()

    sk = np.sqrt(k)
    lo = np.minimum(z, 0.0)
    hi = np.maximum(z, 0.0)
    x = 0.5 * (lo + hi)

    for _ in range(max_iters):
        Fx = x - z + (lam / sk) * np.arcsinh(x / (2.0 * sk))
        if np.max(np.abs(Fx)) < tol:
            break

        Fpx = 1.0 + (lam / sk) * (1.0 / np.sqrt(4.0 * k + x * x))
        x_new = x - Fx / Fpx

        neg = Fx < 0
        lo = np.where(neg, x, lo)
        hi = np.where(~neg, x, hi)

        out = (x_new < lo) | (x_new > hi) | ~np.isfinite(x_new)
        x = np.where(out, 0.5 * (lo + hi), x_new)

    return x


def prox_q_exact_scalar(z, lam, k, tol=1e-10, max_iters=60):
    if lam <= 0.0:
        return float(z)

    sk = np.sqrt(k)
    lo = min(z, 0.0)
    hi = max(z, 0.0)
    x = 0.5 * (lo + hi)

    for _ in range(max_iters):
        Fx = x - z + (lam / sk) * np.arcsinh(x / (2.0 * sk))
        if abs(Fx) < tol:
            break

        Fpx = 1.0 + (lam / sk) * (1.0 / (np.sqrt(4.0 * k + x * x)))
        x_new = x - Fx / Fpx

        if Fx < 0:
            lo = x
        else:
            hi = x

        if (x_new < lo) or (x_new > hi) or (not np.isfinite(x_new)):
            x = 0.5 * (lo + hi)
        else:
            x = x_new

    return float(x)


def sigma2_q_exact(xstar, lam, k):
    sk = np.sqrt(k)
    fpp = 1.0 / (sk * np.sqrt(4.0 * k + xstar * xstar))
    return 1.0 / (1.0 / lam + fpp)


# =============================
# RS-PMAP coupled fixed point (with optional verbose logging)
# =============================
def solve_rspmap(beta, gamma, mode, x_mc, v_mc,
                init_sigma_eff2=None, init_gamma_p=None,
                k_q=None,
                verbose=False, tag=""):
    sigma_eff2 = float(sigma0_2 if init_sigma_eff2 is None else max(init_sigma_eff2, 1e-15))
    gamma_p = float(max(gamma, 1e-14) if init_gamma_p is None else max(init_gamma_p, 1e-14))

    for it in range(max_fp_iters):
        z = x_mc + np.sqrt(sigma_eff2) * v_mc
        lam = gamma_p

        if mode == "lasso":
            xhat = soft_threshold(z, lam)
            p_act = float(np.mean(np.abs(z) > lam))
            mean_sigma2 = lam * p_act

        elif mode == "l0":
            thr = np.sqrt(2.0 * lam)
            xhat = hard_threshold(z, thr)
            p_act = float(np.mean(np.abs(z) > thr))
            mean_sigma2 = lam * p_act

        elif mode == "q":
            if k_q is None:
                raise ValueError("k_q must be provided for mode='q'")
            xhat = prox_q_exact_safeguarded(z, lam, k_q)
            mean_sigma2 = float(np.mean(sigma2_q_exact(xhat, lam, k_q)))

        else:
            raise ValueError("unknown mode")

        mse = float(np.mean((x_mc - xhat) ** 2))
        sigma_new = sigma0_2 + beta * mse
        gamma_p_new = gamma + beta * mean_sigma2

        if verbose and (it == 0 or (it + 1) % PRINT_EVERY_FP_ITERS == 0):
            print(f"    [FP {tag}] it={it+1:3d} sigma_eff2={sigma_eff2:.3e}->{sigma_new:.3e} "
                  f"gamma_p={gamma_p:.3e}->{gamma_p_new:.3e} mse={mse:.3e}")

        if max(abs(sigma_new - sigma_eff2), abs(gamma_p_new - gamma_p)) < tol_fp:
            sigma_eff2, gamma_p = sigma_new, gamma_p_new
            break

        sigma_eff2 = (1 - damp) * sigma_eff2 + damp * sigma_new
        gamma_p = (1 - damp) * gamma_p + damp * gamma_p_new

    # final mse eval
    z = x_mc + np.sqrt(sigma_eff2) * v_mc
    lam = gamma_p
    if mode == "lasso":
        xhat = soft_threshold(z, lam)
    elif mode == "l0":
        xhat = hard_threshold(z, np.sqrt(2.0 * lam))
    elif mode == "q":
        xhat = prox_q_exact_safeguarded(z, lam, k_q)
    else:
        raise ValueError("unknown mode")

    mse = float(np.mean((x_mc - xhat) ** 2))
    return mse, sigma_eff2, gamma_p


def solve_rspmap_l0_two_start(beta, gamma, x_mc, v_mc, init_state=None, verbose=False, tag=""):
    if init_state is None:
        mse_a, s2_a, gp_a = solve_rspmap(beta, gamma, "l0", x_mc, v_mc, verbose=verbose, tag=tag+"A")
    else:
        mse_a, s2_a, gp_a = solve_rspmap(
            beta, gamma, "l0", x_mc, v_mc,
            init_sigma_eff2=init_state[0], init_gamma_p=init_state[1],
            verbose=verbose, tag=tag+"A"
        )

    mse_b, s2_b, gp_b = solve_rspmap(
        beta, gamma, "l0", x_mc, v_mc,
        init_sigma_eff2=max(10.0 * sigma0_2, 1e-3),
        init_gamma_p=max(gamma, 1e-12),
        verbose=verbose, tag=tag+"B"
    )

    if mse_b < mse_a:
        return mse_b, s2_b, gp_b
    return mse_a, s2_a, gp_a


# =============================
# Gamma tuning (with prints)
# =============================
def tune_gamma_two_stage_convex(beta, mode, x_mc, v_mc, prev_gamma=None, prev_state=None, k_q=None, verbose=False, tag=""):
    # coarse scan
    gammas = np.asarray(gamma_grid_coarse, dtype=float)
    if prev_gamma is None:
        order = gammas
    else:
        start_idx = int(np.argmin(np.abs(np.log(gammas) - np.log(prev_gamma))))
        order = np.concatenate([gammas[start_idx:], gammas[:start_idx][::-1]])

    best = (np.inf, None, None)
    last_state = prev_state

    for j, g in enumerate(order):
        mse, s2, gp = solve_rspmap(
            beta, g, mode, x_mc, v_mc,
            init_sigma_eff2=None if last_state is None else last_state[0],
            init_gamma_p=None if last_state is None else last_state[1],
            k_q=k_q,
            verbose=False
        )
        last_state = (s2, gp)
        if mse < best[0]:
            best = (mse, g, (s2, gp))

        if verbose and (j == 0 or (j + 1) % 40 == 0 or j == len(order) - 1):
            print(f"    [Tune {tag}] coarse {j+1:3d}/{len(order)} best_mse={best[0]:.3e} best_g={best[1]:.2e}")

    _, g_best, state_best = best

    # refine
    lg = np.log10(g_best)
    g_ref = np.logspace(lg - refine_half_decades, lg + refine_half_decades, gamma_refine_points)
    mid = int(np.argmin(np.abs(np.log(g_ref) - np.log(g_best))))
    order2 = np.concatenate([g_ref[mid:], g_ref[:mid][::-1]])

    best2 = (np.inf, None, None)
    last_state = state_best
    for j, g in enumerate(order2):
        mse, s2, gp = solve_rspmap(
            beta, g, mode, x_mc, v_mc,
            init_sigma_eff2=last_state[0], init_gamma_p=last_state[1],
            k_q=k_q,
            verbose=False
        )
        last_state = (s2, gp)
        if mse < best2[0]:
            best2 = (mse, g, (s2, gp))

        if verbose and (j == 0 or (j + 1) % 50 == 0 or j == len(order2) - 1):
            print(f"    [Tune {tag}] refine {j+1:3d}/{len(order2)} best_mse={best2[0]:.3e} best_g={best2[1]:.2e}")

    return best2


def tune_gamma_l0_smooth(beta, x_mc, v_mc, prev_gamma=None, prev_state=None, smooth_w=0.1, verbose=False, tag=""):
    candidates = gamma_grid_coarse.copy()
    if prev_gamma is not None:
        lg = np.log10(prev_gamma)
        local = np.logspace(lg - refine_half_decades, lg + refine_half_decades, gamma_refine_points)
        candidates = np.unique(np.concatenate([candidates, local]))
        candidates.sort()

    if prev_gamma is None:
        order = candidates
    else:
        start_idx = int(np.argmin(np.abs(np.log(candidates) - np.log(prev_gamma))))
        order = np.concatenate([candidates[start_idx:], candidates[:start_idx][::-1]])

    best = (np.inf, None, None)
    last_state = prev_state

    for j, g in enumerate(order):
        mse, s2, gp = solve_rspmap_l0_two_start(beta, g, x_mc, v_mc, init_state=last_state, verbose=False, tag=tag+"-l0-")
        last_state = (s2, gp)

        if prev_gamma is None:
            obj = mse
        else:
            dlog = np.log(g) - np.log(prev_gamma)
            obj = mse + smooth_w * (dlog * dlog)

        if obj < best[0]:
            best = (obj, g, (s2, gp))

        if verbose and (j == 0 or (j + 1) % 50 == 0 or j == len(order) - 1):
            print(f"    [Tune {tag}] l0 {j+1:3d}/{len(order)} best_obj={best[0]:.3e} best_g={best[1]:.2e}")

    _, g_best, state_best = best
    mse_best, s2_best, gp_best = solve_rspmap_l0_two_start(beta, g_best, x_mc, v_mc, init_state=state_best, verbose=verbose, tag=tag+"-l0FP-")
    return mse_best, g_best, (s2_best, gp_best)


# =============================
# Replica baseline: Linear (ridge / LMMSE with Var(x)=1)
# =============================
def linear_curve_replica(betas_arr):
    out = np.zeros_like(betas_arr, dtype=float)
    for i, beta in enumerate(betas_arr):
        s2 = sigma0_2
        for _ in range(max_fp_iters):
            mse = s2 / (1.0 + s2)
            s2_new = sigma0_2 + beta * mse
            if abs(s2_new - s2) < tol_fp:
                break
            s2 = (1 - damp) * s2 + damp * s2_new
        out[i] = s2 / (1.0 + s2)
    return out


# =============================
# Empirical: lasso + q
# =============================
def lasso_coordinate_descent(A, y, gamma, max_iters=6000, tol=1e-7):
    m, n = A.shape
    x = np.zeros(n)
    r = y.copy()
    col_norms2 = np.sum(A ** 2, axis=0)

    for _ in range(max_iters):
        max_update = 0.0
        for j in range(n):
            cj = col_norms2[j]
            if cj == 0.0:
                continue
            aj = A[:, j]
            r += aj * x[j]
            zj = aj.dot(r) / cj
            x_new = soft_threshold(zj, gamma / cj)
            r -= aj * x_new
            max_update = max(max_update, abs(x_new - x[j]))
            x[j] = x_new
        if max_update < tol:
            break
    return x


def q_coordinate_descent(A, y, gamma, k_q, max_iters=2500, tol=1e-7):
    m, n = A.shape
    x = np.zeros(n)
    r = y.copy()
    col_norms2 = np.sum(A ** 2, axis=0)

    for _ in range(max_iters):
        max_update = 0.0
        for j in range(n):
            cj = col_norms2[j]
            if cj == 0.0:
                continue
            aj = A[:, j]
            r += aj * x[j]
            zj = aj.dot(r) / cj
            lam_eff = gamma / cj
            x_new = prox_q_exact_scalar(zj, lam_eff, k_q)
            r -= aj * x_new
            max_update = max(max_update, abs(x_new - x[j]))
            x[j] = x_new
        if max_update < tol:
            break
    return x


def linear_lmmse_fast(A, y, sigma0_2_):
    _, n = A.shape
    G = A.T @ A + sigma0_2_ * np.eye(n)
    return np.linalg.solve(G, A.T @ y)


def empirical_points(betas_arr, gamma_lasso_arr, gamma_q_by_k, rng, sample_x_fn):
    mse_lin = np.zeros_like(betas_arr, dtype=float)
    mse_las = np.zeros_like(betas_arr, dtype=float)
    mse_q = {k: np.zeros_like(betas_arr, dtype=float) for k in gamma_q_by_k.keys()}

    for i, beta in enumerate(betas_arr):
        t0 = time.time()
        m = int(round(sim_n / beta))
        m = max(m, 1)

        gam_las = float(gamma_lasso_arr[i])
        gam_q = {k: float(gamma_q_by_k[k][i]) for k in gamma_q_by_k.keys()}

        mses_lin, mses_las = [], []
        mses_q = {k: [] for k in gamma_q_by_k.keys()}

        print(f"\n[Empirical] beta {i+1}/{len(betas_arr)} = {beta:.2f} (m={m}) "
              f"| g_las={gam_las:.2e}, "
              f"g_q={{" + ", ".join([f"{k:g}:{gam_q[k]:.2e}" for k in gam_q]) + "}}")

        for tr in range(sim_trials):
            A = rng.normal(0.0, 1.0 / np.sqrt(m), size=(m, sim_n))
            x_true = sample_x_fn(sim_n, rng)
            w = rng.normal(0.0, sigma0, size=m)
            y = A @ x_true + w

            x_lin = linear_lmmse_fast(A, y, sigma0_2)
            mses_lin.append(np.mean((x_lin - x_true) ** 2))

            x_las = lasso_coordinate_descent(A, y, gam_las, max_iters=max_cd_iters_lasso, tol=cd_tol)
            mses_las.append(np.mean((x_las - x_true) ** 2))

            for k in gamma_q_by_k.keys():
                x_q = q_coordinate_descent(A, y, gam_q[k], k_q=k, max_iters=max_cd_iters_q, tol=cd_tol)
                mses_q[k].append(np.mean((x_q - x_true) ** 2))

            if (tr + 1) % max(1, sim_trials // 5) == 0 or tr == sim_trials - 1:
                print(f"  trial {tr+1:3d}/{sim_trials}: "
                      f"med(Lin)={np.median(mses_lin):.3e}, "
                      f"med(Las)={np.median(mses_las):.3e}")

        mse_lin[i] = float(np.median(mses_lin))
        mse_las[i] = float(np.median(mses_las))
        for k in gamma_q_by_k.keys():
            mse_q[k][i] = float(np.median(mses_q[k]))

        dt = time.time() - t0
        msg = (
            f"[Empirical DONE] beta={beta:.2f} | Linear={mse_lin[i]:.4g} | "
            f"LASSO={mse_las[i]:.4g} | "
            + " ".join([f"q(k={k:g})={mse_q[k][i]:.4g}" for k in gamma_q_by_k.keys()])
            + f" | time={dt:.1f}s"
        )
        print(msg)

    return mse_lin, mse_las, mse_q


# =============================
# Utilities
# =============================
def to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-15))


# =============================
# Run one prior (replica + empirical)
# =============================
def run_one_prior(prior_name, sample_x_fn, seed):
    rng = np.random.default_rng(seed)

    print(f"\n==============================")
    print(f"=== Prior: {prior_name}")
    print(f"betas: {len(betas)} points | MC full={mc_samples_full} | empirical n={sim_n}, trials={sim_trials}")
    print(f"seed={seed}")
    print(f"==============================\n")

    # MC samples
    t0 = time.time()
    x_mc = sample_x_fn(mc_samples_full, rng)
    v_mc = rng.normal(size=mc_samples_full)
    print(f"[Init] drew MC samples in {time.time()-t0:.2f}s")

    # Linear baseline (same form for any Var(x)=1 prior)
    print("[Replica] computing Linear baseline ...")
    t0 = time.time()
    mse_linear_rep = linear_curve_replica(betas)
    print(f"[Replica] Linear baseline done in {time.time()-t0:.2f}s\n")

    # Replica tuned curves
    mse_lasso_rep = np.zeros_like(betas, dtype=float)
    gamma_lasso_rep = np.zeros_like(betas, dtype=float)

    mse_l0_rep = np.zeros_like(betas, dtype=float)
    gamma_l0_rep = np.zeros_like(betas, dtype=float)

    mse_q_rep = {k: np.zeros_like(betas, dtype=float) for k in k_q_list}
    gamma_q_rep = {k: np.zeros_like(betas, dtype=float) for k in k_q_list}

    prev_g_lasso, prev_state_lasso = None, None
    prev_g_l0, prev_state_l0 = None, None
    prev_g_q = {k: None for k in k_q_list}
    prev_state_q = {k: None for k in k_q_list}

    print("[Replica] tuning curves across betas ...")
    for i, beta in enumerate(betas):
        t_beta = time.time()
        verbose_beta = (i % 5 == 0)  # extra detail occasionally
        tag = f"{prior_name}-beta{beta:.2f}"

        print(f"\n[Replica] {prior_name} beta {i+1}/{len(betas)} = {beta:.2f} (verbose={verbose_beta})")

        # LASSO
        mse, g, st = tune_gamma_two_stage_convex(
            beta, "lasso", x_mc, v_mc,
            prev_gamma=prev_g_lasso, prev_state=prev_state_lasso,
            verbose=verbose_beta, tag=tag+"-lasso"
        )
        mse_lasso_rep[i] = mse
        gamma_lasso_rep[i] = g
        prev_g_lasso, prev_state_lasso = g, st
        print(f"  [LASSO] mse={mse:.3e}, g={g:.2e}")

        # l0 (smooth path)
        mse0, g0, st0 = tune_gamma_l0_smooth(
            beta, x_mc, v_mc,
            prev_gamma=prev_g_l0, prev_state=prev_state_l0, smooth_w=l0_smooth_w,
            verbose=verbose_beta, tag=tag
        )
        mse_l0_rep[i] = mse0
        gamma_l0_rep[i] = g0
        prev_g_l0, prev_state_l0 = g0, st0
        print(f"  [l0]    mse={mse0:.3e}, g={g0:.2e}")

        # q regulariser for each k
        for k in k_q_list:
            mse_qk, g_q, st_q = tune_gamma_two_stage_convex(
                beta, "q", x_mc, v_mc,
                prev_gamma=prev_g_q[k], prev_state=prev_state_q[k],
                k_q=k, verbose=verbose_beta, tag=tag+f"-q{k:g}"
            )
            mse_q_rep[k][i] = mse_qk
            gamma_q_rep[k][i] = g_q
            prev_g_q[k], prev_state_q[k] = g_q, st_q
            print(f"  [q k={k:g}] mse={mse_qk:.3e}, g={g_q:.2e}")

        print(f"[Replica] beta done in {time.time()-t_beta:.1f}s")

    # Empirical markers
    print("\n[Empirical] running simulations ...")
    mse_linear_emp, mse_lasso_emp, mse_q_emp = empirical_points(
        betas, gamma_lasso_rep, gamma_q_rep, rng=rng, sample_x_fn=sample_x_fn
    )

    return {
        "mse_linear_rep": mse_linear_rep,
        "mse_lasso_rep": mse_lasso_rep,
        "mse_l0_rep": mse_l0_rep,
        "mse_q_rep": mse_q_rep,
        "gamma_lasso_rep": gamma_lasso_rep,
        "gamma_l0_rep": gamma_l0_rep,
        "gamma_q_rep": gamma_q_rep,
        "mse_linear_emp": mse_linear_emp,
        "mse_lasso_emp": mse_lasso_emp,
        "mse_q_emp": mse_q_emp,
    }


# =============================
# Main
# =============================
def main():
    print("=== Starting (two priors, no abslog, no MMSE) ===")

    res_gauss = run_one_prior("Gaussian", sample_gaussian, seed=BASE_SEED + 1)
    res_lap = run_one_prior("Laplace(Var=1)", sample_laplace_var1, seed=BASE_SEED + 2)

    print("\n[Plot] saving figure ...")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True)

    def plot_one(ax, title, res):
        # Replica curves
        ax.plot(betas, to_db(res["mse_linear_rep"]), color="blue", linewidth=3.0, label="Linear\n(replica)")
        ax.plot(betas, to_db(res["mse_lasso_rep"]), color="#66ff33", linewidth=3.0, label="Lasso\n(replica)")
        ax.plot(betas, to_db(res["mse_l0_rep"]), color="#e41a1c", linewidth=3.0, linestyle="--", label="Zero\nnorm–reg")

        ax.plot(betas, to_db(res["mse_q_rep"][25.0]), color="orange", linewidth=3.0,
                label=r"$q(x/\sqrt{k})$" + "\n(replica), k=25")
        ax.plot(betas, to_db(res["mse_q_rep"][0.05]), color="#ff9900", linewidth=3.0, linestyle="-.",
                label=r"$q(x/\sqrt{k})$" + "\n(replica), k=0.05")

        # Empirical markers
        ax.plot(betas, to_db(res["mse_linear_emp"]), "o", color="blue", markersize=7,
                markerfacecolor="none", label="Linear\n(sim.)")
        ax.plot(betas, to_db(res["mse_lasso_emp"]), "^", color="#66ff33", markersize=8,
                markerfacecolor="none", markeredgewidth=2.0, label="Lasso\n(sim.)")

        ax.plot(betas, to_db(res["mse_q_emp"][25.0]), "s", color="orange", markersize=7,
                markerfacecolor="none", markeredgewidth=2.0, label="q k=25\n(sim.)")
        ax.plot(betas, to_db(res["mse_q_emp"][0.05]), "x", color="#ff9900", markersize=8,
                markeredgewidth=2.0, label="q k=0.05\n(sim.)")

        ax.grid(True, linestyle=":", linewidth=1.0)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(r"Measurement ratio $\beta = n/m$", fontsize=14)
        ax.set_xlim(betas.min(), betas.max())

    plot_one(axL, r"Prior: $x \sim \mathcal{N}(0,1)$", res_gauss)
    plot_one(axR, r"Prior: $x \sim \mathrm{Laplace}(0, b),\ \mathrm{Var}=1$", res_lap)

    axL.set_ylabel("Median squared error (dB)", fontsize=14)

    # Choose y-limits broad enough to avoid clipping
    all_db = []
    for res in (res_gauss, res_lap):
        all_db.append(to_db(res["mse_linear_rep"]))
        all_db.append(to_db(res["mse_lasso_rep"]))
        all_db.append(to_db(res["mse_l0_rep"]))
        all_db.append(to_db(res["mse_q_rep"][25.0]))
        all_db.append(to_db(res["mse_q_rep"][0.05]))
        all_db.append(to_db(res["mse_linear_emp"]))
        all_db.append(to_db(res["mse_lasso_emp"]))
        all_db.append(to_db(res["mse_q_emp"][25.0]))
        all_db.append(to_db(res["mse_q_emp"][0.05]))
    all_db = np.concatenate(all_db)
    y_min = float(np.min(all_db) - 1.0)
    y_max = float(np.max(all_db) + 1.0)
    axL.set_ylim(y_min, y_max)

    # Shared legend (deduplicate by label)
    handles, labels = axL.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            uniq_h.append(h)
            uniq_l.append(l)
            seen.add(l)

    leg = fig.legend(uniq_h, uniq_l, loc="center left", bbox_to_anchor=(1.01, 0.5),
                     fontsize=11, frameon=True)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.2)

    plt.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])
    out = "replica_two_priors_gauss_laplace.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"[Plot] Saved: {out}")
    print("=== Done ===")


if __name__ == "__main__":
    main()