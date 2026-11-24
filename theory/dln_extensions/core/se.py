"""Sharp asymptotics for the reduced model.

DERIVATION (stage 1). Let A be p x n iid N(0, 1/p), delta = n/p, and q(z) the diagonal-network potential,
q'(z) = arcsinh(z/c), c = sqrt(k)/2. Pretraining on dF is

    A(D) = argmin Q(v)  s.t.  A^T v = A^T v*.

Its Fenchel dual is max_nu <nu, A^T v*> - c sum_j (cosh((A nu)_j) - 1), with A(D)_j = c sinh((A nu)_j). Writing
h = A nu, that is

    h* = argmin_{h in V_n}  sum_j [ c cosh(h_j) - v*_j h_j ],      V_n = range(A),

a separable convex function minimised over a uniformly random n-dimensional subspace. Gordon's comparison
applied to min_nu max_u <u, A nu> - Psi*(u), with ||nu|| and ||u|| as the two scalars, collapses it to a
scalar channel: there are mu, tau > 0 with

    A(D) = prox_{q, mu}(v* + tau g),      g ~ N(0,1),   v* ~ P_0,
    E[prox'] = delta                                                        (E1)
    E[(A(D) - v* - tau g)^2] = tau^2 (1 - delta)                            (E2)

where prox_{q,mu}(x) = argmin_z (mu/2)(z-x)^2 + q(z), i.e. mu(z-x) + arcsinh(z/c) = 0.

Two checks of (E1)-(E2) done by hand:
  * LAZY (c -> inf): prox is multiplication by kappa = mu c/(mu c + 1); (E1) gives kappa = delta, (E2) gives
    tau^2 = V(1-delta)/delta, and the error is E[(A(D)-v*)^2] = V(1-delta). That is the exact projection
    result ||A(D)-v*||^2/||v*||^2 = 1 - delta.
  * RICH (c -> 0): prox is soft thresholding at theta = alpha tau, and the tau -> 0 limit of (E1)-(E2) is
        delta = rho + (1-rho) 2 Phi(-alpha),     rho/(1-rho) = 2[phi(alpha)/alpha - Phi(-alpha)],
    the Donoho-Tanner condition. At rho = 0.2 it gives delta_DT = 0.5114.

STAGE 2, LAZY, IN CLOSED FORM. There the potential is quadratic, so A(D) = P_{V_n} v* and
U = (I - P_{V_f}) A(D) with V_f = range(A_F). Because V_f is a uniformly random f-dimensional subspace
*of V_n* (conditioning on V_n leaves the columns rotation-invariant inside it), the fraction removed is
phi = f/n, not f/p -- which is exactly the data-reuse effect. Hence, with phi the fraction of the forget
set deleted,

    ||A(D) - v*|| / ||v*||  = sqrt(1 - delta)
    ||U|| / ||v*||          = sqrt(delta (1 - phi))          (reused)
    ||U|| / ||v*||          = sqrt(delta (1 - delta phi))    (if F were a fresh draw)
    ||A(D\\F) - v*||/||v*|| = sqrt(1 - delta(1 - phi))
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfc
from scipy.stats import norm

def _scan(gap, lo, hi, n=40):
    """Find a bracketing pair for gap on a log grid, skipping points where it is undefined.

    The fixed-point equations stop being solvable at large noise (the potential's curvature swamps any
    multiplier), so a fixed bracket silently reports 'no root' where there is one further in.
    """
    xs = np.geomspace(lo, hi, n)
    prev_x = prev_v = None
    for x in xs:
        v = gap(x)
        if v is None:
            continue
        if prev_v is not None and prev_v * v <= 0:
            return prev_x, x, prev_v
        prev_x, prev_v = x, v
    return None, None, (prev_v if prev_v is not None else None)


def _nodes(half=12.0, n=2001):
    """Trapezoid on a fine truncated grid: Gauss-Hermite overflows well before the accuracy we need."""
    x = np.linspace(-half, half, n)
    w = np.exp(-0.5 * x * x)
    return x, w / w.sum()


_X, _W = _nodes()


def prox(x, mu, c, iters=70):
    """argmin_z (mu/2)(z-x)^2 + q(z) with q'(z) = arcsinh(z/c).

    Solved in phi = arcsinh(z/c) rather than z: the condition becomes mu c sinh(phi) + phi = mu x, whose left
    side is strictly increasing, and the root is bracketed by 0 and min(mu|x|, arcsinh(|x|/c)). Newton on z
    diverges when mu is tiny and x large, which is exactly the corner the fixed point walks through.
    """
    x = np.asarray(x, float)
    s, ax = np.sign(x), np.abs(x)
    hi = np.minimum(mu * ax, np.arcsinh(ax / c))
    lo = np.zeros_like(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_big = mu * c * np.sinh(mid) + mid > mu * ax
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)
    return s * c * np.sinh(0.5 * (lo + hi))


def prox_deriv(z, mu, c):
    s = np.sqrt(z * z + c * c)
    return mu * s / (mu * s + 1.0)


def _moments(mu, tau, c, rho, amp):
    """E[prox'] and E[(prox - x)^2] under v* ~ P_0, g ~ N(0,1)."""
    # v* in {0, +amp} with weights (1-rho, rho): the sign is symmetric and the prox is odd.
    out_d = out_e = 0.0
    for v, pw in ((0.0, 1.0 - rho), (amp, rho)):
        x = v + tau * _X
        z = prox(x, mu, c)
        out_d += pw * np.sum(_W * prox_deriv(z, mu, c))
        out_e += pw * np.sum(_W * (z - x) ** 2)
    return out_d, out_e


def solve_stage1(delta, k, rho=0.2, sigma=0.0, tau_hi=None, tol=1e-11):
    """Solve (E1)-(E2). Returns dict with tau, mu, and the relative pretraining error."""
    c = np.sqrt(k) / 2
    amp = rho ** -0.5
    V = rho * amp ** 2                      # E[v*^2] = 1 for our teacher

    def mu_of(tau):
        """(E1): E[prox'] = delta. Monotone increasing in mu."""
        f = lambda lm: _moments(np.exp(lm), tau, c, rho, amp)[0] - delta
        a, b = np.log(1e-20), np.log(1e22)
        if f(a) > 0 or f(b) < 0:
            return None
        return np.exp(brentq(f, a, b, xtol=1e-13, rtol=1e-14))

    def gap(tau):
        """(E2) divided by tau^2: both sides vanish quadratically as tau -> 0, so the raw difference is
        all rounding there. The sign of this at tau -> 0 is what decides recovery."""
        mu = mu_of(tau)
        if mu is None:
            return None
        _, e = _moments(mu, tau, c, rho, amp)
        return (e + sigma ** 2) / tau ** 2 - (1 - delta)

    a, b, g_lo = _scan(gap, 1e-6, tau_hi if tau_hi else 100.0)
    if a is None:
        # no tau > 0 root. The gap staying negative means the error shrinks all the way in: exact recovery.
        return dict(tau=0.0, mu=np.inf, err=0.0,
                    phase="recovery" if (g_lo is not None and g_lo < 0) else "no-root")
    tau = brentq(gap, a, b, xtol=1e-14, rtol=1e-13)
    mu = mu_of(tau)
    err = 0.0
    for v, pw in ((0.0, 1.0 - rho), (amp, rho)):
        x = v + tau * _X
        err += pw * np.sum(_W * (prox(x, mu, c) - v) ** 2)
    return dict(tau=tau, mu=mu, err=np.sqrt(err / V), phase="error")


# ----------------------------------------------------------------------------------------------- stage 2
# The same Gordon reduction applies to any "minimise a separable convex function over a random subspace":
#
#   min over h in V (dim ratio gamma) of sum_j [ Xi(h_j) - t_j h_j ]
#     =>  Xi'(h) = prox_{Xi*, m}(t + s g),   E[prox'] = gamma,   E[(Xi'(h) - t - s g)^2] = s^2 (1 - gamma).
#
# Stage 1 is that with Xi(h) = c cosh(h), t = v*, gamma = delta. Stage 2 is
#
#   U = argmin Q(u) - <h*, u>   s.t. u in V_f^perp,
#
# which is the same shape with Xi = q, t = h* = arcsinh(A(D)/c), and subspace ratio gamma_2 = 1 - (f / D),
# where D is the dimension of the space the projection really acts in. If V_f were independent of stage 1,
# D = p. It is not: V_f sits inside V_n, and h* lies in V_n too, so the candidate is D = n, i.e. the removed
# fraction is phi = f/n. The lazy limit settles it analytically in favour of D = n; `nested` selects which.
def prox_xi(y, m, c, iters=50):
    """argmin_phi (m/2)(phi - y)^2 + c cosh(phi). Bisection in phi; the root lies between 0 and y."""
    y = np.asarray(y, float)
    s, ay = np.sign(y), np.abs(y)
    lo, hi = np.zeros_like(ay), ay.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_big = m * (mid - ay) + c * np.sinh(np.minimum(mid, 700.0)) > 0
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)
    return s * 0.5 * (lo + hi)


def _stage1_law(mu, tau, c, rho, amp, n=151, half=8.0):
    """Grid for (v*, A(D), h*) with weights, as produced by stage 1.

    The grid is laid out as [v*=0 block, v*=amp block], each of length n, so the teacher value of row i is
    0 for i < n and amp otherwise -- which is what solve_stage2 relies on to form the population error.
    """
    if tau <= 0:
        vals = np.array([0.0, amp])
        return vals, np.array([1 - rho, rho]), np.arcsinh(vals / c)
    g = np.linspace(-half, half, n)
    w = np.exp(-0.5 * g * g)
    w = w / w.sum()
    A, W = [], []
    for v, pw in ((0.0, 1.0 - rho), (amp, rho)):
        A.append(prox(v + tau * g, mu, c))
        W.append(pw * w)
    A, W = np.concatenate(A), np.concatenate(W)
    return A, W, np.arcsinh(A / c)


def solve_stage2(delta, phi_frac, k, rho=0.2, nested=True, n_z=151, n_g=151, half=8.0):
    """phi_frac = |F| / n_F. Returns the limiting ||U|| / ||v*|| and the stage-2 constants.

    The two conditions are solved as one 2-D root in (log m, log tau2), started from the lazy closed form;
    a scan is the fallback. Nesting enters only through gamma2.
    """
    from scipy.optimize import root as _root
    c = np.sqrt(k) / 2
    amp = rho ** -0.5
    s1 = solve_stage1(delta, k, rho)
    A_D, W, h = _stage1_law(s1["mu"], s1["tau"], c, rho, amp, n=n_g, half=half)
    gamma2 = 1.0 - (phi_frac if nested else phi_frac * delta)
    psi = 1.0 - gamma2
    if psi <= 0:
        return dict(tau2=0.0, mu2=np.inf, dUL=float(np.sqrt(np.sum(W * A_D ** 2) / (rho * amp ** 2))),
                    phase="no-deletion", stage1=s1)

    z = np.linspace(-half, half, n_z)
    wz = np.exp(-0.5 * z * z)
    wz = wz / wz.sum()
    WW = W[:, None] * wz[None, :]
    Eh2 = float(np.sum(W * h ** 2))

    def pieces(m, t2):
        y = h[:, None] + t2 * z[None, :]
        ph = prox_xi(y, m, c)
        ch = np.cosh(np.minimum(np.abs(ph), 700.0))
        return ph, y, float(np.sum(WW * (m / (m + c * ch)))), float(np.sum(WW * (ph - y) ** 2))

    def resid(lv):
        m, t2 = np.exp(lv[0]), np.exp(lv[1])
        _, _, dr, e = pieces(m, t2)
        return [dr - gamma2, e / t2 ** 2 - psi]

    start = [np.log(max(c * gamma2 / psi, 1e-12)), 0.5 * np.log(max(psi * Eh2 / gamma2, 1e-24))]
    best = None
    for scale in (1.0, 0.1, 10.0, 0.01, 100.0):
        sol = _root(resid, [start[0] + np.log(scale), start[1]], method="hybr", tol=1e-12)
        if sol.success and max(abs(np.array(resid(sol.x)))) < 1e-8:
            best = sol.x
            break
    if best is None:
        return dict(tau2=np.nan, mu2=np.nan, dUL=np.nan, phase="unsolved", stage1=s1)
    m, t2 = np.exp(best[0]), np.exp(best[1])
    ph, _, _, _ = pieces(m, t2)
    U = np.sign(ph) * c * np.sinh(np.minimum(np.abs(ph), 700.0))
    # dUL = distance to the never-saw-the-class reference (0 on dF); dPOP = error on the forget distribution
    V = rho * amp ** 2
    vgrid = np.concatenate([np.zeros(len(A_D) // 2), np.full(len(A_D) - len(A_D) // 2, amp)])
    return dict(tau2=t2, mu2=m,
                dUL=float(np.sqrt(np.sum(WW * U ** 2) / V)),
                dPOP=float(np.sqrt(np.sum(WW * (U - vgrid[:, None]) ** 2) / V)),
                phase="ok", stage1=s1)


# ------------------------------------------------------------------------------- rich limit: Donoho-Tanner
def dt_delta(rho):
    """delta_DT(rho): the rich-regime recovery threshold, from the tau -> 0 limit of (E1)-(E2)."""
    f = lambda a: 2 * (norm.pdf(a) / a - norm.cdf(-a)) - rho / (1 - rho)
    alpha = brentq(f, 1e-6, 12.0)
    return rho + (1 - rho) * 2 * norm.cdf(-alpha), alpha


# ------------------------------------------------------------------------------------ lazy closed forms
def lazy(delta, phi):
    """Exact lazy asymptotics, all relative to ||v*||. phi = |F| / n_F.

    A(D) = P_Vn v* and U = (I - P_Vf) A(D). Reused means V_f sits inside V_n, so it takes a fraction
    phi = f/n of A(D); fresh means V_f is uniform in R^p, so it takes only f/p = delta phi.
    """
    return dict(
        pretrain=np.sqrt(1 - delta),
        dUL=np.sqrt(delta * (1 - phi)),
        dUL_fresh=np.sqrt(delta * (1 - delta * phi)),
        dPOP=np.sqrt((1 - delta) + delta * phi),
        dPOP_fresh=np.sqrt((1 - delta) + delta ** 2 * phi),
        untrain_ref=np.sqrt(1 - delta * (1 - phi)),
    )


if __name__ == "__main__":
    d, a = dt_delta(0.2)
    print(f"Donoho-Tanner at rho=0.2: delta_DT = {d:.4f}  (alpha = {a:.4f})")
    for rho in (0.05, 0.1, 0.2, 0.3):
        print(f"  rho={rho}: delta_DT = {dt_delta(rho)[0]:.4f}")


def solve_stage2_tilt(tilt, weights, teacher, gamma2, k, n_z=201, half=8.0):
    """Stage 2 from an ARBITRARY tilt law, instead of deriving the tilt from stage 1.

    The reused arm obeys U = c sinh(y + A_F zeta) exactly, with y = A_K nu_K the dual field of the KEPT
    samples (exp4_tilt.py). It is therefore the same fixed point as the fresh arm with the tilt h replaced
    by y. This lets the stage-2 machinery be tested against the reused arm while the law of y is supplied
    empirically, separating "is stage 2 right" from "do we know the law of y".

    tilt, weights, teacher: equal-length arrays giving a weighted sample of (field, prob, v*).
    gamma2: 1 - f/p.
    """
    from scipy.optimize import root as _root
    c = np.sqrt(k) / 2
    psi = 1.0 - gamma2
    z = np.linspace(-half, half, n_z)
    wz = np.exp(-0.5 * z * z)
    wz = wz / wz.sum()
    WW = np.asarray(weights, float)[:, None] * wz[None, :]
    WW = WW / WW.sum()
    h = np.asarray(tilt, float)
    tv = np.asarray(teacher, float)
    Eh2 = float(np.sum(np.asarray(weights, float) * h ** 2) / np.sum(weights))

    def pieces(m, t2):
        y = h[:, None] + t2 * z[None, :]
        ph = prox_xi(y, m, c)
        ch = np.cosh(np.minimum(np.abs(ph), 700.0))
        return ph, float(np.sum(WW * (m / (m + c * ch)))), float(np.sum(WW * (ph - y) ** 2))

    def resid(lv):
        m, t2 = np.exp(lv[0]), np.exp(lv[1])
        _, dr, e = pieces(m, t2)
        return [dr - gamma2, e / t2 ** 2 - psi]

    start = [np.log(max(c * gamma2 / psi, 1e-12)), 0.5 * np.log(max(psi * Eh2 / gamma2, 1e-24))]
    best = None
    for sc in (1.0, 0.1, 10.0, 0.01, 100.0, 1e-3, 1e3):
        sol = _root(resid, [start[0] + np.log(sc), start[1]], method="hybr", tol=1e-12)
        if sol.success and max(abs(np.array(resid(sol.x)))) < 1e-8:
            best = sol.x
            break
    if best is None:
        return dict(dUL=np.nan, phase="unsolved")
    m, t2 = np.exp(best[0]), np.exp(best[1])
    ph, _, _ = pieces(m, t2)
    U = np.sign(ph) * c * np.sinh(np.minimum(np.abs(ph), 700.0))
    V = float(np.sum(np.asarray(weights, float) * tv ** 2) / np.sum(weights))
    return dict(dUL=float(np.sqrt(np.sum(WW * U ** 2) / V)), mu2=m, tau2=t2, phase="ok")


def solve_stage2_corr(delta, phi_frac, k, rho=0.2, eta=0.0, n_g=81, n_z=81, half=7.0):
    """Stage 2 with the stage-2 Gaussian CORRELATED with the stage-1 Gaussian.

    Fresh deletion samples are independent of pretraining, so eta = 0 and this reduces to solve_stage2.
    Reused samples are the ones that built A(D), so the two effective noises are correlated; following
    Clerckx & Bellec (arXiv 2404.02070), that whole dependence should collapse to one scalar eta.

        A(D) = prox_{q,mu1}(v* + tau1 G1)
        Q'(U) = prox_{Xi,mu2}(h + tau2 G2),   G2 = eta G1 + sqrt(1-eta^2) G3,   h = Q'(A(D))
        E[prox2'] = gamma2 = 1 - f/p,   E[(Q'(U) - h - tau2 G2)^2] = tau2^2 (1 - gamma2)
    """
    from scipy.optimize import root as _root
    c = np.sqrt(k) / 2
    amp = rho ** -0.5
    s1 = solve_stage1(delta, k, rho)
    mu1, tau1 = s1["mu"], s1["tau"]
    gamma2 = 1.0 - phi_frac * delta
    psi = 1.0 - gamma2

    g1 = np.linspace(-half, half, n_g)
    w1 = np.exp(-0.5 * g1 ** 2)
    w1 /= w1.sum()
    g3 = np.linspace(-half, half, n_z)
    w3 = np.exp(-0.5 * g3 ** 2)
    w3 /= w3.sum()

    H, W, TV = [], [], []
    for v, pw in ((0.0, 1.0 - rho), (amp, rho)):
        A_D = v + tau1 * g1 if tau1 <= 0 else prox(v + tau1 * g1, mu1, c)
        H.append(np.arcsinh(A_D / c))
        W.append(pw * w1)
        TV.append(np.full(n_g, v))
    h = np.concatenate(H)
    wts = np.concatenate(W)
    tv = np.concatenate(TV)
    G1 = np.tile(g1, 2)
    WW = wts[:, None] * w3[None, :]
    G2 = eta * G1[:, None] + np.sqrt(max(1 - eta ** 2, 0.0)) * g3[None, :]

    def pieces(m, t2):
        y = h[:, None] + t2 * G2
        ph = prox_xi(y, m, c)
        ch = np.cosh(np.minimum(np.abs(ph), 700.0))
        return ph, float(np.sum(WW * (m / (m + c * ch)))), float(np.sum(WW * (ph - y) ** 2))

    def resid(lv):
        m, t2 = np.exp(lv[0]), np.exp(lv[1])
        _, dr, e = pieces(m, t2)
        return [dr - gamma2, e / t2 ** 2 - psi]

    Eh2 = float(np.sum(wts * h ** 2))
    start = [np.log(max(c * gamma2 / psi, 1e-12)), 0.5 * np.log(max(psi * Eh2 / gamma2, 1e-24))]
    best = None
    for sc in (1.0, 0.1, 10.0, 0.01, 100.0, 1e-3, 1e3, 1e-5):
        sol = _root(resid, [start[0] + np.log(sc), start[1]], method="hybr", tol=1e-12)
        if sol.success and max(abs(np.array(resid(sol.x)))) < 1e-7:
            best = sol.x
            break
    if best is None:
        return dict(dUL=np.nan, phase="unsolved")
    m, t2 = np.exp(best[0]), np.exp(best[1])
    ph, _, _ = pieces(m, t2)
    U = np.sign(ph) * c * np.sinh(np.minimum(np.abs(ph), 700.0))
    V = rho * amp ** 2
    return dict(dUL=float(np.sqrt(np.sum(WW * U ** 2) / V)), mu2=m, tau2=t2, eta=eta, phase="ok")


def stage1_norm(delta, k, rho=0.2, sigma=0.0):
    """E[A(D)^2] / E[v*^2]: the SQUARED NORM of the pretrained model, not its error.

    Needed for the rich regime above the collapse threshold, where U = 0 exactly and therefore
        ||U - A(D\\F)||^2 = ||A(D\\F)||^2,
    a pure stage-1 quantity at the reduced ratio alpha' = alpha_PT - alpha_UL.
    Same fixed point as solve_stage1, different functional of the law A = prox(v* + tau g).
    """
    s = solve_stage1(delta, k, rho, sigma)
    if s["phase"] == "recovery":
        return 1.0
    if not np.isfinite(s["mu"]):
        return float("nan")
    c = np.sqrt(k) / 2
    amp = rho ** -0.5
    V = rho * amp ** 2
    val = 0.0
    for v, pw in ((0.0, 1.0 - rho), (amp, rho)):
        val += pw * np.sum(_W * prox(v + s["tau"] * _X, s["mu"], c) ** 2)
    return float(val / V)


def stage1_Q(delta, k, rho=0.2, sigma=0.0):
    """E[Q_k(A(D))] per coordinate, under the stage-1 law A = prox(v* + tau g).

    Pythagoras for Bregman projections onto AFFINE sets gives, exactly and at any k,
        D_Q(A(D), A(D\\F)) = Q(A(D)) - Q(A(D\\F)),
    since A(D) lies in the smaller constraint set. So the untraining do-nothing curve is
        stage1_Q(alpha_PT) - stage1_Q(alpha_PT - alpha_UL),
    measured in Bregman divergence rather than squared norm.
    """
    c = np.sqrt(k) / 2
    Qf = lambda b: 0.5 * (b * np.arcsinh(b / c) - np.sqrt(b * b + c * c) + c)
    s = solve_stage1(delta, k, rho, sigma)
    amp = rho ** -0.5
    if s["phase"] == "recovery":
        return float(rho * Qf(amp) + (1 - rho) * Qf(0.0))
    if not np.isfinite(s["mu"]):
        return float("nan")
    val = 0.0
    for v, pw in ((0.0, 1.0 - rho), (amp, rho)):
        val += pw * np.sum(_W * Qf(prox(v + s["tau"] * _X, s["mu"], c)))
    return float(val)


_NEST_M = 400_000


def _nest_draws(seed=0, M=_NEST_M, rho=0.2):
    r = np.random.default_rng(seed)
    amp = rho ** -0.5
    return (r.standard_normal(M), r.standard_normal(M),
            np.where(r.random(M) < rho, r.choice([-amp, amp], M), 0.0))


def nested_pair(alpha, alpha_prime, k, rho=0.2, r=None, draws=None):
    """Joint law of (beta*, A(D), A(D\\F)) for NESTED datasets, at any k.

    A(D) and A(D\\F) solve the same program on nested constraint sets, so their effective pre-prox
    noises are correlated. The correlation r is NOT tau/tau' in general; it is pinned exactly by

        A_K^T (A(D) - A(D\\F)) = 0     (both interpolate the kept columns)
        => contracting with the dual grad Q(A(D\\F)) = A_K omega:
        E[ Q'(A(D\\F)) * (A(D) - A(D\\F)) ] = 0.

    One scalar equation, one unknown, solved by bisection. In the lazy limit the root coincides with
    tau/tau', which is the exact answer there (nested orthogonal projections).

    Returns (beta*, A(D), A(D\\F), r).
    """
    from scipy.optimize import brentq
    g1, h1, bs = draws if draws is not None else _nest_draws(rho=rho)
    s, sp = solve_stage1(alpha, k, rho), solve_stage1(alpha_prime, k, rho)
    c = np.sqrt(k) / 2
    tau, mu, taup, mup = s["tau"], s["mu"], sp["tau"], sp["mu"]

    def build(rr):
        g2 = rr * g1 + np.sqrt(max(1.0 - rr * rr, 0.0)) * h1
        return prox(bs + tau * g1, mu, c), prox(bs + taup * g2, mup, c)

    def gap(rr):
        A, AK = build(rr)
        return float(np.mean(0.5 * np.arcsinh(AK / c) * (A - AK)))

    if r is None:
        try:
            r = brentq(gap, 0.0, 0.999999, xtol=1e-5)
        except Exception:
            r = min(tau / taup, 1.0) if taup > 0 else 1.0
    A, AK = build(r)
    return bs, A, AK, float(r)


def untrain_donothing(alpha_PT, alpha_UL, k, rho=0.2, draws=None):
    """||A(D) - A(D\\F)||^2 / ||beta*||^2 at any k: the do-nothing curve of the untraining row."""
    bs, A, AK, _ = nested_pair(alpha_PT, alpha_PT - alpha_UL, k, rho, draws=draws)
    return float(np.mean((A - AK) ** 2) / np.mean(bs ** 2))


def _nest_grid(rho, n=61, half=6.0):
    """Deterministic weighted grid over (beta*, g, h) for the nested-pair expectations.

    Much faster and smoother than Monte Carlo, which matters because the correlation r is found by a
    root-find that re-evaluates the whole expectation each step.
    """
    z = np.linspace(-half, half, n)
    w = np.exp(-0.5 * z * z)
    w = w / w.sum()
    amp = rho ** -0.5
    bs = np.array([0.0, amp])
    wb = np.array([1.0 - rho, rho])
    B = bs[:, None, None] * np.ones((1, n, n))
    G = z[None, :, None] * np.ones((2, 1, n))
    H = z[None, None, :] * np.ones((2, n, 1))
    W = wb[:, None, None] * w[None, :, None] * w[None, None, :]
    return B.ravel(), G.ravel(), H.ravel(), W.ravel()


def _s1_state(alpha, k, rho):
    """solve_stage1 plus a flag saying whether the estimator is degenerate.

    Three cases the fixed point cannot be evaluated through prox():
      alpha <= 0        no data, so A = 0 identically;
      phase 'recovery'  tau = 0 and mu = inf, so A = beta* identically (prox is the identity);
      phase 'no-root'   the solver found nothing -- report it rather than pretending it is recovery.
    mu = inf also makes prox() return nan at beta* = 0 (inf * 0), which is where the all-nan columns and
    the 'invalid value in multiply' warnings came from.
    """
    if alpha <= 1e-12:
        return dict(tau=0.0, mu=np.inf, phase="zero"), "zero"
    s = solve_stage1(float(alpha), k, rho)
    if s["phase"] == "recovery":
        return s, "exact"
    if s["phase"] == "no-root" or not np.isfinite(s["mu"]) or s["tau"] <= 0:
        return s, "bad"
    return s, "ok"


def nested_pair_g(alpha, alpha_prime, k, rho=0.2, grid=None, r=None):
    """Grid version of nested_pair. Returns (beta*, A(D), A(D\\F), weights, r)."""
    from scipy.optimize import brentq
    B, G, H, W = grid if grid is not None else _nest_grid(rho)
    c = np.sqrt(k) / 2
    s, kind = _s1_state(alpha, k, rho)
    sp, kindp = _s1_state(alpha_prime, k, rho)
    tau, mu, taup, mup = s["tau"], s["mu"], sp["tau"], sp["mu"]

    def apply(kd, t, m, gz):
        if kd == "zero":
            return np.zeros_like(B)
        if kd == "exact":
            return B.copy()
        if kd == "bad":
            return np.full_like(B, np.nan)
        return prox(B + t * gz, m, c)

    A = apply(kind, tau, mu, G)
    if kindp != "ok" or kind != "ok":          # A(D\F) is deterministic given beta*: no coupling to pin
        return B, A, apply(kindp, taup, mup, G), W, float("nan")

    def build(rr):
        g2 = rr * G + np.sqrt(max(1.0 - rr * rr, 0.0)) * H
        return prox(B + taup * g2, mup, c)

    def gap(rr):
        AK = build(rr)
        return float(np.sum(W * 0.5 * np.arcsinh(AK / c) * (A - AK)))

    if r is None:
        try:
            r = brentq(gap, 0.0, 0.999999, xtol=2e-4)
        except Exception:
            r = min(tau / taup, 1.0) if taup > 0 else 1.0
    return B, A, build(r), W, float(r)


def untrain_curves(alpha_PT, alpha_UL, k, rho=0.2, grid=None):
    """Any-k theory for the untraining row (row 1), FRESH arm plus the do-nothing reference.

      do_nothing : ||A(D) - A(D\\F)||^2 / ||beta*||^2
      fresh      : ||U - A(D\\F)||^2 / ||beta*||^2, U from the stage-2 mirror map with independent noise

    The reused arm is not covered (its design is not independent of pretraining).
    """
    B, A, AK, W, r = nested_pair_g(alpha_PT, alpha_PT - alpha_UL, k, rho, grid=grid)
    V = float(np.sum(W * B ** 2))
    out = dict(do_nothing=float(np.sum(W * (A - AK) ** 2) / V), r=r, fresh=np.nan, fresh_UL=np.nan)
    s2 = solve_stage2(alpha_PT, alpha_UL / alpha_PT, k, rho, nested=False)
    if s2.get("phase") == "ok":
        c = np.sqrt(k) / 2
        nz = 41
        zz = np.linspace(-6.0, 6.0, nz)
        wz = np.exp(-0.5 * zz * zz)
        wz = wz / wz.sum()
        ph = prox_xi(np.arcsinh(A / c)[:, None] + s2["tau2"] * zz[None, :], s2["mu2"], c)
        U = np.sign(ph) * c * np.sinh(np.minimum(np.abs(ph), 700.0))
        WW = W[:, None] * wz[None, :]
        out["fresh"] = float(np.sum(WW * (U - AK[:, None]) ** 2) / V)
        out["fresh_UL"] = float(np.sum(WW * U ** 2) / V)
    return out


# ------------------------------------------------------------------- stage 2 with the two ratios split
def solve_stage2_split(delta, k, rho, gam, psi, n_z=151, n_g=151, half=8.0, s1=None):
    """Stage 2 with the dimension count and the energy balance carrying DIFFERENT ratios.

    The Gordon reduction for 'minimise a separable convex function over a random subspace' closes on

        E[prox']            = gam     (C1)  how many directions the solution still has
        E[(prox - y)^2]/t2^2 = psi    (C2)  how much tilt the multiplier has to absorb

    For a FRESH design both are set by the same number: the constraint subspace is uniform and
    independent of the tilt h, so it removes f/d of the directions and f/d of ||h||^2, giving
    gam = 1 - alpha_UL, psi = alpha_UL.

    For a REUSED design that is no longer true. V_f still has dimension f, so (C1) is unchanged, but
    V_f lies inside V_n = col(X_D) and h = X_D nu lies in V_n too, so the constraint absorbs
    ||P_{V_f} h||^2 = phi ||h||^2 with phi = f/n, not alpha_UL ||h||^2. The two ratios come apart.

    Setting both to 1 - phi / phi (the old `nested` flag) is what over-deletes: it inflates the
    DIMENSION count from f to phi*d as well, which in the rich regime is the difference between a few
    constraints and enough to collapse the model.

    Returns ||U||^2 / ||beta*||^2 (not the root), plus the constants.
    """
    from scipy.optimize import root as _root
    c = np.sqrt(k) / 2
    amp = rho ** -0.5
    s1 = s1 if s1 is not None else solve_stage1(delta, k, rho)
    A_D, W, h = _stage1_law(s1["mu"], s1["tau"], c, rho, amp, n=n_g, half=half)
    V = rho * amp ** 2
    if psi <= 0:
        return dict(U2=float(np.sum(W * A_D ** 2) / V), tau2=0.0, mu2=np.inf, phase="no-deletion")
    z = np.linspace(-half, half, n_z)
    wz = np.exp(-0.5 * z * z)
    wz = wz / wz.sum()
    WW = W[:, None] * wz[None, :]
    Eh2 = float(np.sum(W * h ** 2))

    def pieces(m, t2):
        y = h[:, None] + t2 * z[None, :]
        ph = prox_xi(y, m, c)
        ch = np.cosh(np.minimum(np.abs(ph), 700.0))
        return ph, float(np.sum(WW * (m / (m + c * ch)))), float(np.sum(WW * (ph - y) ** 2))

    def resid(lv):
        m, t2 = np.exp(lv[0]), np.exp(lv[1])
        _, dr, e = pieces(m, t2)
        return [dr - gam, e / t2 ** 2 - psi]

    start = [np.log(max(c * gam / max(psi, 1e-12), 1e-12)),
             0.5 * np.log(max(psi * Eh2 / max(gam, 1e-12), 1e-24))]
    best = None
    for scale in (1.0, 0.1, 10.0, 0.01, 100.0, 1e-3, 1e3):
        sol = _root(resid, [start[0] + np.log(scale), start[1]], method="hybr", tol=1e-12)
        if sol.success and max(abs(np.array(resid(sol.x)))) < 1e-8:
            best = sol.x
            break
    if best is None:
        return dict(U2=np.nan, tau2=np.nan, mu2=np.nan, phase="unsolved")
    m, t2 = np.exp(best[0]), np.exp(best[1])
    ph, _, _ = pieces(m, t2)
    U = np.sign(ph) * c * np.sinh(np.minimum(np.abs(ph), 700.0))
    return dict(U2=float(np.sum(WW * U ** 2) / V), tau2=t2, mu2=m, phase="ok")
