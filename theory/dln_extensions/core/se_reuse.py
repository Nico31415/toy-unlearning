"""Reused-sample unlearning at any k: conditioning on stage 1, then Gordon on the leftover randomness.

SETUP.  A = argmin Q(a) s.t. X^T a = X^T b*   (KKT: h := Q'(A) = X nu,  e := A - b* is orthogonal to col X)
        U = argmin Q(u) - <h,u>  s.t. X_F^T u = 0,   X_F = f of the n columns of X  (phi = f/n).

STEP 1 (conditioning).  {A(X)=a, nu(X)=v} is exactly the affine set {X^T e = 0, X v = h}, so given (A, nu)
        X = P_e^perp  Xt  P_nu^perp + h nu^T/|nu|^2,       Xt a FRESH Gaussian matrix.
    Restricting to the F columns and writing eta = Xt_K nu_K ~ N(0, (1-phi) s_nu^2), s_nu^2 = |nu|^2/d:
        X_F^T u = 0   <=>   Xt_F^T P_e^perp u = t(u) nu_F/|nu_F|,
        t(u) = -(sqrt(phi)/(1-phi)) <h - eta, u>/|nu|.
    A reused constraint is a fresh Gaussian constraint with a right-hand side proportional to one scalar.

STEP 2 (Gordon).  The auxiliary problem is  min Q(u) - <h,u>  s.t.  E[g u] + R(u) <= 0 with
        R^2 = alpha_UL (E[u^2] - s_e^2) + kap^2 S^2/s_nu^2,   S = E[(h-eta)u],  s_e = E[e u]/sqrt(E[e^2]),
        kap = sqrt(phi)/(1-phi).   (fresh: R^2 = alpha_UL E[u^2].)
    Stationarity gives the scalar channel
        Q'(U) + m U = (1-gam) h + gam eta + m s_e e/|e| - theta g
    closed by
        m E[1/(Q''+m)] = alpha_UL            (same dimension count as fresh)
        R = theta E[1/(Q''+m)],   R^2 as above,   gam = theta kap^2 S/(R s_nu^2),   s_e = E[e U]/sqrt(E[e^2]).
    s_nu = mu*tau from stage 1.

KEPT-FORGET LOSS (row 3), from the same representation, for a kept sample i:
        x_i^T (U - A) = rho_v xi_i + nu_i [E[hU]/(1-phi) - E[hA]]/s_nu^2          (reused)
        x_i^T (U - A) = rho_v xi_i + nu_i  E[h(U-A)]/s_nu^2                        (fresh)
    rho_v^2 = E[(U-A)^2] - (E[e(U-A)])^2/E[e^2],  E[nu_i^2] = s_nu^2/alpha_PT.

Lazy limits (checked by hand): ||U||^2 = alpha_PT - alpha_UL, row 3 = alpha_UL (reused);
row 3 = alpha_UL[alpha_PT(1-alpha_UL)+alpha_UL] (fresh).
"""
import numpy as np
from scipy.optimize import root

from se import solve_stage1, prox


def _phi_solve(T, a, iters=60):
    """phi + a sinh(phi) = T, odd and increasing; bisection on [0, |T|]."""
    s, y = np.sign(T), np.abs(T)
    lo, hi = np.zeros_like(y), y.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        big = mid + a * np.sinh(np.minimum(mid, 700.0)) > y
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    return s * 0.5 * (lo + hi)


def _recovered_grid(alpha, k, rho, n_g=81, half=7.0):
    """Stage-1 law when pretraining has recovered b* (rich, alpha > alpha_DT), to leading order in c.

    There tau -> 0 and mu -> inf with s_nu = mu*tau and om = mu*c finite, so A = b* but the MIRROR
    variable h = X nu is not trivial off the support -- it is the dual certificate, and stage 2 needs it:
        off support   h + om sinh(h) = s_nu g,   A = c sinh(h)
        on support    h = L = arcsinh(amp/c),    A = amp + (c/om)(s_nu g - L)
        rho + (1-rho) E[om cosh h/(1 + om cosh h)] = alpha
        rho L^2 + (1-rho) E[h^2] = s_nu^2 (1 - alpha)
    """
    c = np.sqrt(k) / 2
    amp = rho ** -0.5
    L = float(np.arcsinh(amp / c))
    g = np.linspace(-half, half, n_g)
    w = np.exp(-0.5 * g * g)
    w /= w.sum()

    def res(v):
        om, sn = np.exp(v[0]), np.exp(v[1])
        ph = _phi_solve(sn * g, om)
        ch = np.cosh(ph)
        return [rho + (1 - rho) * np.sum(w * om * ch / (1 + om * ch)) - alpha,
                (rho * L * L + (1 - rho) * np.sum(w * ph * ph)) / (sn * sn * (1 - alpha)) - 1.0]

    best = None
    for a0 in (-L, -0.7 * L, -0.4 * L, -1.3 * L):
        for b0 in (np.log(L), np.log(L / 3), np.log(3 * L)):
            sol = root(res, [a0, b0], method="hybr", tol=1e-12)
            if max(abs(np.array(res(sol.x)))) < 1e-8:
                best = sol.x
                break
        if best is not None:
            break
    if best is None:
        return None
    om, sn = np.exp(best[0]), np.exp(best[1])
    ph = _phi_solve(sn * g, om)
    B = np.concatenate([np.zeros(n_g), np.full(n_g, amp)])
    W = np.concatenate([(1 - rho) * w, rho * w])
    A = np.concatenate([c * np.sinh(ph), amp + (c / om) * (sn * g - L)])
    h = np.concatenate([ph, np.full(n_g, L)])
    return dict(B=B, A=A, h=h, e=A - B, W=W, s_nu=sn, c=c, recovered=True, G=np.concatenate([g, g]), om=om)


def stage1_params(alpha, k, rho, n_g=161, half=8.0):
    """(tau, mu) of stage 1 at any k, with no small-c approximation.

    se.solve_stage1 scans tau >= 1e-6 and reports 'recovery' below that, returning tau = 0, mu = inf.
    At finite c the fixed point still exists there (tau ~ c), and near the recovery threshold mu is not
    large, so A = b* is only approximate. Solve the same two equations in the scaled variables
    om = mu*c, s_nu = mu*tau, which stay O(1):
        ph + om sinh(ph) = om b*/c + s_nu g,   A = c sinh(ph)
        E[om cosh ph/(1 + om cosh ph)] = alpha,    E[ph^2] = s_nu^2 (1 - alpha)
    """
    c = np.sqrt(k) / 2
    amp = rho ** -0.5
    s = solve_stage1(alpha, k, rho)
    if s["phase"] == "error":
        return s["tau"], s["mu"]
    if s["phase"] != "recovery":
        return None
    lead = _recovered_grid(alpha, k, rho)
    if lead is None:
        return None
    g = np.linspace(-half, half, n_g)
    w = np.exp(-0.5 * g * g)
    w /= w.sum()
    Bv = np.concatenate([np.zeros(n_g), np.full(n_g, amp)])
    Wv = np.concatenate([(1 - rho) * w, rho * w])
    Gv = np.concatenate([g, g])

    def res(v):
        om, sn = np.exp(v[0]), np.exp(v[1])
        ph = _phi_solve(om * Bv / c + sn * Gv, om)
        ch = np.cosh(ph)
        return [np.sum(Wv * om * ch / (1 + om * ch)) / alpha - 1.0,
                np.sum(Wv * ph * ph) / (sn * sn * (1 - alpha)) - 1.0]

    om0 = float(np.log(lead["om"])); sn0 = float(np.log(lead["s_nu"]))
    for d0 in (0.0, 0.5, -0.5, 1.5, -1.5):
        sol = root(res, [om0 + d0, sn0], method="hybr", tol=1e-12)
        if max(abs(np.array(res(sol.x)))) < 1e-8:
            om, sn = np.exp(sol.x[0]), np.exp(sol.x[1])
            return sn * c / om, om / c
    return None


def stage1_grid(alpha, k, rho, n_g=81, half=7.0):
    """(b*, A, h, e, weights, s_nu) on a grid, in either phase of stage 1."""
    c = np.sqrt(k) / 2
    amp = rho ** -0.5
    p = stage1_params(alpha, k, rho)
    if p is None:
        return None
    tau, mu = p
    g = np.linspace(-half, half, n_g)
    w = np.exp(-0.5 * g * g)
    w /= w.sum()
    B = np.concatenate([np.zeros(n_g), np.full(n_g, amp)])
    W = np.concatenate([(1 - rho) * w, rho * w])
    G = np.concatenate([g, g])
    ph = _phi_solve(mu * (B + tau * G), mu * c)
    A = c * np.sinh(ph)
    return dict(B=B, A=A, h=ph, e=-(ph / mu) + tau * G, W=W, s_nu=mu * tau, c=c, G=G, tau=tau, mu=mu)


def solve_unlearn(alpha, x, k, rho=0.05, reused=True, n_z=81, half=7.0, st=None, start=None, _depth=0,
                  b=0.0, sig=0.0, S=1.0):
    """Returns ||U||^2, kept-forget loss, and the constants. x = alpha_UL.

    General stage 2:  start from the pretrained weights rescaled by sqrt(S) (function S*A, geometry
    c2 = S*c, mirror tilt h unchanged), and fit targets  t_i = b x_i^T b* + sig xi_i  on the forget samples.
      b = 0, sig = 0   forget-to-zero            b = -r   signed / anchored-ascent loss with ratio r
      sig > 0          forget-to-noise           b = 1    true labels (a no-op when S = 1)
    With w = U - b b*, the only changes are  E[u^2] -> E[w^2] + sig^2,  S -> E[(h-eta)w],  s_e -> E[et w],
    and an extra  m b b*  on the right-hand side of the channel.
    """
    st = st if st is not None else stage1_grid(alpha, k, rho)
    if st is None:
        return dict(phase="stage1-not-handled")
    B, A, h, e, W, s_nu, c1 = (st[q] for q in ("B", "A", "h", "e", "W", "s_nu", "c"))
    c = S * c1
    phi = x / alpha
    Ee2 = float(np.sum(W * e * e))
    et = e / np.sqrt(Ee2)
    kap2 = phi / (1 - phi) ** 2 if reused else 0.0
    veta = (1 - phi) * s_nu ** 2
    z = np.linspace(-half, half, n_z)
    wz = np.exp(-0.5 * z * z)
    wz /= wz.sum()
    WW = W[:, None] * wz[None, :]

    def channel(m, th, gam, se):
        sd = np.sqrt(gam * gam * veta + th * th)
        T = ((1 - gam) * h + m * se * et + m * b * B)[:, None] + sd * z[None, :]
        ph = _phi_solve(T, m * c)
        ch = np.cosh(np.minimum(np.abs(ph), 700.0))
        U = c * np.sinh(np.clip(ph, -700, 700))
        Dv = float(np.sum(WW * (c * ch / (1 + m * c * ch))))          # E[1/(Q''+m)]
        return U, Dv

    def resid(v):
        m, th = np.exp(v[0]), np.exp(v[1])
        gam, se = (v[2], v[3]) if reused else (0.0, 0.0)
        U, Dv = channel(m, th, gam, se)
        Ew2 = float(np.sum(WW * (U - b * B[:, None]) ** 2)) + sig * sig
        Sw = float(np.sum(WW * h[:, None] * U)) - gam * veta * Dv - b * EhA
        R = th * Dv
        r = [m * Dv / x - 1.0,
             (R * R - x * (Ew2 - se * se) - kap2 * Sw * Sw / s_nu ** 2) / max(x * Ew2, 1e-300)]
        if reused:
            r += [gam - th * kap2 * Sw / (R * s_nu ** 2),
                  se - float(np.sum(WW * et[:, None] * (U - b * B[:, None])))]
        return r

    # closed-form start at U ~ A
    D0 = float(np.sum(W * np.sqrt((S * A) ** 2 + c * c)))
    EA2, EhA = float(np.sum(W * A * A)), float(np.sum(W * h * A))
    m0 = x / D0
    Sw0 = EhA * (S - b)
    se0 = float(np.sum(W * et * (S * A - b * B)))
    g0 = float(np.clip(kap2 * Sw0 / (D0 * s_nu ** 2), -0.5, 0.5)) if reused else 0.0
    R0 = np.sqrt(max(x * (float(np.sum(W * (S * A - b * B) ** 2)) - (se0 ** 2 if reused else 0.0) + sig * sig)
                     + kap2 * Sw0 ** 2 / s_nu ** 2, 1e-300))
    starts = [start] if start is not None else []
    for sc in (1.0, 0.3, 3.0, 0.1, 10.0, 0.03, 30.0):
        base = [np.log(m0 * sc), np.log(R0 / D0)]
        starts.append(base + ([g0, se0] if reused else []))
        starts.append([np.log(m0 * sc), np.log(R0 / D0 * sc)] + ([g0 / 2, 0.0] if reused else []))
    best = None
    for s0 in starts:
        if s0 is None:
            continue
        try:
            sol = root(resid, s0, method="hybr", tol=1e-11)
        except Exception:
            continue
        if np.all(np.isfinite(sol.x)) and max(abs(np.array(resid(sol.x)))) < 1e-7:
            best = sol.x
            break
    if best is None and _depth < 1:
        # continuation in alpha_UL from an easier budget
        prev, last = None, None
        for xx in np.linspace(0.3 * x, x, 9):
            rr = solve_unlearn(alpha, float(xx), k, rho, reused, n_z, half, st, prev, _depth=1, b=b, sig=sig, S=S)
            prev = rr.get("x0", prev)
            last = rr if rr.get("phase") == "ok" else last
        if rr.get("phase") == "ok":
            return rr
        if last is not None and last["U2"] < 1e-3 and b == 0 and sig == 0:
            # fully collapsed: U = 0 solves the problem and the multipliers are no longer identified
            V0 = float(np.sum(W * B * B))
            rv2 = EA2 - float(np.sum(W * e * A)) ** 2 / Ee2
            return dict(phase="ok", U2=0.0, kept=(rv2 + EhA ** 2 / (alpha * s_nu ** 2)) / V0, m=np.inf,
                        theta=np.nan, gam=np.nan, se=0.0, x0=None, U=np.zeros((len(B), n_z)), z=z, wz=wz,
                        EhU=0.0, pop=1.0, collapsed=True)
    if best is None:
        return dict(phase="unsolved")
    m, th = np.exp(best[0]), np.exp(best[1])
    gam, se = (best[2], best[3]) if reused else (0.0, 0.0)
    U, Dv = channel(m, th, gam, se)
    EU2 = float(np.sum(WW * U * U))
    EhU = float(np.sum(WW * h[:, None] * U))
    V = float(np.sum(W * B * B))
    dUA = U - A[:, None]
    rho_v2 = float(np.sum(WW * dUA ** 2)) - float(np.sum(WW * e[:, None] * dUA)) ** 2 / Ee2
    lin = ((EhU - b * EhA) / (1 - phi) + (b - 1) * EhA) if reused else (EhU - EhA)
    kept = (rho_v2 + lin ** 2 / (alpha * s_nu ** 2)) / V
    return dict(phase="ok", U2=EU2 / V, kept=kept, m=m, theta=th, gam=gam, se=se, x0=list(best),
                U=U, z=z, wz=wz, EhU=EhU, b=b, S=S, EhA=EhA,
                pop=float(np.sum(WW * (U - B[:, None]) ** 2)) / V)


def solve_untrain(alpha, x, k, rho, st, sol, reused, n_e=31, half=6.0):
    # general targets/scale: identity (q) holds with U replaced by w = U - b b*; do-nothing is S*A
    """||U - A(D\\F)||^2 and the do-nothing ||A - A(D\\F)||^2, from the joint law of (b*, A, U, A_K).

    A_K = A(D\\F) has its own stage-1 channel at alpha' = alpha_PT - alpha_UL with noise g'. Jointly,
        g' = r g + sqrt(1-r^2) (q zeta + sqrt(1-q^2) eps),
    g the pretraining noise, zeta the (normalised) stage-2 noise of U, eps independent. Two exact identities
    pin the two correlations. With a = s_nu/sqrt(alpha_PT E[e^2]) (nu_i = -a x_i^T e_{-i}), et = e/|e|:
      (r)  forget samples are fresh for A_K:   E[h e'] = -alpha_UL (s_nu/sqrt(alpha_PT)) E[et e']
      (q)  <Q'(A_K), U> = sum over kept i of nu'_i x_i^T U, with x_i^T U = psi_i + C nu_i:
           E[Q'(A_K) U] = -alpha' a' ( E[e' U] - E[et e'] E[et U] - (s_nu/sqrt(alpha_PT)) C E[et e'] )
           reused: C = E[hU]/((1-phi) s_nu^2).   Fresh arm: zeta is independent of g', q = 0.
    """
    from scipy.optimize import brentq
    B, A, h, e, W, s_nu, c, G = (st[q_] for q_ in ("B", "A", "h", "e", "W", "s_nu", "c", "G"))
    V = float(np.sum(W * B * B))
    U, z, wz = sol["U"], sol["z"], sol["wz"]
    b, S = sol.get("b", 0.0), sol.get("S", 1.0)
    Wm = U - b * B[:, None]
    WW = W[:, None] * wz[None, :]
    EU2 = float(np.sum(WW * U * U))
    ap = alpha - x
    if ap <= 1e-9:
        return dict(dn=S * S * float(np.sum(W * A * A)) / V, un=EU2 / V, r=np.nan, q=np.nan)
    pk = stage1_params(ap, k, rho)
    if pk is None:
        return dict(dn=np.nan, un=np.nan, r=np.nan, q=np.nan)
    taup, mup = pk
    snp = taup * mup
    Ee2 = float(np.sum(W * e * e))
    et = e / np.sqrt(Ee2)
    sa = s_nu / np.sqrt(alpha)
    eps = np.linspace(-half, half, n_e)
    we = np.exp(-0.5 * eps * eps)
    we /= we.sum()

    # ---- r from the pair (A, A_K) alone
    W2 = W[:, None] * we[None, :]

    def AK2(r):
        return prox(B[:, None] + taup * (r * G[:, None] + np.sqrt(1 - r * r) * eps[None, :]), mup, c)

    def gap_r(r):
        ep = AK2(r) - B[:, None]
        return float(np.sum(W2 * h[:, None] * ep)) + x * sa * float(np.sum(W2 * et[:, None] * ep))

    try:
        r = brentq(gap_r, 0.0, 0.999999, xtol=1e-5)
    except Exception:
        return dict(dn=np.nan, un=np.nan, r=np.nan, q=np.nan)
    AKr = AK2(r)
    dn = float(np.sum(W2 * (S * A[:, None] - AKr) ** 2)) / V
    Eep2 = float(np.sum(W2 * (AKr - B[:, None]) ** 2))
    Eete = float(np.sum(W2 * et[:, None] * (AKr - B[:, None])))
    EAK2 = float(np.sum(W2 * AKr ** 2))
    apr = snp / np.sqrt(ap * Eep2)
    EetU = float(np.sum(WW * et[:, None] * Wm))
    phi = x / alpha
    if reused:
        C = (sol["EhU"] - b * sol["EhA"]) / ((1 - phi) * s_nu ** 2)
    W3 = WW[:, :, None] * we[None, None, :]

    def AK3(q):
        gp = r * G[:, None, None] + np.sqrt(1 - r * r) * (q * z[None, :, None] + np.sqrt(1 - q * q) * eps[None, None, :])
        return prox(B[:, None, None] + taup * gp, mup, c)

    def gap_q(q):
        AK = AK3(q)
        lhs = float(np.sum(W3 * np.arcsinh(AK / c) * Wm[:, :, None]))
        EepU = float(np.sum(W3 * (AK - B[:, None, None]) * Wm[:, :, None]))
        return lhs + ap * apr * (EepU - Eete * EetU - sa * C * Eete)

    q = 0.0
    if reused:
        try:
            q = brentq(gap_q, -0.9999, 0.9999, xtol=1e-4)
        except Exception:
            return dict(dn=dn, un=np.nan, r=r, q=np.nan)
    EUAK = float(np.sum(W3 * AK3(q) * U[:, :, None]))
    return dict(dn=dn, un=(EU2 - 2 * EUAK + EAK2) / V, r=r, q=q)
