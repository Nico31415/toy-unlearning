"""Hard-label cross-entropy with a regulariser kept on (weight decay on the weights = l1 on beta):
   U = argmin sum_i log(1 + exp(-y_i x_i^T beta)) + lam sum_j R(beta_j),  y_i = +-1 drawn with P(y=1) = sigmoid(gam x_i^T beta*)  (gam = inf: y = sign).
Large d (cavity). Sample: logit p = prox_{V l_y}(zeta), zeta = M z + sqrt(Q - M^2) xi, gradient g = l'(p).
Coordinate: beta = prox_{lam R/A}((B beta* + sqrt(C) h)/A),
   A = al E[l''/(1+V l'')],  B = -al E[d/dz of E_y g],  C = al E[g^2],  V = E[prox_R']/A,  Q = E beta^2,  M = E beta beta*."""
import numpy as np
from scipy.special import expit
from scipy.stats import norm
gx, wx = np.polynomial.hermite_e.hermegauss(80); wx = wx / wx.sum()
_x, _w = np.polynomial.legendre.leggauss(80); gz = np.r_[-4.5 + 4.5 * _x, 4.5 + 4.5 * _x]; wz = np.r_[_w, _w] * 4.5 * np.exp(-gz**2 / 2); wz /= wz.sum()

def out_prox(zeta, V, y):
    lo, hi = zeta - V - 1e-9, zeta + V + 1e-9                    # p - V y sigmoid(-y p) = zeta has its root within V of zeta
    for _ in range(70):
        mid = 0.5 * (lo + hi); f = mid - V * y * expit(-y * mid) - zeta
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    return 0.5 * (lo + hi)

def _step(Q, M, V, al, lam, rho, gam, prox, grid):
    Z, XI, W2, Py = grid; amp = rho**-0.5
    zeta = M * Z + np.sqrt(max(Q - M * M, 1e-12)) * XI
    A = C = EzG = 0.0; loss = 0.0
    for y, P in ((1.0, Py), (-1.0, 1 - Py)):
        p = out_prox(zeta, V, y); g = -y * expit(-y * p); l2 = expit(p) * expit(-p)
        A += np.sum(W2 * P * l2 / (1 + V * l2)); C += np.sum(W2 * P * g * g); EzG += np.sum(W2 * P * Z * g); loss += np.sum(W2 * P * np.logaddexp(0, -y * p))
    Bc = -(EzG - M * A); A, Bc, C = al * A, al * Bc, al * C          # E[d_z g] = E[z g] - M E[d_zeta g]  (Gaussian integration by parts)
    q0, _, v0 = prox(0.0, np.sqrt(C) / A, lam / A); q1, m1, v1 = prox(Bc * amp / A, np.sqrt(C) / A, lam / A)
    return (1 - rho) * q0 + rho * q1, rho * amp * m1, ((1 - rho) * v0 + rho * v1) / A, loss

def solve(al, lam, rho, gam, prox, init=(1.0, 0.3, 1.0)):
    from scipy.optimize import least_squares
    Z, XI = np.meshgrid(gz, gx, indexing="ij"); W2 = np.outer(wz, wx); Py = expit(gam * Z) if np.isfinite(gam) else (Z > 0).astype(float); grid = (Z, XI, W2, Py)
    def unpack(v): Q = np.exp(v[0]); return Q, np.tanh(v[1]) * np.sqrt(Q), np.exp(v[2])
    def F(v):
        Q, M, V = unpack(v); Qn, Mn, Vn, _ = _step(Q, M, V, al, lam, rho, gam, prox, grid)
        return [np.log(max(Qn, 1e-30) / Q), (Mn - M) / np.sqrt(Q), np.log(max(Vn, 1e-30) / V)]
    Q0, M0, V0 = init; r = least_squares(F, [np.log(Q0), np.arctanh(np.clip(M0 / np.sqrt(Q0), -0.999, 0.999)), np.log(V0)], xtol=1e-12, ftol=1e-12, gtol=1e-12)
    Q, M, V = unpack(r.x); loss = _step(Q, M, V, al, lam, rho, gam, prox, grid)[3]
    return dict(Q=Q, M=M, V=V, loss=loss, acc=1 - np.arccos(np.clip(M / np.sqrt(Q), -1, 1)) / np.pi, err=float(np.max(np.abs(r.fun))), it=r.nfev)

def prox_l1(a, s, t):
    """moments of soft-threshold(X, t), X ~ N(a, s^2): E b^2, E b, P(b != 0)"""
    u, w = (a - t) / s, (-a - t) / s
    return (((a - t)**2 + s * s) * norm.cdf(u) + (a - t) * s * norm.pdf(u) + ((a + t)**2 + s * s) * norm.cdf(w) - (a + t) * s * norm.pdf(w),
            (a - t) * norm.cdf(u) + s * norm.pdf(u) + (a + t) * norm.cdf(w) - s * norm.pdf(w), norm.cdf(u) + norm.cdf(w))
def prox_l2(a, s, t): return (a * a + s * s) / (1 + t)**2, a / (1 + t), 1 / (1 + t)

def fit(X, y, lam, reg, iters=100000):
    """FISTA for sum log(1+exp(-y x^T b)) + lam R(b)."""
    L = 0.25 * np.linalg.norm(X, 2) ** 2; b = np.zeros(X.shape[0]); yk = b.copy(); tk = 1.0
    for _ in range(iters):
        gr = X @ (-y * expit(-y * (X.T @ yk))); u = yk - gr / L
        bn = np.sign(u) * np.maximum(np.abs(u) - lam / L, 0) if reg == "l1" else u / (1 + lam / L)
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2; yk = bn + (tk - 1) / tn * (bn - b)
        if np.max(np.abs(bn - b)) < 1e-8: b = bn; break
        b, tk = bn, tn
    return b

def sim(a):
    al, lam, gam, reg, seed, d, rho = a; rng = np.random.default_rng(1000 * seed + 7)
    bs = np.zeros(d); bs[:int(rho * d)] = rng.choice([-1, 1], int(rho * d)) * rho**-0.5; X = rng.standard_normal((d, int(al * d))) / np.sqrt(d); z = X.T @ bs
    y = np.where(rng.random(len(z)) < expit(gam * z), 1.0, -1.0) if np.isfinite(gam) else np.sign(z)
    b = fit(X, y, lam, reg); Xt = rng.standard_normal((d, 4000)) / np.sqrt(d)
    return [b @ b / d, b @ bs / d, np.mean(np.logaddexp(0, -y * (X.T @ b))), np.mean(np.sign(Xt.T @ b) == np.sign(Xt.T @ bs))]

def solve_path(al, lam, rho, gam, prox):
    """continuation from a large penalty down to lam (warm starts keep the iteration stable)."""
    init = (1.0, 0.3, 1.0); th = None
    for l in np.geomspace(max(lam, 0.5 * np.sqrt(al)), lam, 8):
        th = solve(al, l, rho, gam, prox, init=init); init = (th["Q"], th["M"], th["V"])
    return th

if __name__ == "__main__":
    import pickle
    from multiprocessing import Pool
    rho, d = 0.2, 1000; rows = []; store = {}
    cfgs = [(al, lam, gam, reg) for reg in ("l1", "l2") for gam in (np.inf, 2.0) for al in (0.3, 1.5) for lam in ((0.05, 0.3) if reg == "l1" else (0.5,))]
    with Pool(2) as p:
        for (al, lam, gam, reg) in cfgs:
            S = np.array(p.map(sim, [(al, lam, gam, reg, s, d, rho) for s in range(16)])); store[(al, lam, gam, reg)] = S; pickle.dump(store, open("se_logistic_sims.pkl", "wb"))
            th = solve_path(al, lam, rho, gam, prox_l1 if reg == "l1" else prox_l2); m, e = S.mean(0), S.std(0, ddof=1) / np.sqrt(16)
            tv = [th["Q"], th["M"], th["loss"], th["acc"]]; rows += [(m[j] - tv[j]) / max(e[j], 1e-12) for j in range(4)]
            print(f"{reg} labels={'sign' if not np.isfinite(gam) else 'sampled'} al={al} lam={lam:<4} | norm {tv[0]:.4f}/{m[0]:.4f}±{e[0]:.4f} | overlap {tv[1]:.4f}/{m[1]:.4f}±{e[1]:.4f} | train loss {tv[2]:.4f}/{m[2]:.4f}±{e[2]:.4f} | test acc {tv[3]:.4f}/{m[3]:.4f}±{e[3]:.4f} | fp {th['err']:.0e}", flush=True)
    rows = np.array(rows); print(f"{len(rows)} numbers: |z|<2 in {100*np.mean(np.abs(rows)<2):.0f}%, max |z| = {np.abs(rows).max():.1f}")
