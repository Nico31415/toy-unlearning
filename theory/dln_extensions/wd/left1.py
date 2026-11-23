"""Non-convex penalty: weight decay on a depth-3 diagonal network, beta = u1 u2 u3 - v1 v2 v3,
(lam_w/2) sum |w_l|^2  ->  lam |beta|^{2/3}, lam = 3 lam_w / 2.  Replica (global-prox channel) vs what algorithms find."""
import numpy as np, pickle
from se_wd import solve, lasso
rho, aPT, aUL, p = 0.2, 0.3, 0.2, 2 / 3

def prox_nc(x, t):
    """global argmin_b 1/2 (b-x)^2 + t |b|^p, 0<p<1, and its derivative."""
    ax = np.abs(x); bmin = (t * p * (1 - p)) ** (1 / (2 - p))
    b = np.maximum(ax, bmin)
    for _ in range(80):                                  # Newton on the outer branch of b + t p b^(p-1) = |x|
        g = b + t * p * b ** (p - 1) - ax; gp = 1 + t * p * (p - 1) * b ** (p - 2)
        b = np.maximum(b - g / np.maximum(gp, 1e-12), bmin)
    ok = (np.abs(b + t * p * b ** (p - 1) - ax) < 1e-9) & (0.5 * (b - ax) ** 2 + t * b ** p < 0.5 * ax ** 2)
    d = np.where(ok, 1 / (1 + t * p * (p - 1) * b ** (p - 2)), 0.0)
    return np.where(ok, np.sign(x) * b, 0.0), d

def data(d, seed):
    rng = np.random.default_rng(seed); bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
    n, f = int(aPT * d), int(aUL * d); X = rng.standard_normal((d, n)) / np.sqrt(d); z = X.T @ bs; t = z.copy(); t[:f] = 0
    return bs, X, z, t, f

def met(b, bs, X, z, t, f):
    r = X.T @ b - t; return [b @ b / (bs @ bs), b @ bs / (bs @ bs), np.mean(r[f:] ** 2) / np.mean(z ** 2)]

def ist(X, t, lam, b0, iters=30000):                     # proximal gradient with the global scalar prox
    L = np.linalg.norm(X, 2) ** 2; b = b0.copy()
    for _ in range(iters):
        bn, _ = prox_nc(b - X @ (X.T @ b - t) / L, lam / L)
        if np.max(np.abs(bn - b)) < 1e-10: b = bn; break
        b = bn
    return b

def deep3(X, t, lam_w, a0, d, steps=300000, lr=0.02):
    U = np.full((3, d), a0); V = np.full((3, d), a0)
    for _ in range(steps):
        pu, pv = U.prod(0), V.prod(0); g = X @ (X.T @ (pu - pv) - t)
        U, V = U - lr * (g * pu / U + lam_w * U), V - lr * (-g * pv / V + lam_w * V)
    return U.prod(0) - V.prod(0)

G = [(aUL / aPT, 1.0, 0.0), (1 - aUL / aPT, 1.0, 1.0)]
obj = lambda b, X, t, lam: 0.5 * np.sum((X.T @ b - t) ** 2) + lam * np.sum(np.abs(b) ** p)
print("lam | replica (global prox): |U|^2, overlap, kept res | prox-gradient from 0 | prox-gradient from lasso | depth-3 network GD (d=300), init 0.5 | init 0.1")
rows = []
for lam in (0.01, 0.03, 0.06, 0.1, 0.15):
    th = solve(aPT, G, lam, rho, prox=prox_nc, iters=8000, damp=0.8)
    A, B, C2, O = [], [], [], []
    for seed in range(3):
        bs, X, z, t, f = data(1500, seed)
        b0 = ist(X, t, lam, np.zeros(1500)); bl = ist(X, t, lam, lasso(X, t, np.ones(len(t)), lam))
        A.append(met(b0, bs, X, z, t, f)); B.append(met(bl, bs, X, z, t, f)); O.append([obj(b0, X, t, lam), obj(bl, X, t, lam)])
    D1, D2 = [], []
    for seed in range(2):
        bs, X, z, t, f = data(300, seed)
        D1.append(met(deep3(X, t, 2 * lam / 3, 0.5, 300), bs, X, z, t, f)); D2.append(met(deep3(X, t, 2 * lam / 3, 0.1, 300), bs, X, z, t, f))
    r = (lam, [th["Q"], th["M"], th["res"][1]], np.mean(A, 0), np.mean(B, 0), np.mean(D1, 0), np.mean(D2, 0), th["err"], np.mean(O, 0))
    rows.append(r)
    print(f"{lam:.2f} | {np.round(r[1],4)} (fp err {th['err']:.0e}) | {np.round(r[2],4)} | {np.round(r[3],4)} | {np.round(r[4],4)} | {np.round(r[5],4)} | objective from 0 / from lasso: {r[7][0]:.3f} / {r[7][1]:.3f}", flush=True)
pickle.dump(rows, open("left1.pkl", "wb"))
