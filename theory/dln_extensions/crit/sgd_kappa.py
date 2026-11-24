"""How uniform is SGD's shrink factor across coordinates, and does a single kappa (as in the replica curves) describe SGD's stage 2?
Loss (1/2n) sum r^2; step u <- u(1 - 2 gam g), v <- v(1 + 2 gam g), g = mini-batch gradient in beta. kappa_j = prod_k (1 - 4 gam^2 g_kj^2)."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd", "crit"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, numpy as np, zlib
from ce_tests import project
rho, d, aPT, aUL, k = 0.2, 400, 0.3, 0.2, 1e-2
c = np.sqrt(k) / 2
print("seed gam b | steps | kappa: geometric mean, min, max, coeff. of variation, corr(log kappa, |A|) | rel. distance of the SGD end point to: flow from A | per-coordinate kappa | one kappa (geom. mean)")
for seed in (0, 1):
    rng = np.random.default_rng(zlib.crc32(repr(("SG", seed)).encode())); bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
    n, f = int(aPT * d), int(aUL * d); X = rng.standard_normal((d, n)) / np.sqrt(d); Xf = X[:, :f]
    A = project(np.zeros(d), c, A_eq=X, b_eq=X.T @ bs)[0]; s = np.sqrt(A * A + c * c); u0, v0 = np.sqrt((s + A) / 2), np.sqrt((s - A) / 2)
    U_gf = project(A, c, A_eq=Xf, b_eq=np.zeros(f))[0]
    for gam, b in ((2.0, 1), (0.7, 1), (2.0, 8), (2.0, f)):
        u, v = u0.copy(), v0.copy(); logk = np.zeros(d); r2 = np.random.default_rng(seed + 10)
        for it in range(3_000_000):
            beta = u * u - v * v
            if it % 200 == 0 and np.max(np.abs(Xf.T @ beta)) < 1e-6: break
            idx = r2.choice(f, b, replace=False) if b < f else slice(None)
            g = Xf[:, idx] @ (Xf[:, idx].T @ beta) / b
            logk += np.log1p(-4 * gam * gam * g * g); u, v = u * (1 - 2 * gam * g), v * (1 + 2 * gam * g)
        U = u * u - v * v; kap = np.exp(logk); kbar = np.exp(logk.mean())
        Pj = project(kap * A, kap * c, A_eq=Xf, b_eq=np.zeros(f))[0]; P1 = project(kbar * A, kbar * c, A_eq=Xf, b_eq=np.zeros(f))[0]
        nu = np.linalg.norm(U); rd = lambda P: np.linalg.norm(U - P) / nu
        print(f"{seed} {gam} {b:3d} | {it:7d} | {kbar:.3f} {kap.min():.3f} {kap.max():.3f} {kap.std()/kap.mean():.3f} {np.corrcoef(logk, np.abs(A))[0,1]:+.2f} | {rd(U_gf):.4f} | {rd(Pj):.4f} | {rd(P1):.4f}", flush=True)
