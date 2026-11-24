"""Identities in the full setting (retain / shared / forget blocks, retain anchor in stage 2), by gradient descent on the weights:
(1) soft-label CE pretraining = Bregman projection; (2) weight decay switched off during stage 2 = rescaling kappa = exp(-2 Lam)."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd", "crit"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, numpy as np
from scipy.special import expit
from ce_tests import project
rho, d = 0.2, 250; dR, dSh = 100, 50
rng = np.random.default_rng(3); bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
def draw(n, lo, hi):
    X = np.zeros((d, n)); X[lo:hi] = rng.standard_normal((hi - lo, n)) / np.sqrt(d); return X
XR, XF = draw(120, 0, dR + dSh), draw(45, dR, d); X = np.hstack([XR, XF]); y = X.T @ bs
Xd, XK = XF[:, :30], XF[:, 30:]
for k in (1.0, 1e-2):
    c = np.sqrt(k) / 2; a0 = np.sqrt(c / 2)
    # (1) cross-entropy on teacher probabilities sigma(x^T b*), gradient descent on (u, v)
    u = np.full(d, a0); v = u.copy(); p = expit(y); lr = 0.02
    for it in range(600000):
        r = expit(X.T @ (u * u - v * v)) - p
        if it % 500 == 0 and np.max(np.abs(X.T @ (u * u - v * v) - y)) < 1e-6: break
        g = X @ r; u, v = u - lr * 2 * u * g, v + lr * 2 * v * g
    A = u * u - v * v; P = project(np.zeros(d), c, A_eq=X, b_eq=y)[0]
    print(f"k={k:g}: CE pretraining with blocks, |GD - projection|/|GD| = {np.linalg.norm(A-P)/np.linalg.norm(A):.1e}  ({it} steps)", flush=True)
    # (2) stage 2: deleted -> 0, retain samples anchored on labels, squared loss, weight decay on for a while
    M = np.hstack([Xd, XR]); t = np.r_[np.zeros(30), XR.T @ bs]; s = np.sqrt(P * P + c * c); u1, v1 = np.sqrt((s + P) / 2), np.sqrt((s - P) / 2)
    for Lam in (0.25, 1.0):
        u, v = u1.copy(), v1.copy(); tt, lr = 0.0, 0.005
        while tt < 3000:
            r = M.T @ (u * u - v * v) - t; lam = 0.5 if tt < Lam / 0.5 else 0.0
            if lam == 0 and np.max(np.abs(r)) < 1e-6: break
            g = M @ r; u, v = u - lr * (2 * u * g + lam * u), v - lr * (-2 * v * g + lam * v); tt += lr
        U = u * u - v * v; kap = np.exp(-2 * Lam); Pk = project(kap * P, kap * c, A_eq=M, b_eq=t)[0]; P1 = project(P, c, A_eq=M, b_eq=t)[0]
        print(f"   stage 2 with retain anchor, Lam={Lam}: |GD - projection from kappa A|/|GD| = {np.linalg.norm(U-Pk)/np.linalg.norm(U):.1e}  (from A: {np.linalg.norm(U-P1)/np.linalg.norm(U):.1e})", flush=True)
