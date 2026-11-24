"""Weight decay that is switched off (Jacobs, Zhou & Burkholz 2025, Thm 3.11) in our model.
Flow: u' = -2u g - lam(t) u, v' = +2v g - lam(t) v, g = grad_beta of 1/2|X^T beta - t|^2, beta = u^2 - v^2.
Claim: after switch-off and convergence, the end point is the Bregman projection from kappa*beta_start with geometry kappa*c,
kappa = exp(-2 int lam dt): the same as multiplying the weights by sqrt(kappa) before training, whenever the decay happened."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, numpy as np
from ce_tests import project
rng = np.random.default_rng(0); d, rho = 150, 0.2
bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
X = rng.standard_normal((d, 45)) / np.sqrt(d); y = X.T @ bs

def gd(u, v, Xs, ts, lam_fn, T_total, lr=0.01):
    t = 0.0
    while t < T_total:
        g = Xs @ (Xs.T @ (u*u - v*v) - ts); lam = lam_fn(t)
        u, v = u - lr * (2*u*g + lam*u), v - lr * (-2*v*g + lam*v); t += lr
    return u, v

def cgeom(u, v): return 2 * np.mean(u * v)          # beta = c sinh(theta) with c = 2uv (uniform here)

print("stage 1 (from the balanced start u = v): weight decay lam for time T, then off, run to convergence")
for k0, lam, T in ((1.0, 0.5, 1.0), (1.0, 0.2, 5.0), (1e-2, 1.0, 0.5)):
    a0 = np.sqrt(np.sqrt(k0) / 4); u, v = gd(np.full(d, a0), np.full(d, a0), X, y, lambda t: lam if t < T else 0.0, 400.0)
    b = u*u - v*v; kap = np.exp(-2 * lam * T); c = 2 * a0 * a0
    P = project(np.zeros(d), kap * c, A_eq=X, b_eq=y)[0]; P0 = project(np.zeros(d), c, A_eq=X, b_eq=y)[0]
    print(f"  k={k0:g} lam={lam} T={T}: kappa={kap:.3f} | measured 2uv/c = {cgeom(u,v)/c:.3f} | |GD - proj(kappa c)|/|GD| = {np.linalg.norm(b-P)/np.linalg.norm(b):.1e} | vs proj(c): {np.linalg.norm(b-P0)/np.linalg.norm(b):.1e}")

print("stage 2 (from the pretrained model, deleted samples -> 0): weight decay during part of stage 2, then off")
a0 = np.sqrt(np.sqrt(1e-2) / 4); c = 2 * a0 * a0
u1, v1 = gd(np.full(d, a0), np.full(d, a0), X, y, lambda t: 0.0, 400.0); A = u1*u1 - v1*v1
Xf, tf = X[:, :30], np.zeros(30)
for lam, t_on, t_off in ((0.5, 0.0, 1.0), (0.5, 3.0, 4.0), (0.2, 0.0, 4.0)):
    u, v = gd(u1.copy(), v1.copy(), Xf, tf, lambda t: lam if t_on <= t < t_off else 0.0, 400.0)
    U = u*u - v*v; kap = np.exp(-2 * lam * (t_off - t_on))
    P = project(kap * A, kap * c, A_eq=Xf, b_eq=tf)[0]; P1 = project(A, c, A_eq=Xf, b_eq=tf)[0]
    print(f"  lam={lam} on [{t_on},{t_off}]: kappa={kap:.3f} | |GD - proj from kappa*A|/|GD| = {np.linalg.norm(U-P)/np.linalg.norm(U):.1e} | vs proj from A: {np.linalg.norm(U-P1)/np.linalg.norm(U):.1e} | |U|^2/|b*|^2 = {U@U/(bs@bs):.4f} (no decay: {P1@P1/(bs@bs):.4f})")
