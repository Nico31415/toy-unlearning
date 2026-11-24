import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import numpy as np
rng = np.random.default_rng(0)
n, d, al = 40, 100, 0.05
X = rng.standard_normal((n, d)); bs = np.zeros(d); bs[:5] = [1.5, -1.0, 1.0, -0.8, 0.7]; y = X @ bs
D = (X ** 2).mean(0)

def train(wp, wm, Xs, ys, gam, b, steps, rng, tol=1e-13):
    """minibatch SGD without replacement inside a step. Returns weights, sum_k ln(1 - gam^2 u_k^2), sum of full loss (Riemann)."""
    ns = len(ys); logk = np.zeros(d); intL = 0.0
    for k in range(steps):
        r = Xs @ (wp ** 2 - wm ** 2) - ys
        L = (r @ r) / (4 * ns); intL += gam * L
        if L < tol: break
        idx = rng.choice(ns, b, replace=False) if b < ns else np.arange(ns)
        u = (r[idx] @ Xs[idx]) / b
        logk += np.log1p(-(gam * u) ** 2)
        wp, wm = wp * (1 - gam * u), wm * (1 + gam * u)
    return wp, wm, logk, intL, k

print("A. batch-size dependence, same step size gamma = 0.02 (pretraining from alpha = 0.05)")
print(" b  | steps | median ln(alpha/alpha_inf): exact product | formula 2*gamma*D*c_b*intL (+GD term) | c_b=(n-b)/(b(n-1)) | test err | GD-from-alpha_inf test err")
gam = 0.02
for b in (1, 2, 5, 10, 20, 40):
    wp, wm, logk, intL, k = train(np.full(d, al), np.full(d, al), X, y, gam, b, 400000, np.random.default_rng(1))
    cb = (n - b) / (b * (n - 1)); ex = -0.5 * logk      # ln(alpha/alpha_inf) per coordinate
    beta = wp ** 2 - wm ** 2
    ainf = al * np.exp(-ex)
    w2p, w2m, *_ = train(ainf.copy(), ainf.copy(), X, y, 0.0005, n, 2000000, rng)   # near gradient flow from alpha_inf
    b2 = w2p ** 2 - w2m ** 2
    print(f"{b:3d} | {k:6d} | {np.median(ex):.4f} | {np.median(2 * gam * D * cb * intL):.4f} | {cb:.3f} | {np.sum((beta-bs)**2):.4f} | {np.sum((b2-bs)**2):.4f}  (|SGD - GF(alpha_inf)| = {np.linalg.norm(beta-b2):.4f})")

print("\nB. stage 2 with SGD = gradient flow from the rescaled start kappa*A, kappa_j = prod_k (1 - gamma^2 u_kj^2)")
wp, wm, *_ = train(np.full(d, al), np.full(d, al), X, y, 0.0005, n, 2000000, rng)            # pretrain ~ gradient flow
A = wp ** 2 - wm ** 2
f = 15; Xf, yf = X[:f], np.zeros(f)                                                            # deleted samples, zero targets
for gam2, b in ((0.02, 1), (0.02, 5), (0.005, 1)):
    sp, sm, logk, intL, k = train(wp.copy(), wm.copy(), Xf, yf, gam2, b, 400000, np.random.default_rng(2))
    U = sp ** 2 - sm ** 2; kap = np.exp(logk)
    gp, gm, *_ = train(wp * np.sqrt(kap), wm * np.sqrt(kap), Xf, yf, 0.0005, f, 2000000, rng)  # GF from sqrt(kappa)-rescaled weights
    Uk = gp ** 2 - gm ** 2
    g1p, g1m, *_ = train(wp.copy(), wm.copy(), Xf, yf, 0.0005, f, 2000000, rng)               # GF from A (kappa = 1)
    U1 = g1p ** 2 - g1m ** 2
    print(f"gamma={gam2} b={b}: median kappa = {np.median(kap):.3f} (min {kap.min():.3f}) | |U_SGD - U_GF(kappa A)|/|U_SGD| = {np.linalg.norm(U-Uk)/np.linalg.norm(U):.4f} | |U_SGD - U_GF(A)|/|U_SGD| = {np.linalg.norm(U-U1)/np.linalg.norm(U):.4f} | |U|^2: SGD {U@U:.4f}, GF(A) {U1@U1:.4f}")
