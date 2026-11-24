"""Numerical sanity checks of two lemmas used in the proofs (not proofs).
(a) dual contraction: ||lam - lam_M||(t) <= 4 ||X|| int_0^t ||(w - w_M)(theta_M(s))|| ds   (squared loss, w = c sinh, w_M = c sinh(clip_M), monotone)
(b) SGD reparametrisation: u <- u(1-2 gam G), v <- v(1+2 gam G)  <=>  theta <- theta - 2 artanh(2 gam G), s <- s + log(1 - 4 gam^2 G^2), beta = c e^s sinh(theta)."""
import numpy as np
rng = np.random.default_rng(0); d, n, c, dt, T = 300, 100, 0.05, 2e-4, 3.0
bs = np.zeros(d); bs[:60] = rng.choice([-1, 1], 60) * 5**0.5; X = rng.standard_normal((d, n)) / np.sqrt(d); y = X.T @ bs; nx = np.linalg.norm(X, 2)
for M in (2.0, 3.0, 4.0):
    wM = lambda th: c * np.sinh(np.clip(th, -M, M)); lam = np.zeros(n); lamM = np.zeros(n); bound = 0.0; worst = -np.inf
    for k in range(int(T / dt)):
        th, thM = -X @ lam, -X @ lamM; bound += dt * 4 * nx * np.linalg.norm(c * np.sinh(thM) - wM(thM))
        lam = lam + 4 * dt * (X.T @ (c * np.sinh(th)) - y); lamM = lamM + 4 * dt * (X.T @ wM(thM) - y); worst = max(worst, np.linalg.norm(lam - lamM) - bound)
    print(f"(a) M={M}: max|theta| true {np.abs(X @ lam).max():.2f}; ||lam-lam_M|| = {np.linalg.norm(lam - lamM):.4f} <= bound {bound:.4f}; max violation {worst:.1e}")
u = np.full(d, np.sqrt(c / 2)); v = u.copy(); th = np.zeros(d); s = np.zeros(d); gam = 0.03
for k in range(200):
    b = u * u - v * v; G = X @ ((rng.random(n) < 0.2) / 0.2 * (X.T @ b - y)); u, v = u * (1 - 2 * gam * G), v * (1 + 2 * gam * G)
    th = th - 2 * np.arctanh(2 * gam * G); s = s + np.log(1 - 4 * gam**2 * G**2)
print(f"(b) max |beta_uv - c e^s sinh(theta)| after 200 SGD steps: {np.abs(u*u - v*v - c*np.exp(s)*np.sinh(th)).max():.1e}  (max |2 gam G| last step {np.abs(2*gam*G).max():.2f})")
