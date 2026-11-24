"""Deterministic identity: gradient descent with one-sided decay, then off, reaches the tilted Bregman projection."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd", "crit"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, numpy as np
from ce_tests import project
rho, al, d = 0.2, 0.3, 400
rng = np.random.default_rng(5); bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
X = rng.standard_normal((d, int(al * d))) / np.sqrt(d); y = X.T @ bs
for k in (1.0, 1e-2):
    c = np.sqrt(k) / 2
    for Lam in (0.5, 1.5):
        for lr in (0.01, 0.0025):
            a0 = np.sqrt(c / 2); u = np.full(d, a0); v = u.copy(); t = 0.0; Ton = Lam / 0.5
            while t < 1500:
                r = X.T @ (u * u - v * v) - y; lam = 0.5 if t < Ton else 0.0
                if t >= Ton and np.max(np.abs(r)) < 1e-7: break
                g = X @ r; u, v = u - lr * (2 * u * g + lam * u), v - lr * (-2 * v * g); t += lr
            b = u * u - v * v; cp = c * np.exp(-Lam); P = project(np.full(d, cp * np.sinh(-Lam)), cp, A_eq=X, b_eq=y)[0]; P0 = project(np.zeros(d), c, A_eq=X, b_eq=y)[0]
            print(f"k={k:g} Lam={Lam} lr={lr}: |GD - tilted projection|/|GD| = {np.linalg.norm(b-P)/np.linalg.norm(b):.1e}   (vs plain projection: {np.linalg.norm(b-P0)/np.linalg.norm(b):.1e})", flush=True)
