"""Teacher-probability toy: A(D) = min Q s.t. X^T b = gamma X^T b*. Ascent on hard labels y=sign(x^T b*).
Margins start at gamma|z_i|. Test frozen-weight family b(s) = c sinh(th0 - s g_w), g_w = sum w(m_i^0) y_i x_i."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import numpy as np, zlib, sys
from scipy.special import expit, log_expit
from scipy.optimize import minimize_scalar
from ce_tests import project, flow, rel, _LXX
RHO = 0.2

def setup(seed, apt, d, k, gamma):
    rng = np.random.default_rng(zlib.crc32(repr(("H", seed, apt, d)).encode()))
    bs = np.where(rng.random(d) < RHO, rng.choice([-1, 1], d) * RHO ** -0.5, 0.0)
    X = rng.standard_normal((d, int(round(apt * d)))) / np.sqrt(d)
    z = X.T @ bs; y = np.sign(z); c = np.sqrt(k) / 2
    A, v = project(np.zeros(d), c, A_eq=X, b_eq=gamma * z)
    return bs, X, z, y, c, A

def wf(kind, bnpo, m, m0):
    if kind == "GA": return expit(-m)
    return 2 * expit(bnpo * (log_expit(m) - log_expit(m0))) * expit(-m)

def run(kind, k, apt=0.3, gamma=2.0, bnpo=1.0, d=200, seed=0, accs=(0.95, 0.9, 0.8, 0.7, 0.6, 0.5)):
    bs, X, z, y, c, A = setup(seed, apt, d, k, gamma)
    Z = X * y; m0 = Z.T @ A; th0 = np.arcsinh(A / c)
    w0 = wf(kind, bnpo, m0, m0); gw = Z @ w0
    def gc(beta):
        m = Z.T @ beta
        return Z @ wf(kind, bnpo, m, m0), X, (expit(m) * expit(-m) * (1 if kind == "GA" else 2 * (1 + bnpo))) + 1e-9
    todo = list(accs)
    print(f"\n{kind} bnpo={bnpo} k={k} apt={apt}:  t | forget-train acc | mean m | cos(b,b*) | best-s err frozen-weight | corr(Lam, w0)")
    def rec(beta, t):
        m = Z.T @ beta
        if todo and np.mean(m > 0) <= todo[0]:
            todo.pop(0)
            Lam = -np.linalg.lstsq(Z, np.arcsinh(beta / c) - th0, rcond=None)[0]
            f = lambda s: np.linalg.norm(c * np.sinh(th0 - s * gw) - beta)
            r = minimize_scalar(f, bounds=(0, 50 * t + 1), method="bounded", options=dict(xatol=1e-8))
            print(f"{t:6.2f} | {np.mean(m>0):.2f} | {m.mean():6.2f} | {beta@bs/np.linalg.norm(beta)/np.linalg.norm(bs):.3f} | {r.fun/np.linalg.norm(beta-A):.3f} | {np.corrcoef(Lam,w0)[0,1]:.2f}")
    _LXX.clear()
    flow(A.copy(), c, gc, lambda b, t: not todo, rec, maxsteps=200_000)

for kind, bn in (("GA", 1), ("NPO", 1.0), ("NPO", 4.0)):
    for k in (1e-6, 1.0):
        run(kind, k, bnpo=bn)
