"""Per-sample closure test: Lam_i = G(t; b_i, a_i) with dLam/dt' = 4 w(tau - a_i Lam - b_i r(t')), self-consistent b_i."""
import numpy as np
from scipy.special import expit, log_expit
from soft import data, TAU
from ce_tests import project, flow, rel, _LXX

def run(kind, k, bnpo=1.0, times=(0.5, 1, 1.5, 2), seed=0, apt=0.3, blocks=(0, 0, 200)):
    c = np.sqrt(k) / 2
    d, bs, XR, yR, XF, yF, XK, yK = data(seed, apt, blocks)
    A, v = project(np.zeros(d), c, A_eq=XF, b_eq=TAU * yF)
    th0 = np.arcsinh(A / c); Z = XF * yF; g = Z.sum(1); lpref = log_expit(TAU); n = Z.shape[1]
    wfun = (lambda m: expit(-m)) if kind == "GA" else (lambda m: 2 * expit(bnpo * (log_expit(m) - lpref)) * expit(-m))
    def gc(beta):
        m = Z.T @ beta
        return Z @ wfun(m), XF, (expit(m) * expit(-m) * (1 if kind == "GA" else 2 * (1 + bnpo))) + 1e-9
    def closure(t, ramp, iters=300, K=200):
        Lam = np.full(n, 4 * wfun(TAU) * t)
        ts = np.linspace(0, t, K + 1); dt = t / K
        for it in range(iters):
            beta = c * np.sinh(th0 - Z @ Lam)
            m = Z.T @ beta
            a = (Z ** 2).T @ np.sqrt(beta ** 2 + c ** 2)
            b = TAU - m - a * Lam
            L = np.zeros(n)
            for j in range(K):                      # scalar ODEs, midpoint in the ramp
                r = ((ts[j] + dt / 2) / t) if ramp else 1.0
                # a ramps too (a0 -> a): a(t') = a0 + (a - a0) r
                L = L + dt * 4 * wfun(TAU - a * L - b * r)
            new = 0.7 * Lam + 0.3 * L
            if np.max(np.abs(new - Lam)) < 1e-9: break
            Lam = new
        return c * np.sinh(th0 - Z @ Lam)
    todo = list(times)
    print(f"\n{kind} k={k} bnpo={bnpo}:  t | mean m | err uniform(s=mean Lam) | err closure(const b) | err closure(ramp b)")
    def rec(beta, t):
        if todo and t >= todo[0]:
            todo.pop(0)
            m = Z.T @ beta
            Lam = -np.linalg.lstsq(Z, np.arcsinh(beta / c) - th0, rcond=None)[0]
            U2 = c * np.sinh(th0 - Lam.mean() * g)
            print(f"{t:5.2f} | {m.mean():6.2f} | {rel(beta,U2,A):.3f} | {rel(beta,closure(t,False),A):.3f} | {rel(beta,closure(t,True),A):.3f}")
    _LXX.clear()
    flow(A.copy(), c, gc, lambda b, t: not todo, rec, maxsteps=300_000)

for kind, bn, T in (("GA", 1, (0.5, 1, 1.5, 2)), ("NPO", 1.0, (0.5, 1, 2, 4)), ("NPO", 4.0, (1, 2, 4, 8, 16))):
    for k in (1e-6, 1.0):
        run(kind, k, bnpo=bn, times=T)
