import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import numpy as np, time, sys
from scipy.special import expit, log_expit
from scipy.integrate import solve_ivp
from soft import data, TAU, Q
from ce_tests import project, flow, rel, cos, _LXX

def run(kind, k, blocks=(0, 0, 200), apt=0.3, bnpo=1.0, times=(0.25, 0.5, 1, 1.5, 2, 3, 5, 8), seed=0):
    c = np.sqrt(k) / 2
    d, bs, XR, yR, XF, yF, XK, yK = data(seed, apt, blocks)
    A, v = project(np.zeros(d), c, A_eq=XF, b_eq=TAU * yF)
    th0 = np.arcsinh(A / c); Z = XF * yF; g = Z.sum(1); lpref = log_expit(TAU)
    def wfun(m):
        if kind == "GA": return expit(-m)
        return 2 * expit(bnpo * (log_expit(m) - lpref)) * expit(-m)
    def gc(beta):
        m = Z.T @ beta
        return Z @ wfun(m), XF, (expit(m) * expit(-m) * (1 if kind == "GA" else 2 * (1 + bnpo))) + 1e-9
    # uniform theory: ds/dt = 4 * mean w(m_i(s)),  beta(s) = c sinh(th0 - s g)
    sol = solve_ivp(lambda t, s: [4 * np.mean(wfun(Z.T @ (c * np.sinh(np.clip(th0 - s[0] * g, -50, 50)))))], (0, max(times)), [0.0],
                    dense_output=True, rtol=1e-8, atol=1e-10)
    todo = list(times)
    print(f"\n{kind} k={k} bnpo={bnpo}:  t | mean m, sd m | mean Lam, sd Lam, corr(Lam,m) | err uniform(ODE s) | err uniform(s=mean Lam) | err proj(m<=mean m) | acc")
    def rec(beta, t):
        if todo and t >= todo[0]:
            todo.pop(0)
            m = Z.T @ beta
            Lam = -np.linalg.lstsq(Z, np.arcsinh(beta / c) - th0, rcond=None)[0]
            s = sol.sol(t)[0]
            U1 = c * np.sinh(th0 - s * g); U2 = c * np.sinh(th0 - Lam.mean() * g)
            P, _ = project(A, c, A_ge=-Z, b_ge=-np.full(len(m), m.mean()))
            print(f"{t:5.2f} | {m.mean():7.2f} {m.std():6.2f} | {Lam.mean():6.3f} {Lam.std():6.3f} {np.corrcoef(Lam, m)[0,1]:6.2f} | {rel(beta,U1,A):.3f} | {rel(beta,U2,A):.3f} | {rel(beta,P,A):.3f} | {np.mean(m>0):.2f}")
    _LXX.clear()
    flow(A.copy(), c, gc, lambda b, t: not todo, rec, maxsteps=300_000)

for kind, bn in (("GA", 1), ("NPO", 1.0), ("NPO", 4.0)):
    for k in (1e-6, 1.0):
        run(kind, k, bnpo=bn, times=(0.25, 0.5, 1, 1.5, 2, 2.5) if kind == "GA" else (0.25, 0.5, 1, 2, 4, 8, 16, 32))
