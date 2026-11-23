"""State evolution for soft-label CE pretraining of a diagonal network = min Q_k(beta) s.t. x_i^T beta = tau*sign(x_i^T beta*).
Scalar channel: beta = prox_{Q/mu}(nu*b + sig*h), nu = tau*sqrt(2/pi), h~N(0,1), b ~ teacher entry.
Equations: E[prox'] = alpha ;  sig^2 = (tau^2 - 2 tau sqrt(2/pi) kappa + E[beta^2]) / alpha, kappa = E[beta b]."""
import numpy as np
from scipy.optimize import least_squares

H = np.linspace(-9, 9, 4001); WH = np.exp(-H ** 2 / 2); WH /= WH.sum()
M1 = np.sqrt(2 / np.pi)

def prox(x, mu, c, it=60):
    """solve arcsinh(b/c) + mu (b - x) = 0"""
    b = np.sign(x) * np.maximum(np.abs(x) - np.arcsinh(np.abs(x) / c) / mu, 0) + 0.0   # start
    b = np.where(np.abs(b) < 1e-300, x * mu * c / (1 + mu * c), b)
    for _ in range(it):
        f = np.arcsinh(b / c) + mu * (b - x)
        b = b - f / (1 / np.sqrt(b * b + c * c) + mu)
    return b

def moments(mu, sig, tau, k, rho):
    c = np.sqrt(k) / 2; amp = rho ** -0.5; nu = tau * M1
    out = {}
    b0 = prox(sig * H, mu, c); b1 = prox(nu * amp + sig * H, mu, c)        # teacher 0 and +amp (the -amp case is symmetric)
    d0 = mu / (mu + 1 / np.sqrt(b0 ** 2 + c ** 2)); d1 = mu / (mu + 1 / np.sqrt(b1 ** 2 + c ** 2))
    E = lambda f0, f1: (1 - rho) * WH @ f0 + rho * WH @ f1
    return dict(dp=E(d0, d1), q=E(b0 ** 2, b1 ** 2), kappa=rho * amp * (WH @ b1))

def solve(alpha, tau, k, rho, x0=None):
    def F(v):
        mu, sig = np.exp(v)
        m = moments(mu, sig, tau, k, rho)
        return [m["dp"] - alpha, np.log(sig ** 2 * alpha / (tau ** 2 - 2 * tau * M1 * m["kappa"] + m["q"]))]
    best = None
    for s in ([x0] if x0 is not None else []) + [np.array([a, b]) for a in (0., 2., -2., 4.) for b in (0., 1., -1.)]:
        r = least_squares(F, s, xtol=1e-13, ftol=1e-13)
        if best is None or r.cost < best.cost: best = r
        if best.cost < 1e-18: break
    mu, sig = np.exp(best.x)
    m = moments(mu, sig, tau, k, rho)
    m.update(mu=mu, sig=sig, resid=best.cost, x=best.x, cos=m["kappa"] / np.sqrt(m["q"]))
    m["acc"] = 1 - np.arccos(np.clip(m["cos"], -1, 1)) / np.pi          # test accuracy on fresh inputs
    return m

if __name__ == "__main__":
    import sys, time, zlib
    sys.path.insert(0, ".")
    from ce_tests import project
    tau, rho, d, NS = np.log(9.0), 0.2, 600, 4
    print("alpha |  k   | theory: kappa   q    acc  | sim: kappa        q            acc        | resid")
    for k in (1e-6, 1e-2, 1.0):
        c = np.sqrt(k) / 2; x0 = None
        for al in (0.1, 0.3, 0.5, 0.7, 0.9):
            m = solve(al, tau, k, rho, x0); x0 = m["x"]
            S = []
            for s in range(NS):
                rng = np.random.default_rng(zlib.crc32(repr((k, al, s)).encode()))
                bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho ** -0.5, 0.0)
                X = rng.standard_normal((d, int(al * d))) / np.sqrt(d)
                y = np.sign(X.T @ bs)
                A, v = project(np.zeros(d), c, A_eq=X, b_eq=tau * y)
                Xt = rng.standard_normal((d, 4000)) / np.sqrt(d)
                S.append([A @ bs / d, A @ A / d, np.mean(np.sign(Xt.T @ A) == np.sign(Xt.T @ bs))])
            S = np.array(S); mn, se = S.mean(0), S.std(0) / np.sqrt(NS)
            print(f"{al:4.1f}  | {k:5.0e} | {m['kappa']:6.3f} {m['q']:7.3f} {m['acc']:.3f} | {mn[0]:6.3f}±{se[0]:.3f} {mn[1]:7.3f}±{se[1]:.3f} {mn[2]:.3f}±{se[2]:.3f} | {m['resid']:.0e}")
