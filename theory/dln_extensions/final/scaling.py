"""Limits of the stage-2 trajectory (squared loss, forget coordinates only, zero targets on the forget samples, converged pretraining). Simulation, d = 1000.
Lazy limit (c large): beta ~ c theta, the flow is linear and depends on time only through c t.
Rich limit (c small): for a stage 2 much shorter than log(1/c), weights that pretraining left at zero stay there and the others move multiplicatively:
    beta_j(t) = beta_j(T1) exp(sign(beta_j(T1)) (theta_j(t) - theta_j(T1))),  i.e.  d beta_j / dt = -4 |beta_j| grad_j   (no dependence on c)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ce"))
from ce_tests import project
rho, d, aPT, aUL = 0.2, 1000, 0.3, 0.2
def one(k, seed, taus, unit):
    """unit = 'ct': times are tau / c; unit = 't': times are tau."""
    rng = np.random.default_rng(seed); c = np.sqrt(k) / 2; bs = np.zeros(d); bs[:int(rho * d)] = rng.choice([-1, 1], int(rho * d)) * rho**-0.5
    X = rng.standard_normal((d, int(aPT * d))) / np.sqrt(d); f = int(aUL * d); XF, XK = X[:, :f], X[:, f:]; A = project(np.zeros(d), c, A_eq=X, b_eq=X.T @ bs)[0]; AK = project(np.zeros(d), c, A_eq=XK, b_eq=XK.T @ bs)[0]
    sc = 1 / c if unit == "ct" else 1.0; dt = 0.004 * sc if unit == "ct" else 0.004; th = np.arcsinh(A / c); out = []; lim = []; A0 = A.copy(); sg = np.sign(A0); dth = np.zeros(d); t = 0.0
    for tau in taus:
        while t < tau * sc - 1e-12:
            th -= 4 * dt * (XF @ (XF.T @ (c * np.sinh(th)))); dth -= 4 * dt * (XF @ (XF.T @ (A0 * np.exp(sg * dth)))); t += dt
        b = c * np.sinh(th); bl = A0 * np.exp(sg * dth); out.append([np.sum((b - AK)**2) / d, b @ b / d]); lim.append([np.sum((bl - AK)**2) / d, bl @ bl / d])
    return np.array(out), np.array(lim)
if __name__ == "__main__":
    ns = 4; taus = [0.1, 0.25, 0.5, 1.0, 2.0]
    print("LAZY: distance to retrained / norm at times tau/c  (should not depend on k for large k)")
    for k in (1.0, 1e2, 1e4, 1e6):
        R = np.mean([one(k, s, taus, "ct")[0] for s in range(ns)], 0); print(f"  k={k:<8g} dist " + " ".join(f"{x:.4f}" for x in R[:, 0]) + " | norm " + " ".join(f"{x:.4f}" for x in R[:, 1]), flush=True)
    print("RICH: distance to retrained / norm at times tau  (should converge to the multiplicative flow as k -> 0)")
    for k in (1e-2, 1e-4, 1e-6, 1e-8):
        R = [one(k, s, taus, "t") for s in range(ns)]; a = np.mean([r[0] for r in R], 0); l = np.mean([r[1] for r in R], 0)
        print(f"  k={k:<8g} dist " + " ".join(f"{x:.4f}" for x in a[:, 0]) + " | norm " + " ".join(f"{x:.4f}" for x in a[:, 1]) + "\n      multiplicative flow from the same start: dist " + " ".join(f"{x:.4f}" for x in l[:, 0]) + " | norm " + " ".join(f"{x:.4f}" for x in l[:, 1]), flush=True)
