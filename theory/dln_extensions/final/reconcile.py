"""Why did the earlier simulation say ascent is no better than doing nothing, and the hard-label theory the opposite?
2 x 2 in simulation: pretraining (soft labels, converged interpolator | hard labels, cross-entropy stopped at T1) x sample ratios (old 0.3/0.2 | new 0.6/0.3).
Ascent on the deleted samples; cosine to the retrained model along the whole path."""
import sys, numpy as np, pickle
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "ce"))
from scipy.special import expit
from multiprocessing import Pool
from ce_tests import project
rho, GAM, d, T1 = 0.2, 3.0, 1000, 3.0

def ce_flow(X, y, c, T, dt=0.02):
    th = np.zeros(X.shape[0])
    for _ in range(int(T / dt)): th -= 4 * dt * (X @ (-y * expit(-y * (X.T @ (c * np.sinh(th))))))
    return c * np.sinh(th)

def cell(a):
    pre, aPT, aUL, k, seed = a; c = np.sqrt(k) / 2; rng = np.random.default_rng(seed)
    bs = np.zeros(d); bs[:int(rho * d)] = rng.choice([-1, 1], int(rho * d)) * rho**-0.5; n, f = int(aPT * d), int(aUL * d)
    X = rng.standard_normal((d, n)) / np.sqrt(d); Xd, XK = X[:, :f], X[:, f:]; y = np.sign(X.T @ bs)
    if pre == "soft": A = project(np.zeros(d), c, A_eq=X, b_eq=GAM * X.T @ bs)[0]; Au = project(np.zeros(d), c, A_eq=XK, b_eq=GAM * XK.T @ bs)[0]
    else: A = ce_flow(X, y, c, T1); Au = ce_flow(XK, y[f:], c, T1)
    Xho = rng.standard_normal((d, 4000)) / np.sqrt(d); yho = np.sign(Xho.T @ bs); acc = lambda B: np.mean(np.sign(Xho.T @ B) == yho)
    cos = lambda B: B @ Au / np.sqrt((B @ B) * (Au @ Au)); Zd = Xd * y[:f]; th = np.arcsinh(A / c); L2 = np.linalg.norm(Zd, 2)**2; path = []
    for it in range(8000):
        B = c * np.sinh(th); m = Zd.T @ B
        if it % 2 == 0: path.append((cos(B), acc(B), np.mean(m > 0), B @ B / d))
        if path[-1][1] < 0.52 or np.abs(th).max() > 30: break
        g = Zd @ expit(-m); eta = 0.03 / (4 * np.max(np.abs(g)) + 1e-300)          # time reparametrised: same path
        th -= 4 * eta * g
    P = np.array(path); i = int(np.argmax(P[:, 0])); ra = acc(Au); j = np.argmax(P[:, 1] <= ra) if np.any(P[:, 1] <= ra) else len(P) - 1
    return (pre, aPT, aUL, k), [P[0, 0], P[i, 0], P[j, 0], P[0, 1], P[i, 1], ra, P[0, 3], Au @ Au / d]

if __name__ == "__main__":
    jobs = [(pre, aPT, aUL, k, s) for pre in ("soft", "hard") for aPT, aUL in ((0.3, 0.2), (0.6, 0.3)) for k in (1.0, 0.01) for s in range(8)]
    with Pool(2) as p: R = p.map(cell, jobs)
    pickle.dump(R, open("reconcile.pkl", "wb")); print("pretraining aPT aUL k | cos: do nothing, best on path, at retrained accuracy | held-out acc: start, at best cos, retrained | norm: start, retrained")
    for key in dict.fromkeys(r[0] for r in R):
        V = np.array([r[1] for r in R if r[0] == key]); m, e = V.mean(0), V.std(0, ddof=1) / np.sqrt(len(V))
        print(f"{key[0]:5s} {key[1]} {key[2]} {key[3]:5g} | {m[0]:.3f}±{e[0]:.3f}  {m[1]:.3f}±{e[1]:.3f}  {m[2]:.3f}±{e[2]:.3f} | {m[3]:.3f} {m[4]:.3f} {m[5]:.3f} | {m[6]:.2f} {m[7]:.2f}", flush=True)
