"""Other explicit regularisers: replica channel with a numerical prox vs simulation.
R1(b) = hyperbolic entropy with scale c0 (the network's own implicit-bias potential, used as an explicit penalty), R1' = arcsinh(b/c0)
R2(b) = (2/3)|b|^{3/2} (bridge penalty), R2' = sign(b) sqrt|b|
Problem: deleted samples (alpha_UL=0.2) to zero targets, kept samples (0.1) anchored on their labels."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import numpy as np, pickle
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from se_wd import solve
rho, aPT, aUL, d, c0 = 0.2, 0.3, 0.2, 1500, 0.05
REG = {"hyperbolic entropy, $R'(\\beta)=\\mathrm{arcsinh}(\\beta/c_0)$, $c_0=0.05$":
           (lambda b: b * np.arcsinh(b / c0) - np.sqrt(b * b + c0 * c0), lambda b: np.arcsinh(b / c0), lambda b: 1 / np.sqrt(b * b + c0 * c0), np.linspace(0.003, 0.12, 8)),
       "bridge penalty, $R(\\beta)=\\frac{2}{3}|\\beta|^{3/2}$":
           (lambda b: 2 / 3 * np.abs(b) ** 1.5, lambda b: np.sign(b) * np.sqrt(np.abs(b)), lambda b: 0.5 / np.sqrt(np.abs(b) + 1e-300), np.linspace(0.01, 0.5, 8))}
def make_prox(R1, R2):
    def prox(x, t):                                    # solve b + t R'(b) = x by bisection (R' increasing, odd)
        lo, hi = np.minimum(x, 0.0), np.maximum(x, 0.0)
        for _ in range(70):
            mid = 0.5 * (lo + hi); f = mid + t * R1(mid) - x
            lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
        b = 0.5 * (lo + hi); return b, 1 / (1 + t * R2(b))
    return prox
G = [(aUL / aPT, 1.0, 0.0), (1 - aUL / aPT, 1.0, 1.0)]
out = {}
for name, (R0, R1, R2, lams) in REG.items():
    rows = []
    for lam in lams:
        th = solve(aPT, G, lam, rho, prox=make_prox(R1, R2), iters=3000)
        S = []
        for seed in range(4):
            rng = np.random.default_rng(seed); bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
            n, f = int(aPT * d), int(aUL * d); X = rng.standard_normal((d, n)) / np.sqrt(d); z = X.T @ bs; t = z.copy(); t[:f] = 0
            obj = lambda b: (0.5 * np.sum((X.T @ b - t) ** 2) + lam * np.sum(R0(b)), X @ (X.T @ b - t) + lam * R1(b))
            b = minimize(obj, np.full(d, 1e-3), jac=True, method="L-BFGS-B", options=dict(maxiter=20000, ftol=1e-15, gtol=1e-9, maxcor=30)).x
            r = X.T @ b - t; S.append([b @ b / (bs @ bs), b @ bs / (bs @ bs), np.mean(r[f:] ** 2) / np.mean(z ** 2)])
        S = np.array(S); rows.append((lam, [th["Q"], th["M"], th["res"][1]], S.mean(0), S.std(0, ddof=1) / 2, th["err"]))
        print(name[:12], f"lam={lam:.3f}", np.round(rows[-1][1], 4), np.round(rows[-1][2], 4), f"fp err {th['err']:.0e}", flush=True)
    out[name] = rows
pickle.dump(out, open("reg_fig.pkl", "wb"))
fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6)); YL = (r"$\|U\|^2/\|\beta^\star\|^2$", r"overlap  $\langle U,\beta^\star\rangle/\|\beta^\star\|^2$", "residual on the anchored kept samples")
for ri, (name, rows) in enumerate(out.items()):
    lams = [r[0] for r in rows]
    for j in range(3):
        a = ax[ri, j]; a.plot(lams, [r[1][j] for r in rows], "-", color="#7a4fa3", lw=2, label="replica curve (numerical prox)")
        a.errorbar(lams, [r[2][j] for r in rows], yerr=[r[3][j] for r in rows], fmt="D", color="#7a4fa3", mfc="white", ms=6, capsize=2, label="simulation, $d=1500$, 4 seeds")
        a.set_xlabel(r"$\lambda$  (strength of the explicit penalty $\lambda\sum_jR(\beta_j)$)"); a.set_ylabel(YL[j]); a.set_ylim(0, None); a.grid(alpha=.25)
        if j == 0: a.legend(fontsize=8)
        if j == 1: a.set_title(name, fontsize=11)
fig.suptitle(r"Other explicit regularisers: replica curves (lines) vs simulation (markers, ±1 s.e.).  Deleted samples to zero targets, kept samples anchored; $\alpha_{PT}=0.3$, $\alpha_{UL}=0.2$, $\rho=0.2$.", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig("fig_other_regularisers.png", dpi=140); print("ok")
