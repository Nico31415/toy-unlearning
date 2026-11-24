"""Switched-off regulariser other than weight decay: decay only the positive pathway, (lam/2)|u|^2 (Jacobs et al. Example 3.13, c_u=1, c_v=0).
u -> u e^{-Lam}, v unchanged: scale c -> c e^{-Lam} and the mirror variable theta = log(u/v) shifts by -Lam on every coordinate.
End point after switch-off (pretraining from the balanced start): argmin sum_j [psi_{c'}(b_j) + Lam b_j] s.t. X^T b = y, c' = c e^{-Lam}.
Replica channel: b = prox_{R/mu}(b* + s h), R'(b) = arcsinh(b/c') + Lam;  E prox' = alpha, s^2 = E (b - b*)^2 / alpha."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import numpy as np, pickle, sys
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rho, al, d = 0.2, 0.3, 400
H = np.linspace(-9, 9, 4001); WH = np.exp(-H**2 / 2); WH /= WH.sum()

def prox(x, mu, c, tilt):
    lo = x - (np.abs(tilt) + 40) / mu - 1e-9; hi = x + (np.abs(tilt) + 40) / mu + 1e-9      # arcsinh term is bounded by ~40 here
    for _ in range(90):
        mid = 0.5 * (lo + hi); f = np.arcsinh(mid / c) + tilt + mu * (mid - x)
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    return 0.5 * (lo + hi)
def theory(c, tilt):
    from scipy.optimize import brentq
    amp = rho ** -0.5
    def mom(mu, s):
        tot = dict(dp=0, q=0, m=0, e=0, mean=0)
        for bstar, wt in ((0.0, 1 - rho), (amp, rho / 2), (-amp, rho / 2)):
            b = prox(bstar + s * H, mu, c, tilt); dpr = mu / (mu + 1 / np.sqrt(b * b + c * c))
            tot["dp"] += wt * (WH @ dpr); tot["q"] += wt * (WH @ b**2); tot["m"] += wt * bstar * (WH @ b); tot["e"] += wt * (WH @ (b - bstar)**2); tot["mean"] += wt * (WH @ b)
        return tot
    s = 1.0
    for it in range(400):
        lm = brentq(lambda v: mom(np.exp(v), s)["dp"] - al, -18, 18, xtol=1e-12)
        m = mom(np.exp(lm), s); sn = np.sqrt(m["e"] / al)
        if abs(sn - s) < 1e-10: break
        s = 0.5 * s + 0.5 * sn
    m["resid"] = abs(sn - s); return m

def sim(k, Lam, seed, au=1.0, av=0.0, lam0=0.5, lr=0.01):
    rng = np.random.default_rng(1000 + seed); bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
    X = rng.standard_normal((d, int(al * d))) / np.sqrt(d); y = X.T @ bs
    a0 = np.sqrt(np.sqrt(k) / 4); u = np.full(d, a0); v = u.copy(); t, Ton = 0.0, Lam / lam0
    while t < 800:
        r = X.T @ (u * u - v * v) - y; g = X @ r; lam = lam0 if t < Ton else 0.0
        if t >= Ton and np.max(np.abs(r)) < 1e-6: break
        u, v = u - lr * (2 * u * g + lam * au * u), v - lr * (-2 * v * g + lam * av * v); t += lr
    b = u * u - v * v; mm = b @ bs / (bs @ bs); return [b @ b / (bs @ bs), mm, b.mean() - mm * bs.mean()]

LAMS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0); out = {}
for k in (1.0, 1e-2):
    c = np.sqrt(k) / 2
    for Lam in LAMS:
        th = theory(c * np.exp(-Lam), Lam); S = np.array([sim(k, Lam, s) for s in range(4)])
        out[(k, Lam)] = ([th["q"], th["m"], th["mean"]], S.mean(0), S.std(0, ddof=1) / 2)
        print(k, Lam, np.round(out[(k, Lam)][0], 4), np.round(S.mean(0), 4), f"resid {th['resid']:.0e}", flush=True)
pickle.dump(out, open("left2.pkl", "wb"))
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6)); COL = {1e-2: "#1b6ca8", 1.0: "#c1553b"}
YL = (r"$\|A\|^2/\|\beta^\star\|^2$", r"overlap  $\langle A,\beta^\star\rangle/\|\beta^\star\|^2$", r"mean weight off the teacher  $\frac{1}{d}\sum_j(A_j-m\beta^\star_j)$  (0 without the tilt)")
for j in range(3):
    a = ax[j]
    for k in (1e-2, 1.0):
        a.plot(LAMS, [out[(k, L)][0][j] for L in LAMS], "-", color=COL[k], lw=2, label=f"$k={k:g}$: replica curve (tilted potential)")
        a.errorbar(LAMS, [out[(k, L)][1][j] for L in LAMS], yerr=[out[(k, L)][2][j] for L in LAMS], fmt="o", color=COL[k], mfc="white", ms=6, capsize=2, label=f"$k={k:g}$: gradient descent, $d=400$, 4 seeds")
    a.set_xlabel(r"$\Lambda=\int\lambda_t\,dt$  (total decay on the positive pathway $u$ only, then switched off)"); a.set_ylabel(YL[j]); a.grid(alpha=.25)
    if j == 0: a.legend(fontsize=8)
fig.suptitle(r"Switched-off one-sided decay $\frac{\lambda_t}{2}\|u\|^2$ during pretraining ($\alpha_{PT}=0.3$, $\rho=0.2$), then training to convergence. Lines: theory, nothing fitted.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig("fig_onesided_decay.png", dpi=140); print("ok")
