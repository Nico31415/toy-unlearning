import sys, pickle, zlib, numpy as np
sys.path.insert(0, "/home/claude/out/setup2"); sys.path.insert(0, "/home/claude/out/ce")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ce_tests import project
from se_wd import solve as solve_wd, lasso
rho, aPT, aUL, d = 0.2, 0.3, 0.2, 400
LAMS = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5); KS = (1e-2, 1.0); lam0 = 0.5

def sim_switch(k, Lam, seed, lr=0.01):
    """real (u, v) gradient descent in stage 2 with weight decay lam0 for time Lam/lam0, then off, to convergence."""
    rng = np.random.default_rng(zlib.crc32(repr(("WD", seed)).encode())); c = np.sqrt(k) / 2
    bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
    n, f = int(aPT * d), int(aUL * d); X = rng.standard_normal((d, n)) / np.sqrt(d); Xf, XK = X[:, :f], X[:, f:]
    A = project(np.zeros(d), c, A_eq=X, b_eq=X.T @ bs)[0]; Au = project(np.zeros(d), c, A_eq=XK, b_eq=XK.T @ bs)[0]
    s = np.sqrt(A * A + c * c); u, v = np.sqrt((s + A) / 2), np.sqrt((s - A) / 2)
    t, Ton = 0.0, Lam / lam0
    while t < 600:
        r = Xf.T @ (u * u - v * v); g = Xf @ r; lam = lam0 if t < Ton else 0.0
        if t >= Ton and np.max(np.abs(r)) < 1e-6: break
        u, v = u - lr * (2 * u * g + lam * u), v - lr * (-2 * v * g + lam * v); t += lr
    U = u * u - v * v; nb = bs @ bs; zk = XK.T @ bs
    return [np.sum((U - Au)**2) / nb, U @ U / nb, np.mean((XK.T @ U - zk)**2) / np.mean(zk**2)]

try: out = pickle.load(open("wd_fig.pkl", "rb"))
except Exception: out = {}
if "switch" not in out:
    from se_reuse import stage1_grid, solve_unlearn, solve_untrain
    res = {}
    for k in KS:
        st = stage1_grid(aPT, k, rho, n_g=41)
        for Lam in LAMS:
            S = np.exp(-2 * Lam); r = solve_unlearn(aPT, aUL, k, rho, True, st=st, n_z=41, S=S); u = solve_untrain(aPT, aUL, k, rho, st, r, True, n_e=25)
            sims = np.array([sim_switch(k, Lam, s) for s in range(4)])
            res[(k, Lam)] = dict(th=[u.get("un", np.nan), r.get("U2", np.nan), r.get("kept", np.nan)], sim=sims.mean(0), se=sims.std(0, ddof=1) / 2, keys=list(r.keys()))
            print(k, Lam, np.round(res[(k, Lam)]["th"], 4), np.round(sims.mean(0), 4), flush=True)
    out["switch"] = res; pickle.dump(out, open("wd_fig.pkl", "wb"))
if "const" not in out:
    G = [(aUL / aPT, 1.0, 0.0), (1 - aUL / aPT, 1.0, 1.0)]; lams = np.linspace(0.005, 0.3, 9); res = {}; dd = 1500
    for lam in lams:
        th = solve_wd(aPT, G, lam, rho); S = []
        for seed in range(4):
            rng = np.random.default_rng(seed); bs = np.where(rng.random(dd) < rho, rng.choice([-1, 1], dd) * rho**-0.5, 0.0)
            n = int(aPT * dd); f = int(aUL * dd); X = rng.standard_normal((dd, n)) / np.sqrt(dd); z = X.T @ bs; t = z.copy(); t[:f] = 0
            b = lasso(X, t, np.ones(n), lam); r = X.T @ b - t
            S.append([b @ b / (bs @ bs), b @ bs / (bs @ bs), np.mean(r[f:]**2) / np.mean(z**2)])
        S = np.array(S); res[lam] = dict(th=[th["Q"], th["M"], th["res"][1]], sim=S.mean(0), se=S.std(0, ddof=1) / 2)
    out["const"] = res; pickle.dump(out, open("wd_fig.pkl", "wb"))

fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.8)); COL = {1e-2: "#1b6ca8", 1.0: "#c1553b"}
YL1 = (r"untraining:  $\|U-A(D\setminus F)\|^2/\|\beta^\star\|^2$", r"unlearning:  $\|U-A(D\setminus F_{\rm full})\|^2/\|\beta^\star\|^2$", r"loss on kept forget samples:  $\|X_K^\top U-y_K\|^2/\|y_K\|^2$")
for j in range(3):
    a = ax[0, j]
    for k in KS:
        th = np.array([out["switch"][(k, L)]["th"][j] for L in LAMS]); m = np.array([out["switch"][(k, L)]["sim"][j] for L in LAMS]); e = np.array([out["switch"][(k, L)]["se"][j] for L in LAMS])
        a.plot(LAMS, th, "-", color=COL[k], lw=2, label=f"$k={k:g}$: replica curve at $S=e^{{-2\\Lambda}}$")
        a.errorbar(LAMS, m, yerr=e, fmt="o", color=COL[k], mfc="white", ms=6, capsize=2, label=f"$k={k:g}$: gradient descent with weight decay, $d=400$, 4 seeds")
    a.set_xlabel(r"$\Lambda=\int\lambda_t\,dt$  (total weight decay applied before switching it off)"); a.set_ylabel(YL1[j]); a.set_ylim(0, None); a.grid(alpha=.25)
    if j == 0: a.legend(fontsize=8)
ax[0, 1].set_title("Weight decay switched off, then trained to convergence  (stage 2: deleted samples to zero targets, reused, no anchor)", fontsize=11)
lams = sorted(out["const"]); YL2 = (r"$\|U\|^2/\|\beta^\star\|^2$", r"overlap  $\langle U,\beta^\star\rangle/\|\beta^\star\|^2$", r"residual on the anchored kept samples")
for j in range(3):
    a = ax[1, j]; th = [out["const"][l]["th"][j] for l in lams]; m = [out["const"][l]["sim"][j] for l in lams]; e = [out["const"][l]["se"][j] for l in lams]
    a.plot(lams, th, "-", color="#4b8b3b", lw=2, label="replica curve (lasso channel)"); a.errorbar(lams, m, yerr=e, fmt="s", color="#4b8b3b", mfc="white", ms=6, capsize=2, label="simulation, $d=1500$, 4 seeds")
    a.set_xlabel(r"$\lambda$  (constant weight decay, never switched off; penalty $\lambda\|\beta\|_1$)"); a.set_ylabel(YL2[j]); a.set_ylim(0, None); a.grid(alpha=.25)
    if j == 0: a.legend(fontsize=8)
ax[1, 1].set_title("Constant weight decay  (stage 2: deleted samples to zero targets, kept samples anchored on their labels)", fontsize=11)
fig.suptitle(r"Weight decay: replica curves (lines) vs simulation (markers, ±1 s.e.).  All-forget coordinates, $\alpha_{PT}=0.3$, $\alpha_{UL}=0.2$, $\rho=0.2$.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig("fig_weight_decay.png", dpi=140); print("ok")
