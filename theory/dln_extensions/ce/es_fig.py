import sys, pickle, time, numpy as np
sys.path.insert(0, "/home/claude/out/setup2")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from dmft2 import run, simulate, setup, metrics
rho, aPT, aUL, dt, T2 = 0.2, 0.3, 0.2, 0.05, 6.0
CFG = ((1e-6, 10.0, "rich ($k=10^{-6}$)"), (1e-2, 6.0, "intermediate ($k=10^{-2}$)"), (1.0, 6.0, "lazy-ish ($k=1$)"))
try: out = pickle.load(open("es_fig.pkl", "rb"))
except Exception: out = {}
for k, T1, _ in CFG:
    if k in out: continue
    ts, P, W, A, i1 = setup(aPT, aUL, T1, T2, dt); t0 = time.time()
    th = run(aPT, P, W, A, ts, k, rho, N=2000); un, ul = metrics(th["Q"], th["M"], i1, len(ts) - 2)
    S = []
    for s in range(6):
        r = simulate(aPT, P, W, A, ts, k, rho, d=3000, seed=s); b = r["beta"]; last = b.shape[2] - 1
        S.append([np.sum((b[0] - b[1][:, [last]]) ** 2, 0) / r["nb"], np.sum(b[0] ** 2, 0) / r["nb"]])
    from se_reuse import stage1_grid, solve_unlearn, solve_untrain
    st = stage1_grid(aPT, k, rho, n_g=41); r = solve_unlearn(aPT, aUL, k, rho, True, st=st, n_z=41); u = solve_untrain(aPT, aUL, k, rho, st, r, True, n_e=25)
    out[k] = dict(t=ts[i1:-1] - ts[i1], un=un[i1:], ul=ul[i1:], sim=np.array(S)[:, :, i1:], end=(u["un"], r["U2"]), dn=u["dn"])
    pickle.dump(out, open("es_fig.pkl", "wb")); print(k, f"{time.time()-t0:.0f}s", flush=True)
fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))
YL = (r"untraining:  $\|U(\tau)-A(D\setminus F)\|^2/\|\beta^\star\|^2$", r"unlearning:  $\|U(\tau)-A(D\setminus F_{\rm full})\|^2/\|\beta^\star\|^2$")
for ci, (k, T1, name) in enumerate(CFG):
    o = out[k]; m, e = o["sim"].mean(0), o["sim"].std(0, ddof=1) / np.sqrt(o["sim"].shape[0]); sel = np.unique(np.r_[0:20:2, 20:len(o["t"]):8])
    for ri, (th, col) in enumerate(((o["un"], "#1b6ca8"), (o["ul"], "#c1553b"))):
        a = ax[ri, ci]
        a.plot(o["t"], th, "-", color=col, lw=2.2, label="exact large-$d$ dynamics (two-stage DMFT), nothing fitted")
        a.errorbar(o["t"][sel], m[ri][sel], yerr=e[ri][sel], fmt="o", color=col, mfc="white", ms=5, capsize=2, label="simulation, $d=3000$, 6 seeds (±1 s.e.)")
        a.axhline(o["end"][ri], color="k", ls=":", lw=1.3, label="exact converged end point (static theory)")
        if ri == 0:
            j = int(np.argmin(th)); a.plot(o["t"][j], th[j], "*", color="k", ms=13, zorder=5, label=f"best stopping time (theory): $\\tau={o['t'][j]:.2f}$")
            a.set_title(name, fontsize=12)
        a.set_ylim(0, None); a.set_xlim(-0.1, T2); a.grid(alpha=.25); a.legend(fontsize=8)
        if ri == 1: a.set_xlabel(r"stopping time $\tau$ of stage 2 (gradient-flow time after pretraining)")
        if ci == 0: a.set_ylabel(YL[ri])
fig.suptitle(r"Early-stopped stage 2, zero targets on the deleted samples (reused), no anchor.  All-forget coordinates, $\alpha_{PT}=0.3$, $\alpha_{UL}=0.2$, $\rho=0.2$.", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig("fig_early_stopping.png", dpi=140); print("ok")
for k, _, _ in CFG:
    o = out[k]; j = int(np.argmin(o["un"])); m = o["sim"].mean(0)
    print(f"k={k:g}: do nothing {o['dn']:.3f} | best stop tau={o['t'][j]:.2f}: theory {o['un'][j]:.3f}, sim {m[0][j]:.3f} | converged exact {o['end'][0]:.3f}; at tau=6: theory {o['un'][-1]:.3f}, sim {m[0][-1]:.3f} | unlearn at tau=6: theory {o['ul'][-1]:.4f}, sim {m[1][-1]:.4f}, exact {o['end'][1]:.4f}")
