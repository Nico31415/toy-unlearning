import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import pickle, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = pickle.load(open("ksweep_sim.pkl", "rb")); T = pickle.load(open("ksweep_theory.pkl", "rb"))
KS = sorted({r["k"] for r in S}); AUL = sorted({r["aul"] for r in S}); lx = np.log10(KS)
def sim(aul, f):
    v = np.array([[f(r) for r in S if r["k"] == k and r["aul"] == aul] for k in KS])
    return v.mean(1), v.std(1, ddof=1) / np.sqrt(v.shape[1])
def th(aul, key):
    return np.array([[r[key] for r in T if r["k"] == k and r["aul"] == aul][0] for k in KS])
fig, ax = plt.subplots(2, len(AUL), figsize=(6.3 * len(AUL), 8.6), sharey=False)
YL = (r"untraining:  $\|U-A(D\setminus F)\|^2/\|\beta^\star\|^2$", r"unlearning:  $\|U-A(D\setminus F_{\rm full})\|^2/\|\beta^\star\|^2$")
C = {"nothing": "0.45", "uniform": "#1b6ca8", "GA": "#c1553b", "NPO4": "#4b8b3b"}
for ci, aul in enumerate(AUL):
    for ri, (mi, key) in enumerate(((0, "untrain"), (1, "unlearn"))):
        a = ax[ri, ci]
        for name, lab in (("nothing", "do nothing"), ("uniform", "CE to the uniform target, converged")):
            m, e = sim(aul, lambda r: r[name][mi])
            a.errorbar(lx, m, yerr=e, fmt="o", color=C[name], ms=6, capsize=2, zorder=4)
            a.plot(lx, th(aul, f"{name}_{key}"), "-", color=C[name], lw=2, label=f"{lab} (line: exact large-$d$ theory)")
        for name, lab in (("GA", "gradient ascent, best stopping time"), ("NPO4", r"NPO ($\beta=4$), best stopping time")):
            m, e = sim(aul, lambda r: r[name][f"best_{key}"])
            a.errorbar(lx, m, yerr=e, fmt="s--", color=C[name], ms=6, capsize=2, mfc="white", lw=1.3, label=f"{lab} (simulation only)")
        a.set_xticks(lx); a.grid(alpha=.25); a.set_ylim(0, 0.62 if ri == 0 else 0.56)
        if ri == 1: a.set_xlabel(r"$\log_{10}k$  (initialisation scale; left = rich, right = lazy)")
        a.set_ylabel(YL[ri])
        if ri == 0: a.set_title(rf"$\alpha_{{UL}}=|F|/d={aul}$", fontsize=12)
ax[0, 0].legend(fontsize=8.5, loc="upper right")
fig.suptitle(r"CE toy against $k$: all-forget coordinates, $\alpha_{PT}=n_F/d=0.3$, $\rho=0.2$, logit scale $\gamma=3$, $d=400$, 4 seeds, "
             "deleted samples reused, no anchor.\nMarkers: simulation (±1 s.e.). Solid lines: theory, nothing fitted.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig("fig_ce_ksweep.png", dpi=140); print("ok")
