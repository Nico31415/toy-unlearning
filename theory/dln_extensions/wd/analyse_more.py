import sys, pickle, numpy as np
sys.path.insert(0, "/home/claude/out/setup2"); sys.path.insert(0, "/home/claude/out/ce")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from se_wd import solve, prox_l1
src = open("left2.py").read(); exec(src[:src.index("def sim(")])            # prox, theory for the tilted channel (definitions only)
from more_seeds import LAMS, L1, REG, c0, rho, aPT, aUL
res = pickle.load(open("more_seeds.pkl", "rb"))
def agg(tag, key):
    v = np.array([r[3] for r in res if r[0] == tag and r[1] == key]); return v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v)), len(v)
Z = []
# ---- A: switched-off weight decay
from se_reuse import stage1_grid, solve_unlearn, solve_untrain
TA = {}
for k in (1e-2, 1.0):
    st = stage1_grid(aPT, k, rho, n_g=61); rows = []
    for Lam in LAMS:
        r = solve_unlearn(aPT, aUL, k, rho, True, st=st, n_z=61, S=np.exp(-2 * Lam)); u = solve_untrain(aPT, aUL, k, rho, st, r, True, n_e=31)
        rows.append([u["un"], r["U2"], r["kept"]])
    TA[k] = np.array(rows)
# ---- B: one-sided decay
TB = {k: np.array([[(t := theory(np.sqrt(k) / 2 * np.exp(-L), L))["q"], t["m"], t["mean"]] for L in L1]) for k in (1e-2, 1.0)}
# ---- C: constant regularisers
def make_prox(R1, R2):
    def prox_(x, t):
        lo, hi = np.minimum(x, 0.0), np.maximum(x, 0.0)
        for _ in range(70):
            mid = 0.5 * (lo + hi); f = mid + t * R1(mid) - x; lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
        b = 0.5 * (lo + hi); return b, 1 / (1 + t * R2(b))
    return prox_
PX = {"hyp": make_prox(lambda b: np.arcsinh(b / c0), lambda b: 1 / np.sqrt(b * b + c0 * c0)), "bridge": make_prox(lambda b: np.sign(b) * np.sqrt(np.abs(b)), lambda b: 0.5 / np.sqrt(np.abs(b) + 1e-300)), "l1": prox_l1}
G = [(aUL / aPT, 1.0, 0.0), (1 - aUL / aPT, 1.0, 1.0)]
TC = {nm: np.array([[(t := solve(aPT, G, lam, rho, prox=PX[nm], iters=6000))["Q"], t["M"], t["res"][1]] for lam in REG[nm][2]]) for nm in REG}

def report(title, th, m, e, n, labels):
    z = (m - th) / np.maximum(e, 1e-12); rel = (m - th) / np.maximum(np.abs(th), 1e-12)
    print(f"\n{title}  ({n} seeds)"); 
    for j, lab in enumerate(labels):
        ok = np.abs(th[:, j]) > 2e-3
        print(f"  {lab}: max |sim-theory|/s.e. = {np.max(np.abs(z[:, j])):.1f}; rel. diff range [{100*rel[ok, j].min():+.1f}%, {100*rel[ok, j].max():+.1f}%]; typical s.e./value = {100*np.median(e[ok, j]/np.abs(m[ok, j])):.1f}%")
    Z.extend(list(z.ravel()))
LA = ("untraining dist", "unlearning dist", "kept loss"); LB = ("norm", "overlap", "mean off-teacher weight"); LC = ("norm", "overlap", "kept residual")
D = {}
for k in (1e-2, 1.0):
    m, e, n = agg("A", k); D[("A", k)] = (m, e); report(f"A. switched-off weight decay, stage 2, k={k:g}, d=1600", TA[k], m, e, n, LA)
for k in (1e-2, 1.0):
    m, e, n = agg("B", k); D[("B", k)] = (m, e); report(f"B. switched-off one-sided decay, pretraining, k={k:g}, d=1600", TB[k], m, e, n, LB)
for nm in REG:
    for dd in (1500, 6000):
        m, e, n = agg("C", (nm, dd)); D[("C", nm, dd)] = (m, e); report(f"C. constant regulariser '{nm}', d={dd}", TC[nm], m, e, n, LC)
Z = np.array(Z); Z = Z[np.isfinite(Z)]
print(f"\nall comparisons: {len(Z)} numbers; |z|<2 in {100*np.mean(np.abs(Z)<2):.0f}%, |z|<3 in {100*np.mean(np.abs(Z)<3):.0f}%, max |z| = {np.abs(Z).max():.1f}")
pickle.dump(dict(TA=TA, TB=TB, TC=TC, D=D), open("analyse_more.pkl", "wb"))
# ---- figure
fig, ax = plt.subplots(3, 3, figsize=(15.5, 12.6)); COL = {1e-2: "#1b6ca8", 1.0: "#c1553b"}
YA = (r"untraining:  $\|U-A(D\setminus F)\|^2/\|\beta^\star\|^2$", r"unlearning:  $\|U-A(D\setminus F_{\rm full})\|^2/\|\beta^\star\|^2$", r"loss on kept forget samples  $\|X_K^\top U-y_K\|^2/\|y_K\|^2$")
YB = (r"$\|A\|^2/\|\beta^\star\|^2$", r"overlap  $\langle A,\beta^\star\rangle/\|\beta^\star\|^2$", r"mean off-teacher weight  $\frac{1}{d}\sum_j(A_j-m\beta^\star_j)$")
for j in range(3):
    for k in (1e-2, 1.0):
        m, e = D[("A", k)]; ax[0, j].plot(LAMS, TA[k][:, j], "-", color=COL[k], lw=2, label=f"$k={k:g}$ theory"); ax[0, j].errorbar(LAMS, m[:, j], yerr=e[:, j], fmt="o", color=COL[k], mfc="white", ms=5, capsize=2, label=f"$k={k:g}$ simulation, $d=1600$, 16 seeds")
        m, e = D[("B", k)]; ax[1, j].plot(L1, TB[k][:, j], "-", color=COL[k], lw=2, label=f"$k={k:g}$ theory"); ax[1, j].errorbar(L1, m[:, j], yerr=e[:, j], fmt="o", color=COL[k], mfc="white", ms=5, capsize=2, label=f"$k={k:g}$ simulation, $d=1600$, 24 seeds")
    ax[0, j].set_xlabel(r"$\Lambda=\int\lambda_t\,dt$  (total weight decay, then switched off)"); ax[0, j].set_ylabel(YA[j])
    ax[1, j].set_xlabel(r"$\Lambda=\int\lambda_t\,dt$  (total decay on $u$ only, then switched off)"); ax[1, j].set_ylabel(YB[j])
    for nm, col, lab in (("l1", "#4b8b3b", "weight decay ($\\ell_1$)"), ("hyp", "#7a4fa3", "hyperbolic entropy"), ("bridge", "#b8860b", "bridge $|\\beta|^{3/2}$")):
        m, e = D[("C", nm, 6000)]; x = np.arange(3) + {"l1": 0, "hyp": 4, "bridge": 8}[nm]
        ax[2, j].errorbar(x, 100 * (m[:, j] / TC[nm][:, j] - 1), yerr=100 * e[:, j] / TC[nm][:, j], fmt="s", color=col, ms=6, capsize=3, label=lab + ", $d=6000$, 6 seeds")
        m, e = D[("C", nm, 1500)]; ax[2, j].errorbar(x + 0.25, 100 * (m[:, j] / TC[nm][:, j] - 1), yerr=100 * e[:, j] / TC[nm][:, j], fmt="o", color=col, mfc="white", ms=5, capsize=3, label=lab + ", $d=1500$, 12 seeds")
    ax[2, j].axhline(0, color="k", lw=1); ax[2, j].set_xticks([1, 5, 9]); ax[2, j].set_xticklabels(["weight decay", "hyperbolic entropy", "bridge"]); ax[2, j].set_xlabel("regulariser (three values of $\\lambda$ each, increasing left to right)")
    ax[2, j].set_ylabel("simulation / theory $-1$  (%):  " + ("norm", "overlap", "kept residual")[j])
    for i in range(3): ax[i, j].grid(alpha=.25)
ax[0, 0].legend(fontsize=7.5); ax[1, 0].legend(fontsize=7.5); ax[2, 0].legend(fontsize=7, ncol=1)
ax[0, 1].set_title("Weight decay switched off (stage 2: deleted samples to zero targets, no anchor)", fontsize=10.5)
ax[1, 1].set_title("One-sided decay switched off (pretraining)", fontsize=10.5); ax[2, 1].set_title("Constant regularisers: relative difference from theory at two sizes", fontsize=10.5)
fig.suptitle("Replica curves vs exact finite-size end points, with more seeds (markers ±1 s.e.; lines: theory, nothing fitted)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .97]); fig.savefig("fig_regularisers_tight.png", dpi=130); print("ok")
