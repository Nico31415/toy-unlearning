"""All figures of extensions_maths.pdf in one style. Lines: theory. Markers: simulation, +-1 standard error."""
import sys, pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/home/claude/out/wd")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]; GREY = "#6b6b66"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a84", "axes.linewidth": 0.8, "lines.linewidth": 2, "legend.frameon": False, "figure.facecolor": "white", "savefig.dpi": 170, "mathtext.fontset": "cm"})
ms = lambda V: (np.mean(V, 0), np.std(V, 0, ddof=1) / np.sqrt(len(V)))
def th(a, x, m, e, col, lab): a.plot(x, m, color=col, label=lab); a.fill_between(x, m - e, m + e, color=col, alpha=0.18, lw=0)
def si(a, x, s, se, col): a.errorbar(x, s, se, fmt="o", color=col, ms=4.5, mfc="white", mew=1.4, capsize=2, lw=1.2, zorder=3)
def tag(a, t): a.set_title(t, loc="left", fontweight="bold")
H, F = "/home/claude/out/hard/", "/home/claude/out/final/"

# ---------------------------------------------------------------- 1: end points
from more_seeds import LAMS, L1, REG
P = pickle.load(open("/home/claude/out/wd/analyse_more.pkl", "rb")); fig, ax = plt.subplots(3, 3, figsize=(13, 10.5))
YA = (r"distance to retrained $\|\beta-A(K)\|^2/\|\beta^\star\|^2$", r"$\|\beta\|^2/\|\beta^\star\|^2$ after stage 2", r"loss on kept samples $\|X_K^\top\beta-y_K\|^2/\|y_K\|^2$")
YB = (r"$\|\beta\|^2/\|\beta^\star\|^2$", r"$\beta^\top\beta^\star/\|\beta^\star\|^2$", r"mean weight off the teacher $\frac{1}{d}\sum_j(\beta_j-m\beta^\star_j)$")
YC = (r"$\|\beta\|^2/\|\beta^\star\|^2$", r"$\beta^\top\beta^\star/\|\beta^\star\|^2$", r"loss on kept samples $\|X_K^\top\beta-y_K\|^2/\|y_K\|^2$")
for j in range(3):
    for i, k in enumerate((1e-2, 1.0)):
        m, e = P["D"][("A", k)]; ax[0, j].plot(LAMS, P["TA"][k][:, j], color=C[i], label=rf"$k={k:g}$"); si(ax[0, j], LAMS, m[:, j], e[:, j], C[i])
        m, e = P["D"][("B", k)]; ax[1, j].plot(L1, P["TB"][k][:, j], color=C[i], label=rf"$k={k:g}$"); si(ax[1, j], L1, m[:, j], e[:, j], C[i])
    for i, (nm, lab) in enumerate((("l1", r"$R=|\beta|$"), ("bridge", r"$R=|\beta|^{3/2}$"), ("hyp", r"$R=\psi_c$"))):
        m, e = P["D"][("C", nm, 6000)]; lam = REG[nm][2]; ax[2, j].plot(lam, P["TC"][nm][:, j], color=C[i + 2], label=lab); si(ax[2, j], lam, m[:, j], e[:, j], C[i + 2])
    ax[0, j].set_xlabel(r"total weight decay $\Lambda=\int\lambda_t\,dt$, then switched off"); ax[0, j].set_ylabel(YA[j])
    ax[1, j].set_xlabel(r"total decay on $u$ only, $\Lambda$, then switched off"); ax[1, j].set_ylabel(YB[j])
    ax[2, j].set_xlabel(r"penalty strength $\lambda$, kept on"); ax[2, j].set_ylabel(YC[j])
tag(ax[0, 0], "Weight decay switched off, stage 2 with zero targets"); tag(ax[1, 0], "Decay on $u$ only, pretraining"); tag(ax[2, 0], "Penalty kept on, stage 2 with zero targets")
ax[0, 1].legend(title="line: theory, marker: simulation"); ax[1, 0].legend(title="line: theory, marker: simulation"); ax[2, 0].legend(title="line: theory, marker: simulation")
plt.tight_layout(); plt.savefig("fig_endpoints.png"); plt.close()

# ---------------------------------------------------------------- 2: hard labels, theory vs simulation
fig, ax = plt.subplots(2, 3, figsize=(13, 7.4)); sw = pickle.load(open(H + "lam_sweep.pkl", "rb"))
for j, (col, yl) in enumerate(((0, r"$\|\beta\|^2/d$"), (1, r"$\beta^\top\beta^\star/d$"), (3, r"test accuracy $P[\mathrm{sign}\,x^\top\beta=\mathrm{sign}\,x^\top\beta^\star]$"))):
    for i, (al, o) in enumerate(sw.items()):
        ax[0, j].plot(o["fine"], o["th"][:, col], color=C[i], label=rf"$\alpha={al}$"); ls = sorted(o["S"]); S = np.array([o["S"][l][:, col] for l in ls]); si(ax[0, j], ls, S.mean(1), S.std(1, ddof=1) / np.sqrt(S.shape[1]), C[i])
    ax[0, j].set_xlabel(r"penalty strength $\lambda$ ($R=|\beta|$, kept on)"); ax[0, j].set_ylabel(yl)
tag(ax[0, 0], "Hard labels, end point with a penalty"); ax[0, 0].legend(title="line: theory, marker: simulation")
for j, (key, yl) in enumerate((("Q", r"$\|\beta(t)\|^2/d$"), ("M", r"$\beta(t)^\top\beta^\star/d$"), ("loss", r"train loss $\frac{1}{n}\sum_i\log(1+e^{-y_ix_i^\top\beta(t)})$"))):
    for i, k in enumerate((1, 0.01)):
        o = pickle.load(open(H + f"nl_{k:g}.pkl", "rb")); t = o["ts"][:-1]; (m, e), (s, se) = ms([r[key] for r in o["TH"]]), ms([r[key] for r in o["SI"]]); idx = np.arange(4, len(t), max(len(t) // 9, 1))
        th(ax[1, j], t, m, e, C[i], rf"$k={k:g}$"); si(ax[1, j], t[idx], s[idx], se[idx], C[i])
    ax[1, j].set_xlabel("training time $t$"); ax[1, j].set_ylabel(yl)
tag(ax[1, 0], r"Hard labels, no penalty: pretraining trajectory ($\alpha=0.3$)"); ax[1, 0].legend(title="line: theory, marker: simulation")
plt.tight_layout(); plt.savefig("fig_hard_labels.png"); plt.close()

# ---------------------------------------------------------------- 3: stage 2 on hard labels
runs = [("ga", 0, "ascent"), ("ga", 1, "ascent + retain term"), ("npo", 1, "NPO + retain term"), ("flip", 0, "flipped labels"), ("off", 1, "retain term only")]; cols = [C[1], C[3], C[0], C[2], GREY]
keys = [("cos", r"cosine between $\beta(t)$ and $A(K)$"), ("dist", r"distance $\|\beta(t)-A(K)\|^2/\|\beta^\star\|^2$"), ("accF", r"forget accuracy $\frac{1}{|F|}\sum_{i\in F}1[y_ix_i^\top\beta(t)>0]$")]
fig, ax = plt.subplots(2, 3, figsize=(13, 7.8))
for r, k in enumerate((1, 0.01)):
    for (kind, wr, lab), col in zip(runs, cols):
        o = pickle.load(open(H + f"nl2_{kind}_{k:g}_{wr:g}.pkl", "rb")); n1 = o["n1"]; T = len(o["TH"][0]["Q"]); t = o["dt"] * np.arange(T - n1); idx = np.arange(0, T - n1, max((T - n1) // 7, 1))
        for j, (key, yl) in enumerate(keys):
            (m, e), (s, se) = ms([x[key][n1:] for x in o["TH"]]), ms([x[key][n1:] for x in o["SI"]]); th(ax[r, j], t, m, e, col, lab); si(ax[r, j], t[idx], s[idx], se[idx], col)
            ax[r, j].set_xlabel("time in stage 2, $t-T_1$"); ax[r, j].set_ylabel(yl)
    ax[r, 1].set_ylim(0, 2.3 * ms([x["dist"][n1] for x in o["SI"]])[0]); tag(ax[r, 0], rf"Hard labels, stage 2, $k={k:g}$")
ax[0, 2].legend(title="line: theory, marker: simulation", fontsize=8.5, loc="lower left")
plt.tight_layout(); plt.savefig("fig_hard_distance.png"); plt.close()

# ---------------------------------------------------------------- 4: SGD, budget, task vector
fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2)); a0, a1, a2, a3 = ax.ravel()
for i, (k, phi, lab) in enumerate(((1, 1, r"$k=1$, full batch"), (1, 0.1, r"$k=1$, $n_b/n=0.1$"), (0.01, 0.1, r"$k=0.01$, $n_b/n=0.1$"))):
    o = pickle.load(open(F + f"sgd_{k:g}_0.04_{phi:g}.pkl", "rb")); col = [GREY, C[0], C[1]][i]
    for a, key in ((a0, "err"), (a1, "kap")):
        (m, e), (s, se) = ms([r[key] for r in o["TH"]]), ms([r[key] for r in o["SI"]]); t = np.arange(len(m)); idx = np.arange(6, len(m), len(m) // 9); th(a, t, m, e, col, lab); si(a, t[idx], s[idx], se[idx], col)
a0.set_ylabel(r"$\|\beta-\beta^\star\|^2/\|\beta^\star\|^2$"); a1.set_ylabel(r"mean over coordinates of $\log\kappa_j$")
for a in (a0, a1): a.set_xlabel(r"SGD step (step size $\gamma=0.04$)")
tag(a0, "SGD, pretraining"); tag(a1, "SGD shrinks the scale"); a0.legend(title="line: theory, marker: simulation")
o = pickle.load(open(F + "es_cov_0.01.pkl", "rb")); i1 = o["i1"]
for (cfg, r), col in zip(o["out"].items(), [C[0], C[1], C[2], C[4], C[3]]):
    (m, e), (s, se) = ms(r["th"][:, 0]), ms(r["si"][:, 0]); t = o["dt"] * (np.arange(len(m)) - i1); sl = slice(i1, len(m)); idx = np.arange(i1, len(m), 10)
    th(a2, t[sl], m[sl], e[sl], col, rf"$b={cfg[0]:g}$" + (", anchor" if cfg[1] else "")); si(a2, t[idx], s[idx], se[idx], col)
a2.set_xlabel("time in stage 2, $t-T_1$"); a2.set_ylabel(r"distance $\|\beta(t)-A(K)\|^2/\|\beta^\star\|^2$"); a2.legend(title=r"targets $b\,y$ on the forget samples", fontsize=8.5); tag(a2, r"Squared loss, $k=0.01$: loss against budget")
for i, k in enumerate((1, 0.01)):
    o = pickle.load(open(F + f"whp_{k:g}_1_1.pkl", "rb")); (m, e), (s, se) = ms([r["tv"][:, 0] for r in o["TH"]]), ms([r["tv"][:, 0] for r in o["SI"]]); th(a3, o["lams"], m, e, C[i], rf"$k={k:g}$"); si(a3, o["lams"], s, se, C[i])
a3.set_xlabel(r"task-vector coefficient $\lambda_{\rm tv}$"); a3.set_ylabel(r"cosine between $\beta$ and $A(K)$"); a3.legend(); tag(a3, "Hard labels: task vector")
plt.tight_layout(); plt.savefig("fig_final.png"); plt.close(); print("ok")
