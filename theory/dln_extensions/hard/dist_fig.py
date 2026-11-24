import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
runs = [("ga", 0, "ascent", "#d62728"), ("ga", 1, "ascent + retain term", "#ff7f0e"), ("npo", 1, "NPO + retain term", "#1f77b4"), ("flip", 0, "flipped labels", "#2ca02c"), ("off", 1, "retain term only", "#7f7f7f")]
keys = [("dist", r"distance to retrained $\|\beta(t)-A(D\setminus F)\|^2/\|\beta^\star\|^2$"), ("cos", r"cosine to retrained $\cos(\beta(t),A(D\setminus F))$"), ("accF", r"accuracy on forget samples $\frac{1}{|F|}\sum_{i\in F}1[y_ix_i^\top\beta(t)>0]$")]
st = lambda L, key: (np.mean([o[key] for o in L], 0), np.std([o[key] for o in L], 0, ddof=1) / np.sqrt(len(L)))
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
for r, k in enumerate((1, 0.01)):
    for kind, wr, lab, col in runs:
        o = pickle.load(open(f"nl2_{kind}_{k:g}_{wr:g}.pkl", "rb")); n1 = o["n1"]; T = len(o["TH"][0]["Q"]); t = o["dt"] * np.arange(T - n1)
        for j, (key, yl) in enumerate(keys):
            (m, e), (s, se) = st(o["TH"], key), st(o["SI"], key); a = ax[r, j]
            a.plot(t, m[n1:], color=col, label=lab + " (line: theory)"); a.fill_between(t, (m - e)[n1:], (m + e)[n1:], color=col, alpha=0.2)
            idx = np.arange(0, T - n1, max((T - n1) // 8, 1)); a.errorbar(t[idx], s[n1:][idx], se[n1:][idx], fmt="o", color=col, ms=3.5, capsize=2)
            a.set_xlabel("stage-2 training time $t-T_1$"); a.set_ylabel(yl); a.set_title(rf"$k={k:g}$; markers: simulation $d=3000$, 16 runs", loc="left", fontsize=9)
    ax[r, 0].set_ylim(0, 2.5 * st(o["SI"], "dist")[0][n1])
ax[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_hard_distance.png", dpi=140)
