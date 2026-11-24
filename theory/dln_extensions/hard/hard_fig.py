import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(3, 4, figsize=(15, 10)); C = {0.3: "#1f77b4", 1.5: "#d62728"}
st = lambda L, key: (np.mean([o[key] for o in L], 0), np.std([o[key] for o in L], 0, ddof=1) / np.sqrt(len(L)))
# row 1: penalty kept on
sw = pickle.load(open("lam_sweep.pkl", "rb")); labs = [r"$\|U\|^2/d$", r"$U^\top\beta^\star/d$", r"train loss $\frac{1}{n}\sum_i\log(1+e^{-y_ix_i^\top U})$", r"test accuracy $P(\mathrm{sign}\,x^\top U=\mathrm{sign}\,x^\top\beta^\star)$"]
for j in range(4):
    for al, o in sw.items():
        ax[0, j].plot(o["fine"], o["th"][:, j], color=C[al], label=rf"theory, $\alpha=n/d={al}$")
        ls = sorted(o["S"]); m = np.array([o["S"][l][:, j].mean() for l in ls]); e = np.array([o["S"][l][:, j].std(ddof=1) / np.sqrt(len(o["S"][l])) for l in ls])
        ax[0, j].errorbar(ls, m, e, fmt="o", color=C[al], ms=4, capsize=2, label=rf"simulation $d=1000$, 8 seeds")
    ax[0, j].set_xlabel(r"penalty strength $\lambda$"); ax[0, j].set_ylabel(labs[j])
ax[0, 0].legend(fontsize=7); ax[0, 0].set_title(r"penalty kept on: $U=\arg\min\ \sum_i\log(1+e^{-y_ix_i^\top\beta})+\lambda\|\beta\|_1$", loc="left", fontsize=9)
def band(a, ts, TH, SI, key, lab, n1=None):
    (m, e), (s, se) = st(TH, key), st(SI, key); a.fill_between(ts, m - e, m + e, color="#1f77b4", alpha=0.3); a.plot(ts, m, color="#1f77b4", label="theory (mean ± s.e.)")
    idx = np.arange(0, len(ts), max(len(ts) // 14, 1)); a.errorbar(ts[idx], s[idx], se[idx], fmt="o", color="k", ms=3, capsize=2, label="simulation $d=3000$ (mean ± s.e.)")
    a.set_ylabel(lab)
    if n1 is not None: a.axvline(ts[n1], color="grey", ls=":", lw=1)
# row 2: no penalty, pretraining dynamics
for j, k in enumerate((1, 0.01)):
    o = pickle.load(open(f"nl_{k:g}.pkl", "rb")); ts = o["ts"][:-1]
    band(ax[1, 2 * j], ts, o["TH"], o["SI"], "Q", r"$\|\beta(t)\|^2/d$"); band(ax[1, 2 * j + 1], ts, o["TH"], o["SI"], "loss", r"train loss $\frac{1}{n}\sum_i\log(1+e^{-y_ix_i^\top\beta(t)})$")
    for a in ax[1, 2 * j:2 * j + 2]: a.set_xlabel("training time $t$"); a.set_title(rf"no penalty, pretraining, $k={k:g}$, $\alpha=0.3$", loc="left", fontsize=9)
ax[1, 0].legend(fontsize=7)
# row 3: two stages
for j, (name, tit) in enumerate((("nl2_npo_0.01_1", r"NPO on half the samples + retain term, $k=0.01$"), ("nl2_flip_0.01_0", r"flipped labels on half the samples, $k=0.01$"))):
    o = pickle.load(open(name + ".pkl", "rb")); T = len(o["TH"][0]["Q"]); ts = o["dt"] * np.arange(T)
    band(ax[2, 2 * j], ts, o["TH"], o["SI"], "accF", r"accuracy on forget samples $\frac{1}{|F|}\sum_{i\in F}1[y_ix_i^\top\beta(t)>0]$", o["n1"]); band(ax[2, 2 * j + 1], ts, o["TH"], o["SI"], "lossR", r"loss on retained samples $\frac{1}{|D\setminus F|}\sum_{i\notin F}\log(1+e^{-y_ix_i^\top\beta(t)})$", o["n1"])
    for a in ax[2, 2 * j:2 * j + 2]: a.set_xlabel("training time $t$ (dotted line: stage 2 starts)"); a.set_title(tit, loc="left", fontsize=9)
plt.tight_layout(); plt.savefig("fig_hard_labels.png", dpi=140)
