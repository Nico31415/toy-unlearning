import numpy as np, pickle, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}); fig, ax = plt.subplots(1, 4, figsize=(17, 4.3))
ms = lambda V: (np.mean(V, 0), np.std(V, 0, ddof=1) / np.sqrt(len(V)))
for (k, phi, col, lab) in ((1, 0.1, "#1f77b4", r"$k=1$, batch fraction $0.1$"), (1, 1, "#7f7f7f", r"$k=1$, full batch"), (0.01, 0.1, "#d62728", r"$k=0.01$, batch fraction $0.1$")):
    o = pickle.load(open(f"sgd_{k:g}_0.04_{phi:g}.pkl", "rb"))
    for a, key in ((ax[0], "err"), (ax[1], "kap")):
        (m, e), (s, se) = ms([r[key] for r in o["TH"]]), ms([r[key] for r in o["SI"]]); t = np.arange(len(m)); idx = np.arange(5, len(m), len(m) // 10)
        a.plot(t, m, color=col, label=lab + " (line: theory)"); a.fill_between(t, m - e, m + e, color=col, alpha=0.2); a.errorbar(t[idx], s[idx], se[idx], fmt="o", color=col, ms=3.5, capsize=2)
ax[0].set_ylabel(r"$\|\beta_t-\beta^\star\|^2/\|\beta^\star\|^2$"); ax[1].set_ylabel(r"mean over coordinates of $\log\kappa_j=\log(2u_jv_j/c)$")
for a in ax[:2]: a.set_xlabel(r"SGD step (step size $\gamma=0.04$)"); a.set_title("mini-batch SGD; markers: simulation $d=3000$", loc="left", fontsize=9)
ax[0].legend(fontsize=7)
o = pickle.load(open("es_cov_0.01.pkl", "rb")); i1 = o["i1"]; cols = ["#1f77b4", "#d62728", "#2ca02c", "#17becf", "#ff7f0e"]
for (cfg, r), col in zip(o["out"].items(), cols):
    (m, e), (s, se) = ms(r["th"][:, 0]), ms(r["si"][:, 0]); t = o["dt"] * (np.arange(len(m)) - i1); sl = slice(i1, len(m)); idx = np.arange(i1, len(m), 10)
    ax[2].plot(t[sl], m[sl], color=col, label=rf"target ${cfg[0]:+g}\,y$" + (", anchor" if cfg[1] else "") + " (line: theory)"); ax[2].errorbar(t[idx], s[idx], se[idx], fmt="o", color=col, ms=3, capsize=2)
ax[2].set_xlabel("stage-2 budget $t-T_1$"); ax[2].set_ylabel(r"$\|\beta(t)-A(D\setminus F)\|^2/\|\beta^\star\|^2$"); ax[2].legend(fontsize=7); ax[2].set_title("squared loss, $k=0.01$: which loss at which budget", loc="left", fontsize=9)
for k, col in ((1, "#1f77b4"), (0.01, "#d62728")):
    o = pickle.load(open(f"whp_{k:g}_1_1.pkl", "rb")); (m, e), (s, se) = ms([r["tv"][:, 0] for r in o["TH"]]), ms([r["tv"][:, 0] for r in o["SI"]])
    ax[3].plot(o["lams"], m, color=col, label=rf"$k={k:g}$ (line: theory)"); ax[3].fill_between(o["lams"], m - e, m + e, color=col, alpha=0.2); ax[3].errorbar(o["lams"], s, se, fmt="o", color=col, ms=3.5, capsize=2)
ax[3].set_xlabel(r"task-vector coefficient $\lambda$ in $w=w_{\rm pre}-\lambda\,(w_{\rm reinforced}-w_{\rm pre})$"); ax[3].set_ylabel(r"$\cos(\beta_\lambda,A(D\setminus F))$"); ax[3].legend(fontsize=7); ax[3].set_title("hard labels: task vector; markers: simulation", loc="left", fontsize=9)
plt.tight_layout(); plt.savefig("fig_final.png", dpi=140)
