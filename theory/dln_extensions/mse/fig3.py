import pickle, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
o = pickle.load(open("summary.pkl", "rb")); C = {"B": "#2a78d6", "C": "#eb6834", "D": "#1baf7a", "A": "#6250d6"}; LB = {"B": "(1, 0, 1)", "C": "(1, 0.4, 1)", "D": "(1, 1, 1)", "A": "(0, 1, 0)"}
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5)); f = lambda cn, dn, k: o[(cn, dn)]["loss"]["forget to zero"]["end"][k]
a = ax[0]; ff = [0.1, 0.5, 0.9]; cn = ["ffrac=0.1", "base", "ffrac=0.9"]
for dn in "BCDA": a.plot(ff, [f(c, dn, "m3") / f(c, dn, "m2") for c in cn], "o", mfc="white", color=C[dn], ms=5, label=f"simulation, blocks {LB[dn]}")
a.plot(ff, [0.13, 0.55, 0.77], color=C["C"], lw=1.4, label="linearised with metric $u_j^2+v_j^2$, blocks (1, 0.4, 1)"); a.plot(ff, [0.03, 0.16, 0.33], color="k", lw=1.0, ls="--", label="flat metric (lazy), blocks (1, 0.4, 1)")
a.set_xlabel(r"forget fraction $|F|/|F_{\rm full}|$"); a.set_ylabel(r"$\frac{{\rm MSE\ on\ }F_{\rm full}\setminus F}{{\rm MSE\ on\ }F}$ at the end of finetuning"); a.set_ylim(0, 1); a.legend(frameon=False, fontsize=6.5); a.set_title("forget-to-zero: damage to left-over forget samples", fontsize=8.5)
a = ax[1]; Sn = ["S=0.1", "S=0.3", "base", "S=3"]
for dn in "BCDA": a.plot(range(4), [f(c, dn, "m3") / f(c, dn, "m2") for c in Sn], "o-", mfc="white", color=C[dn], ms=5, lw=1, label=LB[dn])
a.set_xticks(range(4)); a.set_xticklabels(["0.1", "0.3", "1", "3"]); a.set_xlabel("rescaling $S$ (categories, not to scale)"); a.set_ylabel(r"$\frac{{\rm MSE\ on\ }F_{\rm full}\setminus F}{{\rm MSE\ on\ }F}$ at the end of finetuning"); a.set_ylim(0, 1.05); a.legend(frameon=False, fontsize=7, title="blocks", title_fontsize=7); a.set_title("forget-to-zero: effect of rescaling", fontsize=8.5)
a = ax[2]
for dn in "BCDA": a.plot(range(4), [o[(c, dn)]["loss"]["forget to zero"]["dl_end"] for c in Sn], "o-", mfc="white", color=C[dn], ms=5, lw=1, label="to retain-only model, " + LB[dn])
for dn in "BCDA": a.plot(range(4), [o[(c, dn)]["loss"]["forget to zero"]["du_end"] for c in Sn], "s--", mfc="white", color=C[dn], ms=4, lw=0.8)
a.plot([], [], "s--", color="k", mfc="white", ms=4, lw=0.8, label="to model retrained without $F$ (same colours)")
a.set_xticks(range(4)); a.set_xticklabels(["0.1", "0.3", "1", "3"]); a.set_xlabel("rescaling $S$ (categories, not to scale)"); a.set_ylabel(r"$\|\beta-\beta_{\rm ref}\|^2/d$ at the end of finetuning"); a.set_ylim(0, 0.5); a.legend(frameon=False, fontsize=6.5); a.set_title("forget-to-zero: distance to the two references", fontsize=8.5)
fig.tight_layout(); fig.savefig("figs/summary.png", dpi=150)
