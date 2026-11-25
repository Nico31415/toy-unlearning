"""Figures: one page per configuration x data setting. Top row: first part of finetuning, theory (lines) over simulation (markers). Bottom row: whole finetuning, simulation, with the two references."""
import numpy as np, pickle, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mse_sim import LOSSES
from run import CONFIGS, DATA
COL = dict(zip(LOSSES, ["#8a8a8a", "#2a78d6", "#eb6834", "#1baf7a", "#0b6e4f", "#eda100", "#6250d6"]))
YL = {"m1": r"$\mathbb{E}_{x\sim P_R}\,(x^\top\beta-x^\top\beta^\star)^2$", "m2": r"$\frac{1}{|F|}\sum_{i\in F}(x_i^\top\beta-y_i)^2$",
      "m3": r"$\frac{1}{|F_{\rm full}\setminus F|}\sum_{i\in F_{\rm full}\setminus F}(x_i^\top\beta-y_i)^2$", "m4": r"$\mathbb{E}_{x\sim P_F}\,(x^\top\beta-x^\top\beta^\star)^2$"}
TT = {"m1": "retain distribution", "m2": "forget samples F", "m3": "left-over forget samples", "m4": "forget distribution"}
DN = {"C": r"$(d_r,d_{sh},d_f)\propto(1,0.4,1)$", "A": r"$(0,d_{sh},0)$", "B": r"$(1,0,1)$", "D": r"$(1,1,1)$"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.4})

def load(cn, dn):
    S = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"results/{cn}__{dn}__sim*.pkl"))]; T = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"results/{cn}__{dn}__th*.pkl"))]; return S, T

def ms(a): a = np.array(a, float); n = np.sum(~np.isnan(a), 0); return np.nanmean(a, 0), np.nanstd(a, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), n

def page(cn, dn, pdf=None, png=None):
    S, T = load(cn, dn)
    if not S: return
    cfg = S[0]["cfg"]; rec = cfg["rec"]; fig, ax = plt.subplots(2, 4, figsize=(13, 6.2)); tz = (T[0]["dt"] * (len(T[0]["ft"][LOSSES[0]]["m1"]) - 1)) if T else 6.0
    for j, key in enumerate(("m1", "m2", "m3", "m4")):
        for row in (0, 1):
            a = ax[row, j]; ymax = 0
            for li, l in enumerate(LOSSES):
                m, se, n = ms([s["ft"][key][:, li] for s in S]); m[n < len(S)] = np.nan; t = np.arange(len(m)) * rec
                if row == 0:
                    sel = t <= tz * 1.001; step = max(1, int(sel.sum() // 14)); a.errorbar(t[sel][::step], m[sel][::step], se[sel][::step], fmt="o", ms=3.2, mfc="white", mec=COL[l], ecolor=COL[l], elinewidth=0.6, mew=0.9, zorder=3)
                    if T:
                        mt, st, nt = ms([x["ft"][l][key] for x in T]); mt[nt < len(T)] = np.nan; tt = np.arange(len(mt)) * T[0]["dt"]; a.plot(tt, mt, color=COL[l], zorder=2); a.fill_between(tt, mt - st, mt + st, color=COL[l], alpha=0.15, lw=0)
                    v = m[sel]
                else:
                    a.plot(t, m, color=COL[l]); a.fill_between(t, m - se, m + se, color=COL[l], alpha=0.15, lw=0); v = m
                if l not in ("signed", "pure ascent") and np.any(~np.isnan(v)): ymax = max(ymax, np.nanmax(v))
            for nm, ls in (("ref_untrain", ":"), ("ref_unlearn", "--")):
                r = np.nanmean([s[nm + "_end"][key] for s in S])
                if not np.isnan(r): a.axhline(r, color="k", lw=0.9, ls=ls, zorder=1); ymax = max(ymax, r)
            a.set_ylim(-0.03 * ymax, 1.25 * ymax if ymax > 0 else 1); a.set_xlim(0, tz if row == 0 else t[-1]); a.set_xlabel("finetuning time $t$ (step size $\\times$ steps)"); a.set_ylabel(YL[key])
            if row == 0: a.set_title(TT[key], fontsize=9)
    h = [plt.Line2D([], [], color=COL[l], label=l) for l in LOSSES] + [plt.Line2D([], [], color="k", ls=":", lw=0.9, label="ideal untraining: retrained without $F$"), plt.Line2D([], [], color="k", ls="--", lw=0.9, label="ideal unlearning: trained on retain class only"),
         plt.Line2D([], [], color="k", marker="o", mfc="white", ls="", ms=4, label=f"top: simulation, {len(S)} seeds $\\pm$ s.e."), plt.Line2D([], [], color="k", label=f"top: theory, {len(T)} runs" if T else "top: no theory for this case"), plt.Line2D([], [], color="k", alpha=0.5, label="bottom: simulation, whole run")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=7.5); ov = {k: v for k, v in CONFIGS[cn].items() if k != "mu_signed"}
    fig.suptitle(f"{cn}   |   data {dn}: {DN[dn]}   |   changed from baseline: {ov if ov else 'nothing'}", fontsize=10); fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    if pdf: pdf.savefig(fig)
    if png: fig.savefig(png, dpi=130)
    plt.close(fig)

if __name__ == "__main__":
    sel = sys.argv[1].split(";") if len(sys.argv) > 1 else list(CONFIGS)
    with PdfPages("mse_dynamics.pdf") as pdf:
        for cn in sel:
            for dn in DATA: page(cn, dn, pdf=pdf, png=f"figs/{cn}__{dn}.png" if cn == "base" else None)
