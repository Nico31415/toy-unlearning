"""Figures for the modular-arithmetic runs, from aggregate.py's tables.

    python experiments/modular_unlearning/plot.py --src <dir with the CSVs> --dest <dir>

    family.png          forget-task accuracy of A(L) against L per experiment; realised overlap in the legend
    phase_<cfg>.png     distance classes and has-B regions on axes in units of n_learn (as for CIFAR)
    hidden.png          per stage-2 configuration: surviving Fourier amplitude, phase preservation and residual
                        decodability rho (mlp tap, answer probe and Fourier R^2) against n_S
    relearn.png         relearning steps against surviving amplitude and against forget accuracy (Spearman)
    thresholds.json     n_learn, eta^2 and the has-B threshold per experiment
Has B: forget accuracy above chance + 0.1 (A(D)'s accuracy - chance), chance = 1 / p_f. eta^2 = max(0.15, 2 x seed floor).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EPS = 0.1
DIST = {0: ("meets both", "#8DB8E3"), 1: ("unlearns only", "#F2B07E"), 2: ("untrains only", "#9ED29A"),
        3: ("neither", "#BDBDBD")}
CAPS = {0: ("U1", "#DADADA"), 1: ("U2 under-removed", "#F3E08A"), 2: ("U3 both", "#A9C4EC"),
        3: ("U4 untrains only", "#A8D8A0"), 4: ("U5 unlearns only", "#F5BE8E")}


def thresholds(fam, noise):
    out = {}
    for exp, g in fam.groupby("exp"):
        p_f = int(exp.split("-")[0][1:])
        acc = g.groupby("L")["acc_forget"].mean().sort_index()
        chance, top = 1.0 / p_f, acc.iloc[-1]
        learned = acc[acc >= chance + (1 - EPS) * (top - chance)]
        floor = noise[noise.exp == exp].groupby("L")["d2"].mean().max() if len(noise) and (noise.exp == exp).any() else 0.0
        out[exp] = {"n_learn": int(max(learned.index.min() if len(learned) else acc.index.max(), 1)),
                    "has_thr": float(chance + EPS * (top - chance)), "eta2": float(max(0.15, 2 * (floor or 0))),
                    "overlap": float(g[g.L == g.L.max()]["overlap"].mean())}
    return out


def name(r):
    return f"{r['method']}_{r['target']}_{r['arm']}" + (f"_S{r['S']:g}" if r["S"] != 1 else "")


def add_rho(ph, fam):
    ref = fam.set_index(["exp", "seed", "L"])
    for col in [c for c in ph.columns if c.startswith(("probe_", "fourier_r2_")) and c in fam.columns]:
        ad = np.array([ref[col].get((e, s, L), np.nan) for e, s, L in zip(ph.exp, ph.seed, ph.L_full)], float)
        a0 = np.array([ref[col].get((e, s, 0), np.nan) for e, s in zip(ph.exp, ph.seed)], float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ph[f"rho_{col}"] = (ph[col].to_numpy(float) - a0) / (ad - a0)
    accs = fam.set_index(["exp", "seed", "L"])["acc_forget"]
    ph["acc_ad"] = [accs.get((e, s, L), np.nan) for e, s, L in zip(ph.exp, ph.seed, ph.L_full)]
    ph["acc_ut"] = [accs.get((e, s, L), np.nan) for e, s, L in zip(ph.exp, ph.seed, ph.L_left)]
    return ph


def maps(ph, th, dest):
    keys = ["exp", "method", "target", "arm", "S"]
    for cfg, g in ph.groupby(keys):
        t = th[cfg[0]]
        m = g.groupby(["L_full", "n_S"]).mean(numeric_only=True).reset_index()
        ut, ul = m.d_ut2 <= t["eta2"], m.d_ul2 <= t["eta2"]
        dist = np.where(ut & ul, 0, np.where(ul, 1, np.where(ut, 2, 3)))
        ha, hr, hu = m.acc_ad > t["has_thr"], m.acc_ut > t["has_thr"], m.acc_forget > t["has_thr"]
        caps = np.where(~ha, 0, np.where(~hr & hu, 1, np.where(~hr & ~hu, 2, np.where(hu, 3, 4))))
        x, y = m.n_S / t["n_learn"], m.L_full / t["n_learn"]
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4))
        for ax, cls, names, title in ((axs[0], dist, DIST, f"within eta (eta^2={t['eta2']:.2f})"),
                                      (axs[1], caps, CAPS, "has B")):
            for c, (lab, col) in names.items():
                sel = cls == c
                if sel.any():
                    ax.scatter(x[sel], y[sel], s=150, marker="s", color=col, edgecolor="k", lw=0.4, label=lab)
            if cls is dist and "d_noop_ut2" in m.columns:
                # cells where doing nothing already counts as untraining: A(D) is within eta of A(D minus S)
                triv = (m.d_noop_ut2 <= t["eta2"]).to_numpy()
                if triv.any():
                    ax.scatter(x[triv], y[triv], s=40, marker="x", color="k", lw=1.0,
                               label="A(D) already within eta of A(D minus S)")
            lim = float(max(x.max(), y.max()) * 1.1)
            ax.plot([0, lim], [0, lim], "k", lw=0.8)
            ax.set_xlabel("|S| / n_learn")
            ax.set_ylabel("|S_full| / n_learn")
            ax.set_title(title, loc="left", fontsize=10)
            ax.legend(fontsize=7)
        fig.suptitle(f"{cfg[0]} {name(dict(zip(keys, cfg)))} (n_learn {t['n_learn']}, overlap {t['overlap']:.2f})",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(dest / f"phase_{cfg[0]}_{name(dict(zip(keys, cfg)))}.png", dpi=130)
        plt.close(fig)


def hidden(ph, dest):
    cols = [c for c in ("amp", "phase", "rho_probe_mlp0", "rho_fourier_r2_mlp0") if c in ph.columns]
    fig, axs = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.4))
    for cfg, g in ph.groupby(["exp", "method", "target", "arm", "S"]):
        m = g.groupby("n_S")[cols].mean()
        for ax, c in zip(np.atleast_1d(axs), cols):
            ax.plot(m.index, m[c], marker="o", label=f"{cfg[0]} {name(dict(zip(['exp', 'method', 'target', 'arm', 'S'], cfg)))}")
            ax.set_title(c, loc="left", fontsize=10)
            ax.set_xscale("log")
            ax.set_xlabel("n_S")
    np.atleast_1d(axs)[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(dest / "hidden.png", dpi=130)
    plt.close(fig)


def relearn_plot(ph, rl, dest):
    if not len(rl):
        return
    rows = []
    for d, g in rl[rl.source.str.startswith("phase/")].groupby("source"):
        p = ph[ph.dir == d]
        if len(p):
            ok = g.dropna(subset=["steps_to_acc"])
            rows.append({"cost": ok.sort_values("n_RL").steps_to_acc.iloc[-1] if len(ok) else np.nan,
                         "amp": p.amp.iloc[0], "acc_forget": p.acc_forget.iloc[0]})
    df = pd.DataFrame(rows).dropna()
    if not len(df):
        return
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    for ax, c in zip(axs, ("amp", "acc_forget")):
        ax.scatter(df[c], df.cost, s=16)
        ax.set_title(f"Spearman {df[[c, 'cost']].corr('spearman').iloc[0, 1]:.2f}", loc="left", fontsize=10)
        ax.set_xlabel(c)
    axs[0].set_ylabel("relearning steps (largest attack)")
    fig.tight_layout()
    fig.savefig(dest / "relearn.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    src, dest = Path(a.src), Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    read = lambda n: pd.read_csv(src / f"{n}.csv") if (src / f"{n}.csv").exists() else pd.DataFrame()  # noqa: E731
    fam, ph, noise, rl = read("family"), read("phase"), read("noise"), read("phase_relearn")
    if not len(fam):
        print("no family.csv")
        return
    th = thresholds(fam, noise)
    (dest / "thresholds.json").write_text(json.dumps(th, indent=1))
    fig, ax = plt.subplots(figsize=(6, 4))
    for exp, g in fam.groupby("exp"):
        m = g.groupby("L")["acc_forget"].mean()
        ax.plot(np.maximum(m.index, 1), m, marker="o", label=f"{exp} (overlap {th[exp]['overlap']:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("forget-task pairs in pretraining, L")
    ax.set_ylabel("forget-task test accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest / "family.png", dpi=130)
    plt.close(fig)
    if len(ph):
        ph = add_rho(ph, fam)
        ph = ph[ph.exp.isin(th)]
        maps(ph, th, dest)
        hidden(ph, dest)
        relearn_plot(ph, rl, dest)
        ph.to_csv(src / "phase_rho.csv", index=False)


if __name__ == "__main__":
    main()
