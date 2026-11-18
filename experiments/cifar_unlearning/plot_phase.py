"""Figures for the phase experiments, from the tables of aggregate_phase.py.

    python experiments/cifar_unlearning/plot_phase.py --src <dir with the CSVs> --dest <dir>

    family.png             forget accuracy and penultimate probe accuracy of A(L) against L; n_learn marked
    phase_<cfg>.png        per stage-2 configuration (method, target, arm, S): the distance classes (within eta of
                           A(D minus S) and/or A(D minus S_full)) and the has-B regions U1-U5, on axes in units of n_learn
    rho_layers.png         per layer, mean over cells, one line per configuration: residual decodability rho (fresh,
                           scale-blind probe) and residual magnitude rho_amp (along A(D)'s probe direction)
    rho_vs_relearn.png     relearning cost against penultimate rho and against output forget accuracy (Spearman)
    thresholds.json        n_learn, eta^2 and the has-B accuracy threshold per experiment
    mix_untraining.png     E9 (mix-C-ALPHA): untraining on S's own images, membership AUC and kept-class accuracy
                           against ALPHA

Crosses on the distance panel: cells where A(D) itself is already within eta of A(D minus S) (d_noop_ut2), so
"untrains" there is met by doing nothing. Definitions. Has B: forget accuracy (10-way, held-out) above chance + eps * (A(D)'s accuracy - chance), eps = 0.1;
learns B: above chance + 0.9 * (that margin). n_learn: the smallest L whose seed-mean accuracy learns B.
eta^2 per cell = max(0.15, 2 * the seed-noise floor of A(L_full) in noise.csv) (thresholds.json: eta2_by_L);
n_onset: the smallest L whose seed-mean accuracy has B. rho_l = (probe(U) - probe(A(0))) /
(probe(A(L_full)) - probe(A(0))), with the references of the same seed.
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

DIST = {0: ("meets both", "#8DB8E3"), 1: ("unlearns only", "#F2B07E"), 2: ("untrains only", "#9ED29A"),
        3: ("neither", "#BDBDBD")}
CAPS = {0: ("U1 A(D) never had B", "#DADADA"), 1: ("U2 under-removed", "#F3E08A"), 2: ("U3 both goals met", "#A9C4EC"),
        3: ("U4 untrains, not unlearns", "#A8D8A0"), 4: ("U5 unlearns, over-removes", "#F5BE8E")}
EPS = 0.1
ETA2_MIN = 0.15


def thresholds(family: pd.DataFrame, noise: pd.DataFrame, num_classes: int = 10):
    out = {}
    for exp, g in family.groupby("exp"):
        acc = g.groupby("L")["A_f10"].mean().sort_index()
        chance = 0.0  # 10-way accuracy of a model that never predicts the forget classes
        top = acc.iloc[-1]
        learn_thr = chance + (1 - EPS) * (top - chance)
        has_thr = chance + EPS * (top - chance)
        learned = acc[acc >= learn_thr]
        n_learn = int(learned.index.min()) if len(learned) else int(acc.index.max())
        onset = acc[acc > has_thr]
        # Seed-noise floor per family member. Distances of a stage-2 cell are normalised by ||A(L_full) - A(0)||^2,
        # so the floor that applies to a cell is the one at L = L_full. (One maximum over all L is wrong: at small L
        # the normaliser is tiny and the ratio is about 1 whatever happens, which made eta^2 = 2 in ph_e0.)
        nz = noise[noise.exp == exp].groupby("L")["d2"].mean().dropna() if len(noise) and (noise.exp == exp).any() \
            else pd.Series(dtype=float)
        out[exp] = {"n_learn": max(n_learn, 1), "n_onset": int(onset.index.min()) if len(onset) else None,
                    "has_thr": float(has_thr), "eta2": ETA2_MIN,
                    "eta2_by_L": {str(int(L)): float(max(ETA2_MIN, 2 * v)) for L, v in nz.items()},
                    "noise_by_L": {str(int(L)): float(v) for L, v in nz.items()}}
    return out


def eta2_for(t: dict, L_full) -> np.ndarray:
    """eta^2 of each cell: 2 x the seed-noise floor of A(L_full) (at least ETA2_MIN); ETA2_MIN where unmeasured."""
    by = t.get("eta2_by_L", {})
    return np.array([by.get(str(int(L)), t["eta2"]) for L in np.atleast_1d(L_full)], dtype=float)


def add_rho(phase: pd.DataFrame, family: pd.DataFrame) -> pd.DataFrame:
    layers = [c[len("probe_"):] for c in phase.columns if c.startswith("probe_")]
    ref = family.set_index(["exp", "seed", "L"])
    for k in layers:
        col = f"probe_{k}"
        if col not in family.columns:
            continue
        ad = [ref[col].get((e, s, L), np.nan) for e, s, L in zip(phase.exp, phase.seed, phase.L_full)]
        a0 = [ref[col].get((e, s, 0), np.nan) for e, s in zip(phase.exp, phase.seed)]
        ad, a0 = np.asarray(ad, float), np.asarray(a0, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            phase[f"rho_{k}"] = (phase[col].to_numpy(float) - a0) / (ad - a0)
    return phase, layers


def config_name(row) -> str:
    s = f"{row['method']}_{row['target']}_{row['arm']}"
    return s + (f"_S{row['S']:g}" if row["S"] != 1 else "")


def phase_maps(phase: pd.DataFrame, th, dest: Path) -> None:
    keys = ["exp", "method", "target", "arm", "S"]
    for cfg, g in phase.groupby(keys):
        exp = cfg[0]
        t = th[exp]
        m = g.groupby(["L_full", "n_S"]).agg(d_ut2=("d_ut2", "mean"), d_ul2=("d_ul2", "mean"),
                                              A_f10=("A_f10", "mean"), ad=("ref_A_f10_ad", "mean"),
                                              ut=("ref_A_f10_ut", "mean"),
                                              **({"d_noop_ut2": ("d_noop_ut2", "mean")} if "d_noop_ut2" in g else {})
                                              ).reset_index()
        e2 = eta2_for(t, m.L_full)
        ut, ul = m.d_ut2.to_numpy() <= e2, m.d_ul2.to_numpy() <= e2
        dist = np.where(ut & ul, 0, np.where(ul, 1, np.where(ut, 2, 3)))
        has_a, has_r, has_u = m.ad > t["has_thr"], m.ut > t["has_thr"], m.A_f10 > t["has_thr"]
        caps = np.where(~has_a, 0, np.where(~has_r & has_u, 1, np.where(~has_r & ~has_u, 2,
                                                                        np.where(has_u, 3, 4))))
        x, y = m.n_S / t["n_learn"], m.L_full / t["n_learn"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
        for ax, cls, names, title in ((axs[0], dist, DIST, "within eta (eta^2 = 2 x seed noise at L_full, >= 0.15)"),
                                      (axs[1], caps, CAPS, "has B (forget accuracy)")):
            for c, (label, colour) in names.items():
                sel = cls == c
                if sel.any():
                    ax.scatter(x[sel], y[sel], s=160, marker="s", color=colour, edgecolor="k", lw=0.4, label=label)
            if cls is dist and "d_noop_ut2" in m.columns:
                # cells where doing nothing already counts as untraining: A(D) is within eta of A(D minus S)
                triv = m.d_noop_ut2.to_numpy() <= e2
                if triv.any():
                    ax.scatter(x[triv], y[triv], s=40, marker="x", color="k", lw=1.0,
                               label="A(D) already within eta of A(D minus S)")
            lim = max(x.max(), y.max()) * 1.1
            ax.plot([0, lim], [0, lim], "k", lw=0.8)
            ax.set_xscale("symlog", linthresh=0.01)
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_xlabel("|S| / n_learn")
            ax.set_ylabel("|S_full| / n_learn")
            ax.set_title(title, loc="left", fontsize=10)
            ax.legend(fontsize=7, loc="lower right")
        fig.suptitle(f"{exp}: {config_name(dict(zip(keys, cfg)))}  (n_learn = {t['n_learn']})", fontsize=10)
        fig.tight_layout()
        fig.savefig(dest / f"phase_{exp}_{config_name(dict(zip(keys, cfg)))}.png", dpi=130)
        plt.close(fig)


def rho_plots(phase: pd.DataFrame, layers, relearn: pd.DataFrame, family: pd.DataFrame, dest: Path) -> None:
    rho_cols = [f"rho_{k}" for k in layers if f"rho_{k}" in phase.columns]
    if not rho_cols:
        return
    amp_cols = [f"rho_amp_{k}" for k in layers if f"rho_amp_{k}" in phase.columns]
    panels = [(rho_cols, "residual decodability rho (fresh probe)")] + \
        ([(amp_cols, "residual magnitude rho_amp (A(D)'s probe direction)")] if amp_cols else [])
    fig, axs = plt.subplots(1, len(panels), figsize=(7 * len(panels), 4), squeeze=False)
    for ax, (cols, lab) in zip(axs[0], panels):
        for cfg, g in phase.groupby(["exp", "method", "target", "arm", "S"]):
            ax.plot(range(len(cols)), g[cols].mean().to_numpy(), marker="o",
                    label=f"{cfg[0]} {config_name(dict(zip(['exp', 'method', 'target', 'arm', 'S'], cfg)))}")
        ax.axhline(1, color="#888", ls=":")
        ax.axhline(0, color="#888", ls=":")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.split("_")[-1] for c in cols])
        ax.set_ylabel(lab)
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(dest / "rho_layers.png", dpi=130)
    plt.close(fig)
    if relearn is None or not len(relearn) or "dir" not in phase.columns:
        return
    # relearning cost: the smallest n_RL whose relearned forget accuracy is within 0.05 of A(D)'s
    rows = []
    for d, g in relearn[relearn.source.str.startswith("phase/")].groupby("source"):
        p = phase[phase.dir == d]
        if not len(p):
            continue
        target = p.ref_A_f10_ad.iloc[0] - 0.05
        ok = g[g.A_f10_after >= target].n_RL
        rows.append({"n_cost": ok.min() if len(ok) else 2 * g.n_RL.max(), "rho": p[rho_cols[-1]].iloc[0],
                     "rho_amp": p[amp_cols[-1]].iloc[0] if amp_cols else np.nan, "A_f10": p.A_f10.iloc[0]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    cols = [("rho", f"penultimate rho ({rho_cols[-1][4:]})")] + \
        ([("rho_amp", "penultimate rho_amp")] if amp_cols else []) + [("A_f10", "output forget accuracy")]
    fig, axs = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 3.8), sharey=True)
    for ax, (col, lab) in zip(axs, cols):
        ax.scatter(df[col], df.n_cost, s=18)
        r = df[[col, "n_cost"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"Spearman {r:.2f}", loc="left", fontsize=10)
        ax.set_xlabel(lab)
        ax.set_yscale("log")
    axs[0].set_ylabel("attack images to relearn")
    fig.tight_layout()
    fig.savefig(dest / "rho_vs_relearn.png", dpi=130)
    plt.close(fig)


def mix_plot(phase: pd.DataFrame, dest: Path) -> None:
    """E9 (mix-C-ALPHA experiments): untraining read-outs on S's own images against ALPHA, one column per |S|, one line
    per stage-2 configuration. r_S_ut: 0 = A(D minus S) on S's images, 1 = A(D); mia: membership AUC (dotted: the
    retrained model's AUC, the target); A_r: kept-class test accuracy."""
    mix = phase[phase.exp.str.startswith("mix-")].copy()
    if not len(mix):
        return
    mix["alpha"] = mix.exp.str.split("-").str[2].astype(float)
    rows = [c for c in ("r_S_ut_iid", "r_S_ut_cls", "mia_iid", "mia_cls", "A_r") if c in mix.columns]
    ns = sorted(mix.n_S.unique())
    fig, axs = plt.subplots(len(rows), len(ns), figsize=(3.6 * len(ns), 2.6 * len(rows)), squeeze=False, sharex=True)
    keys = ["method", "target", "arm", "S"]
    for j, n in enumerate(ns):
        g = mix[mix.n_S == n]
        for cfg, h in g.groupby(keys):
            m = h.groupby("alpha").mean(numeric_only=True)
            for i, col in enumerate(rows):
                axs[i, j].plot(m.index, m[col], marker="o", label=config_name(dict(zip(keys, cfg))))
        for i, col in enumerate(rows):
            ax = axs[i, j]
            ref = f"ref_ut_{col}" if col.startswith("mia") else None
            if ref and ref in g:
                r = g.groupby("alpha")[ref].mean()
                ax.plot(r.index, r.values, "k:", label="A(D minus S)")
            if col.startswith("r_S_ut"):
                ax.axhline(1, color="#888", ls=":")
                ax.axhline(0, color="#888", lw=0.6)
            ax.set_title(f"{col}, |S| = {n}", loc="left", fontsize=9)
            if i == len(rows) - 1:
                ax.set_xlabel("alpha (fraction of S from the removed class)")
    axs[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(dest / "mix_untraining.png", dpi=130)
    plt.close(fig)


def family_plot(family: pd.DataFrame, th, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    # per-layer probes in network order (the CSV keeps the order of the taps); the last is the deepest layer
    probe = [c for c in family.columns if c.startswith("probe_") and c not in ("probe_acc", "probe_ce", "probe_acc2")]
    for exp, g in family.groupby("exp"):
        m = g.groupby("L").mean(numeric_only=True)
        x = np.maximum(m.index.to_numpy(), 1)
        ax.plot(x, m.A_f10, marker="o", label=f"{exp}: forget accuracy")
        if probe:
            ax.plot(x, m[probe[-1]], marker="s", ls="--", label=f"{exp}: probe {probe[-1][6:]}")
        ax.axvline(th[exp]["n_learn"], color="#888", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("forget images in pretraining, L")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest / "family.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    src, dest = Path(args.src), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    read = lambda n: pd.read_csv(src / f"{n}.csv") if (src / f"{n}.csv").exists() else pd.DataFrame()  # noqa: E731
    family, phase, noise, relearn = read("family"), read("phase"), read("noise"), read("phase_relearn")
    if not len(family):
        print("no family.csv: nothing to plot")
        return
    th = thresholds(family, noise)
    (dest / "thresholds.json").write_text(json.dumps(th, indent=1))
    family_plot(family, th, dest)
    if len(phase):
        phase, layers = add_rho(phase, family)
        phase = phase[phase.exp.isin(th)]
        phase_maps(phase, th, dest)
        rho_plots(phase, layers, relearn, family, dest)
        mix_plot(phase, dest)
        phase.to_csv(src / "phase_rho.csv", index=False)


if __name__ == "__main__":
    main()
