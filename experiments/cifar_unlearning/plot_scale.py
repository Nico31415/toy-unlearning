"""Accuracy against the rescaling factor S: one figure per output root made by an S sweep (rounds/rs_*.sh).

Reads unlearn.csv (aggregate.py). Rows (R = kept classes 8-way, F = forgotten classes 2-way) x columns (after unlearning,
after relearning on the same rows); x = S (log), one line per n and per BN treatment (dashed: stale BatchNorm running
statistics, solid: running statistics scaled exactly, --rs_scale_bn_stats). The unscaled run is S = 1 on both lines.
Also writes scale_table.csv.   python plot_scale.py --src <csv dir> --dest <figure dir> [--loss mse]"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

TAG = re.compile(r"^(?:all|weights)-S(?P<S>[\d.]+?)(?P<bn>-bnrecal)?(?P<stat>-bnstat)?$")


def parse_tag(tag: str):
    """(S, bn treatment) of a rescale tag; 'none' is S = 1 with the original BatchNorm statistics."""
    if tag == "none":
        return 1.0, "stale"
    m = TAG.match(tag)
    if not m:
        return None
    return float(m["S"]), "scaled" if m["stat"] else "stale"


def table(unlearn: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in unlearn[unlearn["theory"].astype(bool)].iterrows():
        parsed = parse_tag(str(r["rescale"]))
        if parsed:
            rows.append(dict(S=parsed[0], bn=parsed[1], mu_f=r["mu_f"], n=r["n_rows"], seed=r["seed"],
                             **{k: r[k] for k in ("ul_A_r8", "ul_A_f2", "rl_A_r8", "rl_A_f2", "ul_steps", "ul_status")
                                if k in r}))
    return pd.DataFrame(rows).sort_values(["bn", "n", "S"]) if rows else pd.DataFrame()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--loss", default="")
    args = p.parse_args(argv)
    path = Path(args.src) / "unlearn.csv"
    t = table(pd.read_csv(path)) if path.exists() else pd.DataFrame()
    if t.empty or t["S"].nunique() < 2:
        print("plot_scale: no S sweep in", args.src)
        return
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    t.to_csv(dest / "scale_table.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharey=True)
    cols = {n: c for n, c in zip(sorted(t["n"].unique()), ["#3b528b", "#5ec962", "#fde725", "#440154"])}
    for j, stage in enumerate(("ul", "rl")):
        for i, (m, label) in enumerate((("A_r8", "R (8-way)"), ("A_f2", "F (2-way)"))):
            ax = axes[i, j]
            for (n, bn), g in t.groupby(["n", "bn"]):
                g = g.groupby("S")[f"{stage}_{m}"].mean().reset_index()
                ax.plot(g["S"], g[f"{stage}_{m}"], marker="o", ms=4, color=cols[n], ls="-" if bn == "scaled" else "--")
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, 1.05)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"Accuracy, {label}")
            if i == 1:
                ax.set_xlabel("S")
        axes[0, j].set_title(("unlearning", "relearning ($n_{UL}$ = $n_{RL}$)")[j], style="italic", fontsize=11)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in cols.values()]
    labels = [f"n = {int(n)}" for n in cols]
    handles += [plt.Line2D([], [], color="k", ls="--"), plt.Line2D([], [], color="k", ls="-")]
    labels += ["BN stats unscaled", "BN stats scaled (exact)"]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(f"{args.loss}: rescaling Pretrained by S before unlearning (seed mean, mu_f = {t['mu_f'].iloc[0]:g})", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(dest / "scale_sweep.png", dpi=150)
    print("wrote", dest / "scale_sweep.png")


if __name__ == "__main__":
    main()
