"""Pilot figures from the aggregated CSVs (run aggregate.py first).

    python experiments/cifar_unlearning/plot.py [--loss ce] [--src ...] [--dest ...]

For every experiment, two figures, mean +/- std over seeds:
    after_unlearning_<exp>.png  metrics after unlearning against the data fraction f,
                                one line per mu_f, with Pretrained and Retrain as
                                dashed reference levels
    after_relearning_<exp>.png  metrics after relearning against n_f, one line per
                                mu_f plus Pretrained, Retrain and Scratch relearned
                                on the same images
Only runs without rescaling (rescale == "none") are plotted here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

UNLEARN_METRICS = {"class": ["A_r", "A_f10", "A_f2", "probe_acc"], "random": ["A_r", "probe_acc", "mia_auc"]}
RELEARN_METRICS = {"class": ["A_f2", "A_r"], "random": ["probe_acc", "A_r"]}
LABELS = {"A_r": "A_r (test, kept classes)", "A_f10": "A_f 10-way (T_f)", "A_f2": "A_f 2-way (T_f)",
          "probe_acc": "accuracy on probe P_f", "mia_auc": "MIA AUC (P_f vs test)"}
REF_STYLE = {"pretrained": ("tab:red", "Pretrained"), "retrain": ("black", "Retrain"),
             "scratch": ("tab:gray", "Scratch")}


def _band(ax, frame: pd.DataFrame, x: str, y: str, label: str, color=None, linestyle="-") -> None:
    stats = frame.groupby(x)[y].agg(["mean", "std"]).reset_index().sort_values(x)
    line, = ax.plot(stats[x], stats["mean"], marker="o", ms=3, label=label, color=color, linestyle=linestyle)
    ax.fill_between(stats[x], stats["mean"] - stats["std"].fillna(0), stats["mean"] + stats["std"].fillna(0),
                    color=line.get_color(), alpha=0.15, linewidth=0)


def plot_experiment(exp: str, unlearn: pd.DataFrame, refs: pd.DataFrame, dest: Path) -> List[Path]:
    kind = unlearn["kind"].iloc[0] if len(unlearn) else refs["kind"].iloc[0]
    written = []

    metrics = [m for m in UNLEARN_METRICS[kind] if f"ul_{m}" in unlearn.columns]
    if metrics and len(unlearn):
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.4), squeeze=False)
        for ax, m in zip(axes[0], metrics):
            for mu_f, group in unlearn.groupby("mu_f"):
                _band(ax, group, "fraction", f"ul_{m}", f"mu_f={mu_f:g}")
            for name in ("pretrained", "retrain"):
                col = f"before_{m}"
                if col in refs.columns and (refs["model"] == name).any():
                    level = refs[refs["model"] == name].groupby("seed")[col].first().mean()
                    color, label = REF_STYLE[name]
                    ax.axhline(level, color=color, linestyle="--", linewidth=1, label=label)
            ax.set_xscale("log")
            ax.set_xlabel("data fraction f")
            ax.set_title(LABELS.get(m, m), fontsize=10)
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"{exp}: after unlearning")
        fig.tight_layout()
        path = dest / f"after_unlearning_{exp}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    metrics = [m for m in RELEARN_METRICS[kind]
               if f"rl_{m}" in unlearn.columns or f"rl_{m}" in refs.columns]
    if metrics:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.4), squeeze=False)
        for ax, m in zip(axes[0], metrics):
            col = f"rl_{m}"
            if col in unlearn.columns:
                for mu_f, group in unlearn.groupby("mu_f"):
                    _band(ax, group, "n_f", col, f"unlearned, mu_f={mu_f:g}")
            if col in refs.columns:
                for name, group in refs.groupby("model"):
                    color, label = REF_STYLE.get(name, (None, name))
                    _band(ax, group, "n_f", col, label, color=color, linestyle="--")
            ax.set_xscale("log")
            ax.set_xlabel("relearning images n_f")
            ax.set_title(LABELS.get(m, m), fontsize=10)
        axes[0][0].legend(fontsize=8)
        fig.suptitle(f"{exp}: after relearning")
        fig.tight_layout()
        path = dest / f"after_relearning_{exp}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--src", default=None)
    parser.add_argument("--dest", default=None)
    args = parser.parse_args(argv)
    src = Path(args.src or f"data/processed/cifar_unlearning/{args.loss}")
    dest = Path(args.dest or f"figures/cifar_unlearning/{args.loss}")
    dest.mkdir(parents=True, exist_ok=True)

    def read(name: str) -> pd.DataFrame:
        path = src / f"{name}.csv"
        return pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()

    unlearn, refs = read("unlearn"), read("references")
    if len(unlearn):
        unlearn = unlearn[unlearn["rescale"] == "none"]
    exps = sorted(set(unlearn.get("exp", pd.Series(dtype=str))) | set(refs.get("exp", pd.Series(dtype=str))))
    for exp in exps:
        u = unlearn[unlearn["exp"] == exp] if len(unlearn) else unlearn
        r = refs[refs["exp"] == exp] if len(refs) else refs
        for path in plot_experiment(exp, u, r, dest):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
