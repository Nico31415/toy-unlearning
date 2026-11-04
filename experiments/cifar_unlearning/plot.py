"""Pilot figures from the aggregated CSVs (run aggregate.py first).

    python experiments/cifar_unlearning/plot.py [--loss ce] [--src ...] [--dest ...]

For every experiment, two figures, mean +/- std over seeds:
    after_unlearning_<exp>.png  metrics after unlearning against the data fraction f,
                                one line per mu_f, with Pretrained and Retrain as
                                dashed reference levels
    after_relearning_<exp>.png  metrics after relearning against n_f, one line per
                                mu_f plus Pretrained, Retrain and Scratch relearned
                                on the same images
A third figure per class-split experiment, sample_efficiency_<exp>.png, shows retain and forget
test error against the number of images used, for pretraining, unlearning and relearning.
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


def _pair_errors(pretrain: pd.DataFrame, pair) -> pd.DataFrame:
    """Test error on the kept and the forgotten classes of a pair, from per-class accuracies."""
    cols = [f"acc_c{k}" for k in range(10)]
    retain = pretrain[[c for k, c in enumerate(cols) if k not in pair]].mean(axis=1)
    out = pretrain.assign(err_retain=1 - retain)
    a, b = pair
    if f"pair_{a}_{b}" in pretrain.columns:  # 2-way: the two forgotten classes only
        out["err_forget2"] = 1 - (pretrain[f"pair_{a}_{b}"] + pretrain[f"pair_{b}_{a}"]) / 2
    return out


def plot_sample_efficiency(exp: str, pretrain: pd.DataFrame, unlearn: pd.DataFrame, refs: pd.DataFrame,
                           dest: Path) -> List[Path]:
    """Rows: one metric each, the same in every column. Columns: the three stages, each against the number
    of images it used. Mean +/- std over seeds.

    Rows are test error on the kept classes (10-way) and on the forgotten classes (2-way: the model only
    chooses between the two forgotten classes). In the relearning column a dotted line in each mu_f's colour is the unlearned model *before*
    relearning, i.e. the unlearning curve itself, and the solid line is after relearning."""
    if not exp.startswith("pair-"):
        return []
    pair = tuple(int(v) for v in exp[len("pair-"):].split("-"))
    if len(unlearn):
        # Only runs that kept the whole retain set: unlearning (and so relearning) then differs by n_f alone.
        unlearn = unlearn[unlearn["n_r"] == unlearn["n_r"].max()]
        if "retain_matched" in unlearn.columns:
            unlearn = unlearn[~unlearn["retain_matched"].astype(bool)]
    rows = [("kept classes, 10-way", "A_r", "err_retain"), ("forgotten classes, 2-way", "A_f2", "err_forget2")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey="row", squeeze=False)
    titles = ["pretraining (from scratch)", "unlearning (from Pretrained)", "relearning (forget images only)"]
    xlabels = ["images in training set", "forget images used (retain set: all)", "forget images used"]
    for j, (title, xlabel) in enumerate(zip(titles, xlabels)):
        axes[0][j].set_title(title, fontsize=10)
        axes[1][j].set_xlabel(xlabel)
    colors = {}
    for row, (name, metric, key) in enumerate(rows):
        axes[row][0].set_ylabel(f"test error, {name}")
        pre, ul, rl = axes[row][0], axes[row][1], axes[row][2]
        frame = _pair_errors(pretrain, pair) if len(pretrain) else pretrain
        if key in frame.columns:
            _band(pre, frame, "n_train", key, "Pretrain on subset", color="tab:blue")
        else:
            pre.text(0.5, 0.5, "n/a: rerun pretrain_curve\n(old results lack logits)", ha="center", va="center",
                     transform=pre.transAxes, fontsize=8, color="gray")
        if len(unlearn):
            u = unlearn.assign(err_ul=1 - unlearn[f"ul_{metric}"], err_rl=1 - unlearn[f"rl_{metric}"])
            for mu_f, group in u.groupby("mu_f"):
                color = colors.setdefault(mu_f, f"C{len(colors)}")
                _band(ul, group, "n_f", "err_ul", f"mu_f={mu_f:g}", color=color)
                stats = group.groupby("n_f")["err_ul"].mean().sort_index()
                rl.plot(stats.index, stats.values, color=color, linestyle=":", linewidth=1.3,
                        label="before relearning" if mu_f == min(colors) else None)
                _band(rl, group, "n_f", "err_rl", f"unlearned, mu_f={mu_f:g}", color=color)
        for name_ref in ("pretrained", "retrain"):
            col = f"before_{metric}"
            if len(refs) and col in refs.columns and (refs["model"] == name_ref).any():
                level = 1 - refs[refs["model"] == name_ref].groupby("seed")[col].first().mean()
                color, label = REF_STYLE[name_ref]
                for ax in (ul, rl):
                    ax.axhline(level, color=color, linestyle=":", linewidth=1,
                               label=f"{label} (before relearning)" if ax is rl else f"{label}")
        if len(refs) and f"rl_{metric}" in refs.columns:
            for name_ref in ("retrain", "scratch"):
                group = refs[refs["model"] == name_ref]
                if len(group):
                    color, label = REF_STYLE[name_ref]
                    _band(rl, group.assign(err=1 - group[f"rl_{metric}"]), "n_f", "err",
                          f"{label}, relearned", color=color, linestyle="--")
    for ax in axes.ravel():
        ax.set_xscale("log")
    axes[0][0].set_ylim(0.0, 1.5)  # headroom for the legends (sharey: the whole row)
    axes[0][0].legend(fontsize=7, loc="upper right")
    axes[0][1].legend(fontsize=7, loc="upper right", ncol=2)
    axes[0][2].legend(fontsize=6, loc="upper right", ncol=2)
    fig.suptitle(f"{exp}: sample efficiency")
    fig.tight_layout()
    path = dest / f"sample_efficiency_{exp}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def _pair_accuracy(pretrain: pd.DataFrame, pair, forget: str) -> pd.DataFrame:
    """Test accuracy on the kept classes (10-way) and on the forgotten classes, from per-class accuracies
    ("10way") or from the pairwise logit comparisons ("2way")."""
    a, b = pair
    kept = [f"acc_c{k}" for k in range(10) if k not in pair]
    out = {"acc_r": pretrain[kept].mean(axis=1)}
    if forget == "2way":
        out["acc_f"] = (pretrain[f"pair_{a}_{b}"] + pretrain[f"pair_{b}_{a}"]) / 2
    else:
        out["acc_f"] = pretrain[[f"acc_c{a}", f"acc_c{b}"]].mean(axis=1)
    return pd.DataFrame({"seed": pretrain["seed"], "n": pretrain["n_train"], **out})


def _with_anchor(frame: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    """Add the n = 0 point (one row per seed, the state before the stage) to a table of n, acc_r, acc_f."""
    start = anchor.reset_index().assign(n=0)
    return pd.concat([start[["seed", "n", "acc_r", "acc_f"]], frame[["seed", "n", "acc_r", "acc_f"]]])


SCRATCH_COLOR = "#e07a1f"  # from-scratch relearning reference in the pipeline figure


def plot_pipeline(exp: str, pretrain: pd.DataFrame, unlearn: pd.DataFrame, refs: pd.DataFrame, dest: Path,
                  forget: str = "2way", sweep: Optional[pd.DataFrame] = None) -> List[Path]:
    """The three stages as accuracy against the number of samples each used (mean +/- std over seeds).

    (a) pretraining from scratch on n images of all 10 classes (the last point is the 50k model that
        stages 2 and 3 start from); (b) unlearning that model with n retain and n forget images, one line style
        per mu_f, starting at n = 0 from the pretrained model; (c) relearning each of those on its n forget
        images, plus a network trained from scratch on the same images ("fresh"), starting at n = 0 from the
        pretrained model (and from the untrained network for fresh). Teal is accuracy on the kept classes
        (10-way), purple on the forgotten classes (2-way or 10-way, `forget`).

    With `sweep` (the relearn_sweep table), a panel (d) is added: the models unlearned with a fixed n_UL,
    relearned with every n_RL, starting at n_RL = 0 from those unlearned models. Runs that diverged are left
    out (their record is the rolled-back unlearned model, not a relearning result)."""
    if not exp.startswith("pair-") or not len(unlearn) or "retain_matched" not in unlearn.columns:
        return []
    # --theory runs (n = rows of both kinds) take precedence over matched-retain runs (n = forget images)
    theory = "theory" in unlearn.columns and unlearn["theory"].astype(bool).any()
    keep = unlearn["theory"].astype(bool) if theory else unlearn["retain_matched"].astype(bool)
    u = unlearn[keep & (unlearn["rescale"] == "none")] if "rescale" in unlearn.columns else unlearn[keep]
    if theory:
        u = u.assign(n_f=u["n_rows"])
        if "theory" in refs.columns:
            refs = refs[refs["theory"].astype(bool) | (refs["model"] == "pretrained")]
            refs = refs.assign(n_f=refs["n_rows"].where(refs["theory"].astype(bool), refs["n_f"]))
    if not len(u) or not len(refs):
        return []
    pair = tuple(int(v) for v in exp[len("pair-"):].split("-"))
    f_col = "A_f2" if forget == "2way" else "A_f10"

    def levels(model: str) -> pd.DataFrame:
        """Per-seed accuracy of a reference model before relearning, as columns acc_r, acc_f."""
        g = refs[refs["model"] == model].groupby("seed")[[f"before_A_r", f"before_{f_col}"]].first()
        return g.rename(columns={"before_A_r": "acc_r", f"before_{f_col}": "acc_f"})

    colors = {"acc_r": "tab:cyan" if False else "#2a9d8f", "acc_f": "#3b1f7a"}
    styles = {0.0: "-", 0.25: "--", 0.5: "-.", 0.75: ":"}
    if sweep is not None:
        sweep = sweep[(sweep["exp"] == exp) & (sweep["rl_status"] != "diverged")] if "exp" in sweep.columns \
            else sweep[sweep["rl_status"] != "diverged"]
        sweep = sweep if len(sweep) else None
    fig, axes = plt.subplots(1, 4 if sweep is not None else 3, figsize=(17 if sweep is not None else 13, 3.8),
                             sharey=True)

    def draw(ax, frame: pd.DataFrame, linestyle="-", marker="o", alpha=1.0, cols=("acc_f", "acc_r"),
             color=None, lw=1.5) -> None:
        for col in cols:
            stats = frame.groupby("n")[col].agg(["mean", "std"]).reset_index().sort_values("n")
            c = color or colors[col]
            ax.plot(stats["n"], stats["mean"], color=c, linestyle=linestyle, marker=marker, ms=3,
                    alpha=alpha, lw=lw)
            ax.fill_between(stats["n"], stats["mean"] - stats["std"].fillna(0), stats["mean"] + stats["std"].fillna(0),
                            color=c, alpha=0.12, linewidth=0)

    # (a) pretraining
    if len(pretrain) and (forget == "10way" or f"pair_{pair[0]}_{pair[1]}" in pretrain.columns):
        frame = _pair_accuracy(pretrain, pair, forget)
        top = levels("pretrained").reset_index().assign(n=50000)
        draw(axes[0], pd.concat([frame, top[["seed", "n", "acc_r", "acc_f"]]]))
    else:
        axes[0].text(0.5, 0.5, "n/a: no pretrain_curve results", ha="center", transform=axes[0].transAxes)
    # (b) unlearning, (c) relearning
    start = levels("pretrained")
    for mu_f, group in u.groupby("mu_f"):
        ls = styles.get(float(mu_f), "-")
        draw(axes[1], _with_anchor(group.assign(n=group["n_f"], acc_r=group["ul_A_r"], acc_f=group[f"ul_{f_col}"]), start), ls)
        draw(axes[2], _with_anchor(group.assign(n=group["n_f"], acc_r=group["rl_A_r"], acc_f=group[f"rl_{f_col}"]), start), ls)
    fresh = refs[refs["model"] == "scratch"]
    if len(fresh):
        frame = fresh.assign(n=fresh["n_f"], acc_r=fresh["rl_A_r"], acc_f=fresh[f"rl_{f_col}"])
        # from scratch: forgotten classes only (its kept-class accuracy is near chance by construction)
        draw(axes[2], _with_anchor(frame, levels("scratch")), linestyle="-", marker="s", cols=("acc_f",),
             color=SCRATCH_COLOR, lw=2.2)
    titles = ["(a) Stage 1: pretraining", "(b) Stage 2: unlearning", "(c) Stage 3: relearning"]
    xlabels = ["images in training set  n_PT", "unlearning images  n_UL" if theory else "retain and forget images  n_UL",
               "relearning images  n_RL (n_UL = n_RL)" if theory else "forget images  n_RL"]
    if sweep is not None:
        n_ul = int(sweep["n_ul"].iloc[0])
        for mu_f, group in sweep.groupby("mu_f"):
            anchor = group.groupby("seed")[["ul_A_r", f"ul_{f_col}"]].first().rename(
                columns={"ul_A_r": "acc_r", f"ul_{f_col}": "acc_f"})
            draw(axes[3], _with_anchor(group.assign(n=group["n_rows"], acc_r=group["rl_A_r"],
                                                    acc_f=group[f"rl_{f_col}"]), anchor), styles.get(float(mu_f), "-"))
        if len(fresh):
            draw(axes[3], _with_anchor(frame, levels("scratch")), linestyle="-", marker="s", cols=("acc_f",),
                 color=SCRATCH_COLOR, lw=2.2)
        titles.append(f"(d) Stage 3 from n_UL = {n_ul}")
        xlabels.append("relearning images  n_RL")

    for ax, title, xlabel in zip(axes, titles, xlabels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(left=0)  # n = 0 is the left edge; symlog would otherwise show a negative axis
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("accuracy")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colors["acc_r"], label="kept classes R (10-way)"),
               Line2D([], [], color=colors["acc_f"], label=f"forgotten classes F ({forget[:-3]}-way)")]
    handles += [Line2D([], [], color="black", linestyle=styles.get(float(m), "-"), label=f"mu_f = {m:g}")
                for m in sorted(u["mu_f"].unique())]
    if len(fresh):
        handles.append(Line2D([], [], color=SCRATCH_COLOR, lw=2.2, marker="s", ms=3,
                              label=f"from scratch, F ({forget[:-3]}-way)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle(f"{exp}: three stages, accuracy against samples used")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    Path(dest).mkdir(parents=True, exist_ok=True)
    path = Path(dest) / f"pipeline_{exp}_{forget}{'_sweep' if sweep is not None else ''}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


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

    unlearn, refs, pretrain = read("unlearn"), read("references"), read("pretrain_curve")
    if len(unlearn):
        unlearn = unlearn[unlearn["rescale"] == "none"]
    scaled = unlearn  # runs whose retain set shrinks with the fraction (the figures against f)
    if len(unlearn) and "retain_full" in unlearn.columns:
        scaled = unlearn[~unlearn["retain_full"].astype(bool) & ~unlearn["retain_matched"].astype(bool)]
    exps = sorted(set(unlearn.get("exp", pd.Series(dtype=str))) | set(refs.get("exp", pd.Series(dtype=str))))
    for exp in exps:
        u = unlearn[unlearn["exp"] == exp] if len(unlearn) else unlearn
        us = scaled[scaled["exp"] == exp] if len(scaled) else scaled
        r = refs[refs["exp"] == exp] if len(refs) else refs
        three = [p for forget in ("2way", "10way") for p in plot_pipeline(exp, pretrain, u, r, dest, forget)]
        for path in plot_experiment(exp, us, r, dest) + plot_sample_efficiency(exp, pretrain, u, r, dest) + three:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
