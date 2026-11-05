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


def plot_experiment(exp: str, unlearn: pd.DataFrame, refs: pd.DataFrame, dest: Path, loss: str = "") -> List[Path]:
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
        fig.suptitle(f"{_title(exp, loss)}: after unlearning")
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
        fig.suptitle(f"{_title(exp, loss)}: after relearning")
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
                           dest: Path, loss: str = "") -> List[Path]:
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
    fig.suptitle(f"{_title(exp, loss)}: sample efficiency")
    fig.tight_layout()
    path = dest / f"sample_efficiency_{exp}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def _pair_accuracy(pretrain: pd.DataFrame, pair, forget: str, retain: str = "10way") -> pd.DataFrame:
    """Test accuracy on the kept classes (10-way from per-class accuracies, or "8way": argmax over the kept
    classes only) and on the forgotten classes, from per-class accuracies ("10way") or from the pairwise
    logit comparisons ("2way")."""
    a, b = pair
    kept = [f"acc_c{k}" for k in range(10) if k not in pair]
    out = {"acc_r": pretrain[f"r8_{min(a, b)}_{max(a, b)}"] if retain == "8way" else pretrain[kept].mean(axis=1)}
    if forget == "2way":
        out["acc_f"] = (pretrain[f"pair_{a}_{b}"] + pretrain[f"pair_{b}_{a}"]) / 2
    else:
        out["acc_f"] = pretrain[[f"acc_c{a}", f"acc_c{b}"]].mean(axis=1)
    return pd.DataFrame({"seed": pretrain["seed"], "n": pretrain["n_train"], **out})


LOSS_NAMES = {"ce": "CE loss", "mse": "MSE loss"}


def _title(exp: str, loss: str = "") -> str:
    """Figure title prefix: the experiment and, when known, the loss the models were trained with."""
    return f"{exp}, {LOSS_NAMES.get(loss, loss)}" if loss else exp


SCRATCH_COLOR = "#e07a1f"  # from-scratch relearning reference in the pipeline figure


def select_unlearning_size(sweep: pd.DataFrame, n_ul: Optional[int] = None) -> pd.DataFrame:
    """The rows of a relearn_sweep table that start from `n_ul` unlearning rows (default: the largest present), so
    one panel never mixes unlearning sizes."""
    if not len(sweep):
        return sweep
    return sweep[sweep["n_ul"] == (n_ul if n_ul is not None else sweep["n_ul"].max())]


def plot_pipeline(exp: str, pretrain: pd.DataFrame, unlearn: pd.DataFrame, refs: pd.DataFrame, dest: Path,
                  forget: str = "2way", sweep: Optional[pd.DataFrame] = None,
                  retain: str = "10way", n_ul: Optional[int] = None, loss: str = "") -> List[Path]:
    """The three stages as test accuracy against the number of samples each used (symlog x axis, n = 0 is the model before the stage; mean and
    standard error over seeds as a line with an error bar at every point).

    Top row: accuracy on the kept classes R; bottom row: accuracy on the forgotten classes F (2-way or 10-way,
    `forget`). (a) pretraining from scratch on n images of all 10 classes (the last point is the 50k model that
    stages 2 and 3 start from); (b) unlearning that model with n rows, one colour per mu_f; (c) relearning each
    of those on the same n rows, plus a network trained from scratch on them (orange, forgotten classes only: its
    kept-class accuracy is chance by construction). With `sweep` (the relearn_sweep table), panel (d) relearns the
    models unlearned with a fixed n_UL on n_RL images. Runs that diverged are left out. `retain` "8way" measures
    the kept classes with the argmax over their own logits only (A_r8), the counterpart of the 2-way forget
    measure. `n_ul`: which unlearning size panel (d) starts from when the sweep table holds several (default: the
    largest)."""
    import numpy as np
    from matplotlib.lines import Line2D

    r_col = "A_r8" if retain == "8way" else "A_r"
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
    if sweep is not None:
        sweep = sweep[(sweep["exp"] == exp) & (sweep["rl_status"] != "diverged")] if "exp" in sweep.columns \
            else sweep[sweep["rl_status"] != "diverged"]
        sweep = select_unlearning_size(sweep, n_ul)
        sweep = sweep if len(sweep) else None

    mu_fs = sorted(float(m) for m in u["mu_f"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {m: cmap(0.9 * i / max(len(mu_fs) - 1, 1)) for i, m in enumerate(mu_fs)}
    pre_color = "#444444"
    n_cols = 4 if sweep is not None else 3
    fig, axes = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 6.2), sharey="row")

    def draw(ax, frame: pd.DataFrame, col: str, color, marker="o", lw=1.8) -> None:
        """Accuracy in column `col` against n: mean line with standard-error bars."""
        frame = frame.assign(err=lambda f: f[col])
        stats = frame.groupby("n")["err"].agg(["mean", "sem"]).reset_index().sort_values("n")
        sem = stats["sem"].fillna(0).to_numpy()
        mean = stats["mean"].to_numpy()
        ax.errorbar(stats["n"], mean, yerr=sem, color=color, marker=marker, ms=3.5,
                    lw=lw, elinewidth=1.1, capsize=2.5, capthick=1.1)

    def rows(frame: pd.DataFrame, r: str, f: str) -> pd.DataFrame:
        return frame.assign(n=frame["n_f"], acc_r=frame[r], acc_f=frame[f])

    def levels(model: str) -> pd.DataFrame:
        """Per-seed accuracy of a reference model before relearning, as columns seed, acc_r, acc_f."""
        g = refs[refs["model"] == model].groupby("seed")[[f"before_{r_col}", f"before_{f_col}"]].first()
        return g.rename(columns={f"before_{r_col}": "acc_r", f"before_{f_col}": "acc_f"}).reset_index()

    def anchored(frame: pd.DataFrame, start: pd.DataFrame) -> pd.DataFrame:
        """Add the n = 0 point (one row per seed: the model before the stage) to a table of n, acc_r, acc_f."""
        return pd.concat([start[["seed", "acc_r", "acc_f"]].assign(n=0), frame[["seed", "n", "acc_r", "acc_f"]]])

    # (a) pretraining
    if len(pretrain) and (forget == "10way" or f"pair_{pair[0]}_{pair[1]}" in pretrain.columns):
        frame = _pair_accuracy(pretrain, pair, forget, retain)
        frame = frame[frame["n"] > 0]  # no point at n = 0 (the untrained network)
        frame = pd.concat([frame, levels("pretrained").assign(n=50000)[["seed", "n", "acc_r", "acc_f"]]])
        draw(axes[0, 0], frame, "acc_r", pre_color)
        draw(axes[1, 0], frame, "acc_f", pre_color)
    else:
        axes[0, 0].text(0.5, 0.5, "n/a: no pretrain_curve results", ha="center", transform=axes[0, 0].transAxes)
    # (b) unlearning, (c) relearning
    for m in mu_fs:
        group = u[u["mu_f"].astype(float) == m]
        for stage, ax_col in (("ul", 1), ("rl", 2)):
            frame = anchored(rows(group, f"{stage}_{r_col}", f"{stage}_{f_col}"), levels("pretrained"))
            draw(axes[0, ax_col], frame, "acc_r", colors[m])
            draw(axes[1, ax_col], frame, "acc_f", colors[m])
    fresh = refs[refs["model"] == "scratch"]
    if len(fresh):
        frame = anchored(rows(fresh, f"rl_{r_col}", f"rl_{f_col}"), levels("scratch"))
        draw(axes[1, 2], frame, "acc_f", SCRATCH_COLOR, marker="s", lw=2.2)
    if sweep is not None:
        for m in mu_fs:
            group = sweep[sweep["mu_f"].astype(float) == m]
            start = group.groupby("seed")[[f"ul_{r_col}", f"ul_{f_col}"]].first().rename(
                columns={f"ul_{r_col}": "acc_r", f"ul_{f_col}": "acc_f"}).reset_index()
            frame = anchored(group.assign(n=group["n_rows"], acc_r=group[f"rl_{r_col}"], acc_f=group[f"rl_{f_col}"]), start)
            draw(axes[0, 3], frame, "acc_r", colors[m])
            draw(axes[1, 3], frame, "acc_f", colors[m])
        if len(fresh):
            draw(axes[1, 3], anchored(rows(fresh, f"rl_{r_col}", f"rl_{f_col}"), levels("scratch")), "acc_f", SCRATCH_COLOR,
                 marker="s", lw=2.2)

    names = ["pretraining", "unlearning", "relearning ($n_{UL}$ = $n_{RL}$)"]
    xlabels = ["Number of samples", "Number of samples" + (" (rows)" if theory else ""), "Number of samples"]
    if sweep is not None:
        names.append(f"relearning from $n_{{UL}}$ = {int(sweep['n_ul'].iloc[0])}")
        xlabels.append("Number of relearning samples")
    for j, (name, xlabel) in enumerate(zip(names, xlabels)):
        axes[0, j].text(-0.02, 1.12, "abcd"[j], transform=axes[0, j].transAxes, fontsize=20, fontweight="bold",
                        va="bottom", ha="right")
        axes[0, j].set_title(name, fontsize=11, style="italic", loc="right")
        axes[1, j].set_xlabel(xlabel)
        for i in range(2):
            ax = axes[i, j]
            if j == 0:
                ax.set_xscale("log")
            else:
                ax.set_xscale("symlog", linthresh=1)
                ax.set_xlim(left=0)  # n = 0 (the model before the stage) is the left edge
            ax.set_ylim(0, 1.05)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelbottom=True)
    axes[0, 0].set_ylabel(f"Accuracy, R ({retain[:-3]}-way)", fontsize=10)
    axes[1, 0].set_ylabel(f"Accuracy, F ({forget[:-3]}-way)", fontsize=10)
    handles = [Line2D([], [], color=colors[m], lw=2, label=f"{m:g}") for m in mu_fs]
    labels = [f"$\\mu_f$ = {m:g}" for m in mu_fs]
    if len(fresh):
        handles.append(Line2D([], [], color=SCRATCH_COLOR, lw=2.2, marker="s", ms=3.5))
        labels.append("from scratch")
    handles.insert(0, Line2D([], [], color=pre_color, lw=2))
    labels.insert(0, "pretraining")
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=9, frameon=False)
    fig.suptitle(f"{_title(exp, loss)}: R = kept classes, F = forgotten classes (mean $\\pm$ s.e.m. over seeds)",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    Path(dest).mkdir(parents=True, exist_ok=True)
    path = Path(dest) / f"pipeline_{exp}_{forget}{'_r8' if retain == '8way' else ''}{'_sweep' if sweep is not None else ''}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--src", default=None)
    parser.add_argument("--dest", default=None)
    parser.add_argument("--n_ul", type=int, default=None,
                        help="the unlearning size panel (d) of the pipeline figures starts from (default: the largest)")
    args = parser.parse_args(argv)
    src = Path(args.src or f"data/processed/cifar_unlearning/{args.loss}")
    dest = Path(args.dest or f"figures/cifar_unlearning/{args.loss}")
    dest.mkdir(parents=True, exist_ok=True)

    def read(name: str) -> pd.DataFrame:
        path = src / f"{name}.csv"
        return pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()

    unlearn, refs, pretrain = read("unlearn"), read("references"), read("pretrain_curve")
    sweep = read("relearn_sweep")
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
        s = sweep[sweep["exp"] == exp] if len(sweep) else None
        # 8-way retain needs A_r8 in every table the figure reads (records from before it was added lack it)
        has_r8 = ("ul_A_r8" in u.columns and u["ul_A_r8"].notna().all() and "before_A_r8" in r.columns
                  and (not len(pretrain) or f"r8_{exp.split('-')[1]}_{exp.split('-')[-1]}" in pretrain.columns)
                  if exp.startswith("pair-") else False)
        jobs = [lambda: plot_experiment(exp, us, r, dest, args.loss),
                lambda: plot_sample_efficiency(exp, pretrain, u, r, dest, loss=args.loss)]
        for retain in ("10way", "8way") if has_r8 else ("10way",):
            for forget in ("2way", "10way"):
                jobs.append(lambda retain=retain, forget=forget: plot_pipeline(exp, pretrain, u, r, dest, forget,
                                                                               retain=retain, loss=args.loss))
                if s is not None and len(s):
                    jobs.append(lambda retain=retain, forget=forget: plot_pipeline(
                        exp, pretrain, u, r, dest, forget, sweep=s, retain=retain, n_ul=args.n_ul, loss=args.loss))
        for job in jobs:  # one failing figure must not stop the others (this runs unattended on the cluster)
            try:
                for path in job():
                    print(f"wrote {path}")
            except Exception as err:  # noqa: BLE001
                print(f"FAILED a figure for {exp}: {type(err).__name__}: {err}")


if __name__ == "__main__":
    main()
