"""Entry point for every stage. Run from the repository root:

    python experiments/cifar_unlearning/run.py <stage> [flags]

Stages, in order:
    download         fetch CIFAR-10 once and cache it as uint8 tensors (login node)
    pretrain         --seed S                      train on all of D
    pretrain_curve   --seed S                      pretrain on nested subsets of D (sample-efficiency curve)
    select_pairs     --seeds 0 1 2 3 4             pick pilot pairs from pretrained confusion
    retrain          --exp E --seed S              the control: train on all of D_r
    unlearn_relearn  --exp E --seed S --ul_mu_f M  unlearn Pretrained at every fraction, then relearn
                     (--forget_sizes n1 n2 ...: n forget images and the whole retain set instead)
    references       --exp E --seed S              relearn Pretrained, Retrain and Scratch at every fraction
    represent        --exp E --seed S              CKA and k-NN analysis of every saved model (post hoc)

E is "random" or "pair-A-B". Every model is saved: Pretrained and Retrain in fp32,
every unlearned model before and after relearning and every relearned reference in
fp16. Every run writes result.json under --out_root and is skipped when its result.json exists, so a SLURM
array can be resubmitted safely. A run that fails a check writes failed.json and
exits non-zero, which stops dependent SLURM jobs.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.cifar_unlearning.config import (  # noqa: E402
    DEFAULT_FRACTIONS, DEFAULT_SEEDS, add_config_args, build_configs, configs_to_dict)
from experiments.cifar_unlearning.data import (  # noqa: E402
    Dataset, Split, load_cifar10, load_split, make_split, make_synthetic, n_for_fraction, normalize,
    parse_experiment, save_split, stratified_order)
from experiments.cifar_unlearning.losses import LOSS_KINDS  # noqa: E402
from experiments.cifar_unlearning.metrics import (  # noqa: E402
    EvalSets, confusion_matrix, evaluate, kept_class_accuracy, predict_logits)
from experiments.cifar_unlearning.representation import (  # noqa: E402
    RepSets, reference_features, representation_metrics)
from experiments.cifar_unlearning.models import (  # noqa: E402
    build_model, is_identity_rescale, load_state, mean_logit_norm, recalibrate_bn, rescale_,
    state_dict_fp16, weight_norms)
from experiments.cifar_unlearning import train as train_module  # noqa: E402
from experiments.cifar_unlearning.train import (  # noqa: E402
    relearn, train_from_scratch, unlearn)

log = partial(print, flush=True)
STAGES = ("download", "pretrain", "pretrain_curve", "select_pairs", "retrain", "unlearn_relearn", "relearn_sweep",
          "references", "reeval", "represent")
REFERENCE_MODELS = ("pretrained", "retrain", "scratch")


# --------------------------------------------------------------------------- helpers

def derive_seed(*parts) -> int:
    """Stable 31-bit seed from any tuple of labels."""
    digest = hashlib.sha256(repr(parts).encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def fmt(x: float) -> str:
    return f"{x:g}"


def _tag(fraction: float, size: Optional[int], matched: bool = False, theory: bool = False) -> str:
    """Folder name of one data-size setting: `n=<images>` for --forget_sizes runs (`m=<images>` when the
    retain set is cut to the same size, `t=<rows>` in --theory mode), `f=<fraction>` otherwise."""
    if size is None:
        return f"f={fmt(fraction)}"
    return f"{'t' if theory else ('m' if matched else 'n')}={size}"


def theory_rows(split: "Split", n: int, fresh: bool = False):
    """--theory mode: n rows from the whole training distribution, as in the theory, where both loss terms
    are evaluated on the same rows. The forget share is the forget pool's share of the training images
    (at least one row of each kind). Unlearning takes the start of the nested orders; relearning
    (`fresh`) takes the end, so the two stages see disjoint rows, like the theory's fresh design per stage.
    Returns (retain indices, forget indices)."""
    pool, retain = split.forget_pool_order, split.retain_order
    n_f = min(max(1, round(n * len(pool) / (len(pool) + len(retain)))), n - 1)
    n_r = n - n_f
    if 2 * n_f > len(pool) or 2 * n_r > len(retain):
        raise SystemExit(f"--theory: {n} rows do not leave disjoint unlearning and relearning rows")
    if fresh:
        return retain[::-1][:n_r].copy(), pool[::-1][:n_f].copy()
    return retain[:n_r], pool[:n_f]


def theory_relearn_set(ctx: "Context", split: "Split", n: int):
    """Rows and target vectors of the theory's forget-only attack (nu_r = 0): the loss towards y_f = y * M_f,
    i.e. onehot for forget rows and, for retain rows, the zero vector (mse) or the uniform distribution (ce,
    where equal logits are the counterpart of zero), over all outputs."""
    r_idx, f_idx = theory_rows(split, n, fresh=True)
    x_r, y_r = ctx.train_subset(r_idx)
    x_f, y_f = ctx.train_subset(f_idx)
    k = ctx.dataset.num_classes
    retain_target = torch.zeros(len(y_r), k, device=x_r.device)
    if ctx.loss == "ce":
        retain_target += 1.0 / k
    targets = torch.cat([retain_target,
                         torch.nn.functional.one_hot(y_f, k).float()])
    return torch.cat([x_r, x_f]), torch.cat([y_r, y_f]), targets, len(r_idx), len(f_idx)


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def read_json(path: Path) -> Dict:
    with open(path) as fh:
        return json.load(fh)


class Context:
    """Parsed flags, configs, device and lazily loaded data for one process."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = build_configs(args)
        self.loss = args.loss
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.out = Path(args.out_root or f"data/cifar_unlearning/{self.loss}")
        self._dataset: Optional[Dataset] = None
        self._x_train: Optional[torch.Tensor] = None
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    @property
    def dataset(self) -> Dataset:
        if self._dataset is None:
            d = self.cfg["data"]
            if d.synthetic:
                self._dataset = make_synthetic(d.synthetic_train_per_class, d.synthetic_test_per_class,
                                               self.cfg["model"].num_classes)
            else:
                self._dataset = load_cifar10(d.data_root, download=False)
        return self._dataset

    @property
    def x_train(self) -> torch.Tensor:
        """All training images, normalised, on the device (about 600 MB for CIFAR-10)."""
        if self._x_train is None:
            self._x_train = normalize(self.dataset.x_train, self.device)
        return self._x_train

    @property
    def y_train(self) -> torch.Tensor:
        return self.dataset.y_train.to(self.device)

    def train_subset(self, idx: np.ndarray):
        index = torch.as_tensor(idx, device=self.device)
        return self.x_train[index], self.y_train[index]

    def split(self, experiment: str, seed: int) -> Split:
        """The split for (experiment, seed), saved to disk the first time it is made.

        Every later stage loads the saved indices, so Retrain, unlearning and the
        references always use the same D_f, probe and subsamples, even if library
        versions change between jobs.
        """
        d = self.cfg["data"]
        path = self.out / "splits" / f"exp={experiment}" / f"seed={seed}.npz"
        if path.exists():
            split = load_split(str(path))
            kind, pair = parse_experiment(experiment)
            expected_forget = None if kind == "class" else d.n_forget
            if (split.kind != kind or split.pair != pair or len(split.probe) != d.n_probe
                    or (expected_forget is not None and len(split.forget_all) != expected_forget)):
                raise SystemExit(f"{path} does not match the current --n_probe/--n_forget; "
                                 "use a new --out_root or delete the saved split")
            return split
        split = make_split(self.dataset, experiment, seed, n_forget=d.n_forget, n_probe=d.n_probe)
        save_split(split, str(path))
        return split

    def eval_sets(self, split: Split) -> EvalSets:
        ds = self.dataset

        def test(idx):
            index = torch.as_tensor(idx, dtype=torch.int64)
            return normalize(ds.x_test[index], self.device), ds.y_test[index].to(self.device)

        x_tr, y_tr = test(split.test_retain)
        x_tf, y_tf = test(split.test_forget)
        x_p, y_p = self.train_subset(split.probe)
        return EvalSets(x_tr, y_tr, x_tf, y_tf, x_p, y_p)

    def new_model(self, init_seed: int) -> torch.nn.Module:
        torch.manual_seed(init_seed)
        m = self.cfg["model"]
        return build_model(m.arch, m.num_classes).to(self.device)

    def load_model(self, path: Path) -> torch.nn.Module:
        model = self.new_model(0)
        load_state(model, torch.load(path, map_location=self.device))
        return model

    def base_record(self, **fields) -> Dict:
        extra = {"augment": True} if train_module.AUGMENT else {}
        return {**fields, "loss": self.loss, "configs": configs_to_dict(self.cfg), **extra}

    # paths -----------------------------------------------------------------
    def pretrain_dir(self, seed: int, read: bool = False) -> Path:
        """Where Pretrain writes; with read=True, where later stages load it from
        (--pretrain_root, if given, so calibration runs can share one pretrained model)."""
        base = Path(self.args.pretrain_root) if (read and self.args.pretrain_root) else self.out
        return base / "pretrain" / f"seed={seed}"

    def retrain_dir(self, exp: str, seed: int) -> Path:
        return self.out / "retrain" / f"exp={exp}" / f"seed={seed}"

    def rescale_tag(self) -> str:
        rs = self.cfg["rs"]
        if is_identity_rescale(rs.mode, rs.S, rs.alpha_body, rs.alpha_head) and not rs.recalibrate_bn:
            return "none"
        if rs.mode == "body_head":
            tag = f"body_head-b{fmt(rs.alpha_body)}-h{fmt(rs.alpha_head)}"
        else:
            tag = f"{rs.mode}-S{fmt(rs.S)}"
        return tag + ("-bnrecal" if rs.recalibrate_bn else "")

    def unlearn_dir(self, exp: str, seed: int, mu_f: float, fraction: float, size: Optional[int] = None) -> Path:
        return (self.out / "unlearn" / f"exp={exp}" / f"seed={seed}" / f"mu_f={fmt(mu_f)}"
                / f"rescale={self.rescale_tag()}" / _tag(fraction, size, self.args.match_retain, self.args.theory))

    def data_grid(self, split: Split):
        """(fraction, size) for every data setting to run. Default: `--fractions`, each scaling the retain
        and forget sets together (size is None). With `--forget_sizes`: the first n images of the forget
        pool and the whole retain set (or the first n retain images with --match_retain); fraction is then
        n / |forget pool|."""
        sizes = self.args.forget_sizes
        if not sizes:
            return [(f, None) for f in sorted(self.args.fractions)]
        pool = len(split.forget_pool_order)
        if self.args.theory:  # n is the number of rows of both kinds
            if min(sizes) < 2:
                raise SystemExit("--theory needs --forget_sizes >= 2 (one retain and one forget row)")
            return [(n / pool, n) for n in sorted(sizes)]
        if min(sizes) < 1 or max(sizes) > pool:
            raise SystemExit(f"--forget_sizes must lie in [1, {pool}] (the forget pool), got {sizes}")
        return [(n / pool, n) for n in sorted(sizes)]

    def references_dir(self, exp: str, seed: int) -> Path:
        return self.out / "references" / f"exp={exp}" / f"seed={seed}"


# Config groups that determine each stage's result. A finished result is only reused
# when these match the current flags; otherwise the run stops instead of silently
# mixing settings in one output folder.
STAGE_GROUPS = {
    "pretrain": ("data", "model", "pt"),
    "pretrain_curve": ("data", "model", "pt"),
    "retrain": ("data", "model", "pt"),
    "unlearn_relearn": ("data", "model", "ul", "rl", "rs"),
    "relearn_sweep": ("data", "model", "ul", "rl", "rs"),
    "reeval": (),
    "references": ("data", "model", "rl"),
    "represent": ("data", "model", "rep"),
}


def _done(path: Path, ctx: "Context", stage: str) -> bool:
    if not path.exists() or ctx.args.overwrite:
        return False
    saved = read_json(path)
    current = configs_to_dict(ctx.cfg)
    if saved.get("augment", False) != train_module.AUGMENT:
        raise SystemExit(f"{path} exists with a different --augment setting. Use a new --out_root.")
    for group in STAGE_GROUPS[stage]:
        if saved.get("configs", {}).get(group) != current[group] or saved.get("loss") != ctx.loss:
            raise SystemExit(f"{path} exists with different '{group}' settings (or loss). Use a new --out_root "
                             "for different hyperparameters, or --overwrite to replace it.")
    log(f"skip: {path} exists")
    return True


# --------------------------------------------------------------------------- stages

def stage_download(ctx: Context) -> None:
    ds = load_cifar10(ctx.cfg["data"].data_root, download=True)
    log(f"CIFAR-10 cached: train {tuple(ds.x_train.shape)}, test {tuple(ds.x_test.shape)}")


def stage_pretrain(ctx: Context) -> None:
    seed = ctx.args.seed
    out = ctx.pretrain_dir(seed)
    if _done(out / "result.json", ctx, "pretrain"):
        return
    model = ctx.new_model(derive_seed("pretrain-init", seed))
    train = train_from_scratch(model, ctx.x_train, ctx.y_train, ctx.cfg["pt"],
                               seed=derive_seed("pretrain-batches", seed), loss_kind=ctx.loss, log=log)
    ds = ctx.dataset
    x_test, y_test = normalize(ds.x_test, ctx.device), ds.y_test.to(ctx.device)
    conf = confusion_matrix(model, x_test, y_test, ds.num_classes)
    record = ctx.base_record(stage="pretrain", seed=seed, train=train,
                             test_acc=(conf.diag().sum() / conf.sum()).item(), confusion=conf)
    _finish(out, model, record, ok=train["converged"],
            reason=f"final train loss {train['final_train_loss']:.4f} above threshold")


def stage_pretrain_curve(ctx: Context) -> None:
    """Pretrain from scratch on the first n images of a nested, class-balanced order of D, for every
    size. The model does not depend on the experiment: the confusion matrix on the whole test set
    and the pairwise logit comparisons are saved, so any class pair's retain error (10-way) and forget error
    (2-way) are read off afterwards. Sizes come from `--pt_sizes` (0 = the untrained network) or from
    `--fractions` of D. `--pt_min_steps` lengthens the schedule of small sets so that they converge too.
    A run that does not reach the train-loss threshold is still recorded (`converged` is False), because
    a model trained on few images is part of the curve."""
    seed = ctx.args.seed
    ds = ctx.dataset
    order = stratified_order(np.arange(len(ds.y_train)), ds.y_train.numpy(), np.random.default_rng([seed, 2]))
    x_test, y_test = normalize(ds.x_test, ctx.device), ds.y_test.to(ctx.device)
    if ctx.args.pt_sizes:
        if min(ctx.args.pt_sizes) < 0 or max(ctx.args.pt_sizes) > len(order):
            raise SystemExit(f"--pt_sizes must lie in [0, {len(order)}], got {ctx.args.pt_sizes}")
        grid = [(n / len(order), n) for n in sorted(ctx.args.pt_sizes)]
    else:
        grid = [(f, n_for_fraction(f, len(order))) for f in sorted(ctx.args.fractions)]
    pt = ctx.cfg["pt"]
    for f, n in grid:
        out = ctx.out / "pretrain_curve" / f"seed={seed}" / _tag(f, n)
        if _done(out / "result.json", ctx, "pretrain_curve"):
            continue
        log(f"=== pretrain_curve seed {seed} {n} images")
        model = ctx.new_model(derive_seed("pretrain-curve-init", seed, f))
        if n == 0:
            train = {"steps": 0, "final_train_loss": None, "converged": None, "history": [], "seconds": 0.0}
        else:
            x, y = ctx.train_subset(order[:n])
            per_epoch = -(-n // pt.batch_size)
            epochs = max(pt.epochs, -(-ctx.args.pt_min_steps // per_epoch))
            shown = [0]

            def epoch_log(message: str) -> None:  # thousands of epochs on tiny sets: print about 20 lines
                shown[0] += 1
                if epochs <= 100 or shown[0] % max(1, epochs // 20) == 0:
                    log(message)

            train = train_from_scratch(model, x, y, dataclasses.replace(pt, epochs=epochs),
                                       seed=derive_seed("pretrain-curve-batches", seed, f),
                                       loss_kind=ctx.loss, log=epoch_log)
        conf = confusion_matrix(model, x_test, y_test, ds.num_classes)
        logits = predict_logits(model, x_test)
        # pairwise[a][b]: share of class-a test images whose class-a logit beats the class-b logit. The 2-way
        # accuracy for the pair (a, b) is then (pairwise[a][b] + pairwise[b][a]) / 2 (equal test images per class).
        pairwise = torch.stack([(logits[y_test == a][:, [a]] > logits[y_test == a]).float().mean(dim=0)
                                for a in range(ds.num_classes)]).cpu()
        # retain8[a][b] (a < b): accuracy on the test images of the other 8 classes, argmax over their logits
        k = ds.num_classes
        retain8 = [[0.0] * k for _ in range(k)]
        for a, b in itertools.combinations(range(k), 2):
            kept = (y_test != a) & (y_test != b)
            retain8[a][b] = retain8[b][a] = kept_class_accuracy(logits[kept], y_test[kept], (a, b))
        out.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict_fp16(model), out / "model_fp16.pt")
        write_json(out / "result.json", ctx.base_record(
            stage="pretrain_curve", seed=seed, fraction=f, n_train=n, train=train,
            test_acc=(conf.diag().sum() / conf.sum()).item(), confusion=conf, pairwise=pairwise, retain8=retain8))


def stage_select_pairs(ctx: Context) -> None:
    """Rank all class pairs by how often the pretrained models confuse them (summed
    over seeds) and pick evenly spaced ranks: easiest, ..., hardest."""
    seeds = ctx.args.seeds
    total = None
    for seed in seeds:
        conf = torch.tensor(read_json(ctx.pretrain_dir(seed, read=True) / "result.json")["confusion"], dtype=torch.float64)
        total = conf if total is None else total + conf
    rates = total / total.sum(dim=1, keepdim=True)
    k = rates.shape[0]
    ranking = sorted(
        ({"exp": f"pair-{a}-{b}", "confusion": (rates[a, b] + rates[b, a]).item()}
         for a, b in itertools.combinations(range(k), 2)),
        key=lambda r: r["confusion"])
    n = ctx.args.n_pairs
    picks = sorted({round(i * (len(ranking) - 1) / max(n - 1, 1)) for i in range(n)})
    chosen = [ranking[i]["exp"] for i in picks]
    write_json(ctx.out / "pilot_pairs.json", {"pairs": chosen, "seeds": seeds, "ranking": ranking})
    log(f"pilot pairs (easiest to hardest): {chosen}")


def stage_retrain(ctx: Context) -> None:
    exp, seed = ctx.args.exp, ctx.args.seed
    out = ctx.retrain_dir(exp, seed)
    if _done(out / "result.json", ctx, "retrain"):
        return
    split = ctx.split(exp, seed)
    x_r, y_r = ctx.train_subset(split.retain_order)
    model = ctx.new_model(derive_seed("retrain-init", exp, seed))
    train = train_from_scratch(model, x_r, y_r, ctx.cfg["pt"], seed=derive_seed("retrain-batches", exp, seed),
                               loss_kind=ctx.loss, log=log)
    metrics = evaluate(model, ctx.eval_sets(split), split.pair)
    record = ctx.base_record(stage="retrain", exp=exp, seed=seed, n_retain=len(split.retain_order),
                             train=train, metrics=metrics)
    _finish(out, model, record, ok=train["converged"],
            reason=f"final train loss {train['final_train_loss']:.4f} above threshold")


def _finish(out: Path, model: torch.nn.Module, record: Dict, ok: bool, reason: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    if ok:
        write_json(out / "result.json", record)
        log(f"wrote {out / 'result.json'}")
    else:
        write_json(out / "failed.json", {**record, "failure": reason})
        log(f"FAILED: {reason}; wrote {out / 'failed.json'}")
        sys.exit(1)


def _relearn_and_record(ctx: Context, model, split: Split, ev: EvalSets, x_f, y_f, seed: int,
                        freeze_bn: bool, targets=None) -> Dict:
    result = relearn(model, x_f, y_f, ctx.cfg["rl"], seed=seed, freeze_bn=freeze_bn, pair=split.pair,
                     loss_kind=ctx.loss, eval_fn=lambda m: evaluate(m, ev, split.pair), log=log, targets=targets)
    return {"relearn": result, "after_relearn": evaluate(model, ev, split.pair)}


def stage_unlearn_relearn(ctx: Context) -> None:
    exp, seed = ctx.args.exp, ctx.args.seed
    ul, rs = ctx.cfg["ul"], ctx.cfg["rs"]
    split = ctx.split(exp, seed)
    todo = [(f, n) for f, n in ctx.data_grid(split)
            if not _done(ctx.unlearn_dir(exp, seed, ul.mu_f, f, n) / "result.json", ctx, "unlearn_relearn")]
    if not todo:
        return
    pretrained_state = torch.load(ctx.pretrain_dir(seed, read=True) / "model.pt", map_location=ctx.device)
    ev = ctx.eval_sets(split)
    for f, size in todo:
        out = ctx.unlearn_dir(exp, seed, ul.mu_f, f, size)
        log(f"=== {exp} seed {seed} mu_f {fmt(ul.mu_f)} rescale {ctx.rescale_tag()} "
            f"{_tag(f, size, ctx.args.match_retain, ctx.args.theory)}")
        if ctx.args.theory:
            retain_idx, forget_idx = theory_rows(split, size)
        elif size is None:
            retain_idx, forget_idx = split.retain_subset(f), split.forget_subset(f)
        else:
            retain_idx = split.retain_order[:size] if ctx.args.match_retain else split.retain_order
            forget_idx = split.forget_pool_order[:size]
        x_r, y_r = ctx.train_subset(retain_idx)
        x_f, y_f = ctx.train_subset(forget_idx)

        model = ctx.new_model(0)
        model.load_state_dict(pretrained_state)
        rescale_info = {"tag": ctx.rescale_tag()}
        if ctx.rescale_tag() != "none":
            rescale_info["before"] = {"weight_norms": weight_norms(model),
                                      "mean_logit_norm": mean_logit_norm(model, ev.x_test_retain)}
            rescale_(model, rs.mode, rs.S, rs.alpha_body, rs.alpha_head)
            if rs.recalibrate_bn:
                recalibrate_bn(model, x_r)
            rescale_info["after"] = {"weight_norms": weight_norms(model),
                                     "mean_logit_norm": mean_logit_norm(model, ev.x_test_retain),
                                     "metrics": evaluate(model, ev, split.pair)}

        n_rows = len(retain_idx) + len(forget_idx)
        weights = (len(retain_idx) / n_rows, len(forget_idx) / n_rows) if ctx.args.theory else (1.0, 1.0)
        ul_result = unlearn(model, x_r, y_r, x_f, y_f, ul, seed=derive_seed("unlearn", exp, seed, ul.mu_f, f),
                            loss_kind=ctx.loss, log=log, row_weights=weights)
        after_unlearn = evaluate(model, ev, split.pair)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict_fp16(model), out / "model_fp16.pt")

        if ctx.args.theory:
            x_rl, y_rl, t_rl, _, _ = theory_relearn_set(ctx, split, size)
        else:
            x_rl, y_rl, t_rl = x_f, y_f, None
        relearned = _relearn_and_record(ctx, model, split, ev, x_rl, y_rl,
                                        seed=derive_seed("relearn", exp, seed, ul.mu_f, f),
                                        freeze_bn=ctx.cfg["rl"].freeze_bn, targets=t_rl)
        torch.save(state_dict_fp16(model), out / "model_relearned_fp16.pt")
        record = ctx.base_record(stage="unlearn_relearn", exp=exp, seed=seed, kind=split.kind, pair=split.pair,
                                 fraction=f, retain_full=size is not None and not ctx.args.match_retain,
                                 retain_matched=size is not None and ctx.args.match_retain and not ctx.args.theory,
                                 theory=ctx.args.theory, n_rows=n_rows, n_r=len(retain_idx),
                                 n_f=len(forget_idx), mu_r=ul.mu_r,
                                 mu_f=ul.mu_f, rescale=rescale_info, unlearn=ul_result,
                                 after_unlearn=after_unlearn, **relearned)
        write_json(out / "result.json", record)


def stage_relearn_sweep(ctx: Context) -> None:
    """--theory mode: relearn the model unlearned with `--ul_from` rows (one mu_f) with every relearning size in
    --forget_sizes, the stage-3 attack of the theory. The unlearned model is the one unlearn_relearn saved, so
    the --ul_* flags must match that run. Relearning rows are the same as for the references of each size, so
    the from-scratch curve applies unchanged. Writes <unlearn dir>/relearn_sweep/t=<n>/result.json."""
    exp, seed, ul = ctx.args.exp, ctx.args.seed, ctx.cfg["ul"]
    if not ctx.args.theory or not ctx.args.ul_from:
        raise SystemExit("relearn_sweep needs --theory and --ul_from")
    split = ctx.split(exp, seed)
    n_ul = ctx.args.ul_from
    src = ctx.unlearn_dir(exp, seed, ul.mu_f, n_ul / len(split.forget_pool_order), n_ul)
    if not (src / "result.json").exists():
        raise SystemExit(f"{src} has no unlearned model: run unlearn_relearn with --forget_sizes including {n_ul}")
    source = read_json(src / "result.json")
    if source["configs"]["ul"] != configs_to_dict(ctx.cfg)["ul"]:
        raise SystemExit(f"{src} was unlearned with different --ul_* settings than the current flags")
    state = torch.load(src / "model_fp16.pt", map_location=ctx.device)
    ev = ctx.eval_sets(split)
    for f, size in ctx.data_grid(split):
        out = src / "relearn_sweep" / _tag(f, size, theory=True)
        if _done(out / "result.json", ctx, "relearn_sweep"):
            continue
        log(f"=== {exp} seed {seed} mu_f {fmt(ul.mu_f)} unlearned t={n_ul}: relearn t={size}")
        model = load_state(ctx.new_model(0), state)
        x_rl, y_rl, t_rl, n_r, n_f = theory_relearn_set(ctx, split, size)
        relearned = _relearn_and_record(ctx, model, split, ev, x_rl, y_rl,
                                        seed=derive_seed("relearn-sweep", exp, seed, ul.mu_f, n_ul, size),
                                        freeze_bn=ctx.cfg["rl"].freeze_bn, targets=t_rl)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict_fp16(model), out / "model_relearned_fp16.pt")
        write_json(out / "result.json", ctx.base_record(
            stage="relearn_sweep", exp=exp, seed=seed, kind=split.kind, pair=split.pair, mu_r=ul.mu_r, mu_f=ul.mu_f,
            theory=True, n_ul=n_ul, n_rows=size, n_r=n_r, n_f=n_f, after_unlearn=source["after_unlearn"], **relearned))


def _add_missing(old: Dict, new: Dict) -> int:
    """Add the keys of `new` that `old` lacks (never changes a stored value). Returns how many were added."""
    added = [key for key in new if key not in old]
    old.update({key: new[key] for key in added})
    return len(added)


def stage_reeval(ctx: Context) -> None:
    """Re-evaluate every saved model of one (experiment, seed) and add the metrics its records lack (e.g. A_r8,
    added after the runs). Covers the unlearned and relearned models of unlearn_relearn, relearn_sweep models,
    the references and their starting models (before.json). Stored values are never changed; records whose
    model was not saved are left as they are."""
    exp, seed = ctx.args.exp, ctx.args.seed
    split = ctx.split(exp, seed)
    ev = ctx.eval_sets(split)

    def metrics(path: Path) -> Dict:
        return evaluate(load_state(ctx.new_model(0), torch.load(path, map_location=ctx.device)), ev, split.pair)

    count = 0
    for res in sorted((ctx.out / "unlearn" / f"exp={exp}" / f"seed={seed}").glob("mu_f=*/rescale=*/*=*/result.json")):
        r, d = read_json(res), res.parent
        n = 0
        if (d / "model_fp16.pt").exists():
            n += _add_missing(r["after_unlearn"], metrics(d / "model_fp16.pt"))
        if (d / "model_relearned_fp16.pt").exists():
            n += _add_missing(r["after_relearn"], metrics(d / "model_relearned_fp16.pt"))
        if n:
            write_json(res, r)
            count += 1
        for sweep in sorted(d.glob("relearn_sweep/*=*/result.json")):
            s = read_json(sweep)
            m = _add_missing(s["after_unlearn"], r["after_unlearn"])
            if (sweep.parent / "model_relearned_fp16.pt").exists():
                m += _add_missing(s["after_relearn"], metrics(sweep.parent / "model_relearned_fp16.pt"))
            if m:
                write_json(sweep, s)
                count += 1
    base = ctx.references_dir(exp, seed)
    if (base / "before.json").exists():
        b = read_json(base / "before.json")
        sources = {"pretrained": ctx.pretrain_dir(seed, read=True) / "model.pt",
                   "retrain": ctx.retrain_dir(exp, seed) / "model.pt"}
        n = 0
        for name in b["metrics"]:
            if name == "scratch":
                model = ctx.new_model(derive_seed("scratch-init", exp, seed, None))
            elif sources.get(name, Path("-")).exists():
                model = ctx.load_model(sources[name])
            else:
                continue
            n += _add_missing(b["metrics"][name], evaluate(model, ev, split.pair))
        if n:
            write_json(base / "before.json", b)
            count += 1
    for res in sorted(base.glob("model=*/*=*/result.json")):
        if (res.parent / "model_fp16.pt").exists():
            r = read_json(res)
            if _add_missing(r["after_relearn"], metrics(res.parent / "model_fp16.pt")):
                write_json(res, r)
                count += 1
    log(f"reeval {exp} seed {seed}: updated {count} records")


def stage_references(ctx: Context) -> None:
    exp, seed = ctx.args.exp, ctx.args.seed
    base = ctx.references_dir(exp, seed)
    split = ctx.split(exp, seed)
    ev = ctx.eval_sets(split)
    sources = {
        "pretrained": ctx.pretrain_dir(seed, read=True) / "model.pt",
        "retrain": ctx.retrain_dir(exp, seed) / "model.pt",
    }

    def fresh(name: str, f: Optional[float]):
        if name == "scratch":
            return ctx.new_model(derive_seed("scratch-init", exp, seed, f))
        return ctx.load_model(sources[name])

    before_path = base / "before.json"
    if not before_path.exists() or ctx.args.overwrite:
        before = {name: evaluate(fresh(name, None), ev, split.pair) for name in REFERENCE_MODELS}
        write_json(before_path, ctx.base_record(stage="references_before", exp=exp, seed=seed, metrics=before))

    for f, size in ctx.data_grid(split):
        if ctx.args.theory:  # the same relearning rows and targets as the unlearned models get
            x_f, y_f, t_f, _, n_forget = theory_relearn_set(ctx, split, size)
            forget_idx = range(n_forget)
        else:
            forget_idx = split.forget_subset(f) if size is None else split.forget_pool_order[:size]
            x_f, y_f = ctx.train_subset(forget_idx)
            t_f = None
        tag = _tag(f, size, theory=ctx.args.theory)
        for name in REFERENCE_MODELS:
            out = base / f"model={name}" / tag
            if _done(out / "result.json", ctx, "references"):
                continue
            log(f"=== {exp} seed {seed} reference {name} {tag}")
            model = fresh(name, f)
            relearned = _relearn_and_record(
                ctx, model, split, ev, x_f, y_f, seed=derive_seed("relearn-ref", exp, seed, name, f),
                freeze_bn=ctx.cfg["rl"].freeze_bn and name != "scratch", targets=t_f)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict_fp16(model), out / "model_fp16.pt")
            record = ctx.base_record(stage="references", exp=exp, seed=seed, kind=split.kind, pair=split.pair,
                                     model=name, fraction=f, n_f=len(forget_idx), theory=ctx.args.theory,
                                     n_rows=len(x_f), **relearned)
            write_json(out / "result.json", record)


def stage_represent(ctx: Context) -> None:
    """CKA (to Retrain and Pretrained) and k-NN accuracy of every saved model of one
    experiment and seed: the references before relearning, every unlearned model
    before and after relearning, and every relearned reference. Writes
    representation.json next to each checkpoint; existing files are skipped."""
    exp, seed = ctx.args.exp, ctx.args.seed
    rep = ctx.cfg["rep"]
    split = ctx.split(exp, seed)

    def images(idx, train: bool):
        if train:
            return ctx.train_subset(idx)
        index = torch.as_tensor(idx, dtype=torch.int64)
        return normalize(ctx.dataset.x_test[index], ctx.device), ctx.dataset.y_test[index].to(ctx.device)

    rs = RepSets(*images(split.test_forget, False), *images(split.test_retain, False),
                 *images(split.probe, True), *images(split.retain_order[:rep.max_support_retain], True))
    pretrained = ctx.load_model(ctx.pretrain_dir(seed, read=True) / "model.pt")
    retrain = ctx.load_model(ctx.retrain_dir(exp, seed) / "model.pt")
    refs = {"retrain": reference_features(retrain, rs), "pretrained": reference_features(pretrained, rs)}

    def measure(model) -> Dict:
        return representation_metrics(model, rs, refs, split.pair, rep.knn_k)

    def load_fp16(path: Path):
        model = ctx.new_model(0)
        return load_state(model, torch.load(path, map_location=ctx.device))

    base = ctx.references_dir(exp, seed)
    before_path = base / "representation_before.json"
    if not _done(before_path, ctx, "represent"):
        scratch = ctx.new_model(derive_seed("scratch-init", exp, seed, None))
        before = {"pretrained": measure(pretrained), "retrain": measure(retrain), "scratch": measure(scratch)}
        write_json(before_path, ctx.base_record(stage="represent", exp=exp, seed=seed, metrics=before))

    unlearn_root = ctx.out / "unlearn" / f"exp={exp}" / f"seed={seed}"
    for folder in sorted(p.parent for p in unlearn_root.glob("mu_f=*/rescale=*/f=*/result.json")):
        out = folder / "representation.json"
        if _done(out, ctx, "represent"):
            continue
        result = read_json(folder / "result.json")
        record = {"after_unlearn": measure(load_fp16(folder / "model_fp16.pt"))}
        if (folder / "model_relearned_fp16.pt").exists():
            record["after_relearn"] = measure(load_fp16(folder / "model_relearned_fp16.pt"))
        write_json(out, ctx.base_record(stage="represent", exp=exp, seed=seed, source="unlearned",
                                        mu_f=result["mu_f"], rescale=result["rescale"]["tag"],
                                        fraction=result["fraction"], **record))
        log(f"wrote {out}")

    for folder in sorted(p.parent for p in base.glob("model=*/f=*/model_fp16.pt")):
        out = folder / "representation.json"
        if _done(out, ctx, "represent"):
            continue
        result = read_json(folder / "result.json")
        write_json(out, ctx.base_record(stage="represent", exp=exp, seed=seed, source=result["model"],
                                        fraction=result["fraction"],
                                        after_relearn=measure(load_fp16(folder / "model_fp16.pt"))))
        log(f"wrote {out}")


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--exp", type=str, default=None, help='"random" or "pair-A-B"')
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--forget_sizes", type=int, nargs="+", default=None,
                        help="unlearn_relearn / references: use the first n images of the forget pool (all "
                             "of the retain set when unlearning) for each n, instead of --fractions")
    parser.add_argument("--pt_sizes", type=int, nargs="+", default=None,
                        help="pretrain_curve: train on this many images (0 = the untrained network) instead of "
                             "--fractions of the training set")
    parser.add_argument("--pt_min_steps", type=int, default=0,
                        help="pretrain_curve: raise the epoch count until a run has at least this many steps, so "
                             "that small training sets also train to convergence")
    parser.add_argument("--match_retain", action="store_true",
                        help="unlearn_relearn with --forget_sizes: also cut the retain set to the first n images, "
                             "so n retain and n forget images are used (folders m=<n>)")
    parser.add_argument("--augment", action="store_true",
                        help="random crop (4 px padding) and horizontal flip on every training minibatch, in all "
                             "training stages; evaluation is never augmented. Use a separate --out_root")
    parser.add_argument("--theory", action="store_true",
                        help="match the diagonal-network theory (needs --forget_sizes): stage 2 on n rows drawn from all "
                             "classes with both loss terms on the same rows; stage 3 the forget-only attack on n fresh "
                             "rows (mse: squared loss to onehot on forget rows, zero on retain rows; ce: KL to onehot "
                             "on forget rows, uniform on retain rows); folders t=<n>")
    parser.add_argument("--ul_from", type=int, default=None,
                        help="relearn_sweep: the number of unlearning rows of the unlearned model to relearn")
    parser.add_argument("--n_pairs", type=int, default=3)
    parser.add_argument("--loss", choices=LOSS_KINDS, default="ce")
    parser.add_argument("--out_root", type=str, default=None,
                        help="default: data/cifar_unlearning/<loss>")
    parser.add_argument("--pretrain_root", type=str, default=None,
                        help="load Pretrained from here instead of --out_root (for calibration runs)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_config_args(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    train_module.set_augment(args.augment)
    if args.theory and (not args.forget_sizes or args.match_retain):
        raise SystemExit("--theory needs --forget_sizes and excludes --match_retain")
    if args.stage in ("retrain", "unlearn_relearn", "relearn_sweep", "references", "reeval", "represent"):
        if args.exp is None:
            raise SystemExit(f"--exp is required for {args.stage}")
        parse_experiment(args.exp)  # validate early
    ctx = Context(args)
    log(f"stage {args.stage} on {ctx.device}, out_root {ctx.out}")
    {
        "download": stage_download,
        "pretrain": stage_pretrain,
        "pretrain_curve": stage_pretrain_curve,
        "select_pairs": stage_select_pairs,
        "retrain": stage_retrain,
        "unlearn_relearn": stage_unlearn_relearn,
        "relearn_sweep": stage_relearn_sweep,
        "reeval": stage_reeval,
        "references": stage_references,
        "represent": stage_represent,
    }[args.stage](ctx)


if __name__ == "__main__":
    main()
