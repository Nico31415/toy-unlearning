"""Entry point for every stage. Run from the repository root:

    python experiments/cifar_unlearning/run.py <stage> [flags]

Stages, in order:
    download         fetch CIFAR-10 once and cache it as uint8 tensors (login node)
    pretrain         --seed S                      train on all of D
    select_pairs     --seeds 0 1 2 3 4             pick pilot pairs from pretrained confusion
    retrain          --exp E --seed S              the control: train on all of D_r
    unlearn_relearn  --exp E --seed S --ul_mu_f M  unlearn Pretrained at every fraction, then relearn
    references       --exp E --seed S              relearn Pretrained, Retrain and Scratch at every fraction

E is "random" or "pair-A-B". Every run writes result.json (plus a checkpoint where
needed) under --out_root and is skipped when its result.json exists, so a SLURM
array can be resubmitted safely. A run that fails a check writes failed.json and
exits non-zero, which stops dependent SLURM jobs.
"""
from __future__ import annotations

import argparse
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
    Dataset, Split, load_cifar10, load_split, make_split, make_synthetic, normalize, parse_experiment,
    save_split)
from experiments.cifar_unlearning.losses import LOSS_KINDS  # noqa: E402
from experiments.cifar_unlearning.metrics import EvalSets, confusion_matrix, evaluate  # noqa: E402
from experiments.cifar_unlearning.models import (  # noqa: E402
    build_model, is_identity_rescale, load_state, mean_logit_norm, recalibrate_bn, rescale_,
    state_dict_fp16, weight_norms)
from experiments.cifar_unlearning.train import (  # noqa: E402
    relearn, train_from_scratch, unlearn)

log = partial(print, flush=True)
STAGES = ("download", "pretrain", "select_pairs", "retrain", "unlearn_relearn", "references")
REFERENCE_MODELS = ("pretrained", "retrain", "scratch")


# --------------------------------------------------------------------------- helpers

def derive_seed(*parts) -> int:
    """Stable 31-bit seed from any tuple of labels."""
    digest = hashlib.sha256(repr(parts).encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def fmt(x: float) -> str:
    return f"{x:g}"


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
        return {**fields, "loss": self.loss, "configs": configs_to_dict(self.cfg)}

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

    def unlearn_dir(self, exp: str, seed: int, mu_f: float, fraction: float) -> Path:
        return (self.out / "unlearn" / f"exp={exp}" / f"seed={seed}" / f"mu_f={fmt(mu_f)}"
                / f"rescale={self.rescale_tag()}" / f"f={fmt(fraction)}")

    def references_dir(self, exp: str, seed: int) -> Path:
        return self.out / "references" / f"exp={exp}" / f"seed={seed}"


# Config groups that determine each stage's result. A finished result is only reused
# when these match the current flags; otherwise the run stops instead of silently
# mixing settings in one output folder.
STAGE_GROUPS = {
    "pretrain": ("data", "model", "pt"),
    "retrain": ("data", "model", "pt"),
    "unlearn_relearn": ("data", "model", "ul", "rl", "rs"),
    "references": ("data", "model", "rl"),
}


def _done(path: Path, ctx: "Context", stage: str) -> bool:
    if not path.exists() or ctx.args.overwrite:
        return False
    saved = read_json(path)
    current = configs_to_dict(ctx.cfg)
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
                        freeze_bn: bool) -> Dict:
    result = relearn(model, x_f, y_f, ctx.cfg["rl"], seed=seed, freeze_bn=freeze_bn, pair=split.pair,
                     loss_kind=ctx.loss, eval_fn=lambda m: evaluate(m, ev, split.pair), log=log)
    return {"relearn": result, "after_relearn": evaluate(model, ev, split.pair)}


def stage_unlearn_relearn(ctx: Context) -> None:
    exp, seed = ctx.args.exp, ctx.args.seed
    ul, rs = ctx.cfg["ul"], ctx.cfg["rs"]
    todo = [f for f in sorted(ctx.args.fractions)
            if not _done(ctx.unlearn_dir(exp, seed, ul.mu_f, f) / "result.json", ctx, "unlearn_relearn")]
    if not todo:
        return
    pretrained_state = torch.load(ctx.pretrain_dir(seed, read=True) / "model.pt", map_location=ctx.device)
    split = ctx.split(exp, seed)
    ev = ctx.eval_sets(split)
    for f in todo:
        out = ctx.unlearn_dir(exp, seed, ul.mu_f, f)
        log(f"=== {exp} seed {seed} mu_f {fmt(ul.mu_f)} rescale {ctx.rescale_tag()} f {fmt(f)}")
        retain_idx, forget_idx = split.retain_subset(f), split.forget_subset(f)
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

        ul_result = unlearn(model, x_r, y_r, x_f, y_f, ul, seed=derive_seed("unlearn", exp, seed, ul.mu_f, f),
                            loss_kind=ctx.loss, log=log)
        after_unlearn = evaluate(model, ev, split.pair)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict_fp16(model), out / "model_fp16.pt")

        relearned = _relearn_and_record(ctx, model, split, ev, x_f, y_f,
                                        seed=derive_seed("relearn", exp, seed, ul.mu_f, f),
                                        freeze_bn=ctx.cfg["rl"].freeze_bn)
        record = ctx.base_record(stage="unlearn_relearn", exp=exp, seed=seed, kind=split.kind, pair=split.pair,
                                 fraction=f, n_r=len(retain_idx), n_f=len(forget_idx), mu_r=ul.mu_r,
                                 mu_f=ul.mu_f, rescale=rescale_info, unlearn=ul_result,
                                 after_unlearn=after_unlearn, **relearned)
        write_json(out / "result.json", record)


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

    for f in sorted(ctx.args.fractions):
        forget_idx = split.forget_subset(f)
        x_f, y_f = ctx.train_subset(forget_idx)
        for name in REFERENCE_MODELS:
            out = base / f"model={name}" / f"f={fmt(f)}"
            if _done(out / "result.json", ctx, "references"):
                continue
            log(f"=== {exp} seed {seed} reference {name} f {fmt(f)}")
            model = fresh(name, f)
            relearned = _relearn_and_record(
                ctx, model, split, ev, x_f, y_f, seed=derive_seed("relearn-ref", exp, seed, name, f),
                freeze_bn=ctx.cfg["rl"].freeze_bn and name != "scratch")
            record = ctx.base_record(stage="references", exp=exp, seed=seed, kind=split.kind, pair=split.pair,
                                     model=name, fraction=f, n_f=len(forget_idx), **relearned)
            write_json(out / "result.json", record)


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--exp", type=str, default=None, help='"random" or "pair-A-B"')
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
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
    if args.stage in ("retrain", "unlearn_relearn", "references"):
        if args.exp is None:
            raise SystemExit(f"--exp is required for {args.stage}")
        parse_experiment(args.exp)  # validate early
    ctx = Context(args)
    log(f"stage {args.stage} on {ctx.device}, out_root {ctx.out}")
    {
        "download": stage_download,
        "pretrain": stage_pretrain,
        "select_pairs": stage_select_pairs,
        "retrain": stage_retrain,
        "unlearn_relearn": stage_unlearn_relearn,
        "references": stage_references,
    }[args.stage](ctx)


if __name__ == "__main__":
    main()
