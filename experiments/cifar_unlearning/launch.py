"""SLURM array definitions built on the repository's `ArgparseArray`.

    python experiments/cifar_unlearning/launch.py <stage> --count       # number of array tasks
    python experiments/cifar_unlearning/launch.py <stage> <array_id>    # run one task

Any further flags are passed through to run.py (for example `--pt_epochs 30`).
Stages after select_pairs read the experiments from <out_root>/pilot_pairs.json
plus the random split; `--exps` overrides that list and `--all_pairs` uses all 45
pairs. Unlike `ArgparseArray.call_script`, the task runs in-process, so a failed
run gives a failed SLURM task and stops dependent jobs.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.array_training import ArgparseArray  # noqa: E402

from experiments.cifar_unlearning import run  # noqa: E402
from experiments.cifar_unlearning.config import DEFAULT_FRACTIONS, DEFAULT_MU_F, DEFAULT_SEEDS  # noqa: E402

ARRAY_STAGES = ("pretrain", "select_pairs", "retrain", "unlearn_relearn", "references")


def experiments(out_root: Path, exps: Optional[List[str]], all_pairs: bool, num_classes: int = 10) -> List[str]:
    if exps:
        return list(exps)
    if all_pairs:
        pairs = [f"pair-{a}-{b}" for a, b in itertools.combinations(range(num_classes), 2)]
    else:
        path = out_root / "pilot_pairs.json"
        if not path.exists():
            raise SystemExit(f"{path} not found: run the select_pairs stage first, or pass --exps")
        pairs = run.read_json(path)["pairs"]
    return pairs + ["random"]


def build_array(stage: str, exps: List[str]) -> ArgparseArray:
    seeds = list(DEFAULT_SEEDS)
    fractions = list(DEFAULT_FRACTIONS)
    if stage == "pretrain":
        return ArgparseArray(posarg0="pretrain", seed=seeds)
    if stage == "select_pairs":
        return ArgparseArray(posarg0="select_pairs", niarg_seeds=seeds)
    if stage == "retrain":
        return ArgparseArray(posarg0="retrain", exp=exps, seed=seeds)
    if stage == "unlearn_relearn":
        return ArgparseArray(posarg0="unlearn_relearn", exp=exps, seed=seeds, ul_mu_f=list(DEFAULT_MU_F),
                             niarg_fractions=fractions)
    if stage == "references":
        return ArgparseArray(posarg0="references", exp=exps, seed=seeds, niarg_fractions=fractions)
    raise ValueError(f"unknown stage {stage!r}")


def task_count(array: ArgparseArray) -> int:
    return len(list(itertools.product(*array.array_args.values())))


def to_argv(args: Dict) -> List[str]:
    """Same conventions as ArgparseArray.call_script: positionals first, then flags."""
    argv = [str(args[k]) for k in sorted(k for k in args if k.startswith("posarg"))]
    for key, value in args.items():
        if key.startswith("posarg"):
            continue
        if isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            argv += [f"--{key}"] + [str(v) for v in value]
        else:
            argv += [f"--{key}", str(value)]
    return argv


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=ARRAY_STAGES)
    parser.add_argument("array_id", type=int, nargs="?")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--exps", nargs="+", default=None)
    parser.add_argument("--all_pairs", action="store_true")
    args, passthrough = parser.parse_known_args(argv)
    out_root = Path(args.out_root or f"data/cifar_unlearning/{args.loss}")

    needs_exps = args.stage in ("retrain", "unlearn_relearn", "references")
    exps = experiments(out_root, args.exps, args.all_pairs) if needs_exps else []
    array = build_array(args.stage, exps)
    if args.count:
        print(task_count(array))
        return
    if args.array_id is None:
        raise SystemExit("give an array_id or --count")
    n = task_count(array)
    if not 0 <= args.array_id < n:
        raise SystemExit(f"array_id {args.array_id} out of range for {n} tasks")
    run_argv = to_argv(array.get_args(args.array_id)) + ["--loss", args.loss, "--out_root", str(out_root)]
    run.main(run_argv + passthrough)


if __name__ == "__main__":
    main()
