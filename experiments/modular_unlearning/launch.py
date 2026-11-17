"""SLURM arrays for the modular-arithmetic stages (same conventions as experiments/cifar_unlearning/launch.py).

    python experiments/modular_unlearning/launch.py <stage> --count --exps ... [--seeds ...] [--fam_Ls ...] [--Lfulls ...]
    python experiments/modular_unlearning/launch.py <stage> <array_id> --exps ... [same] [flags passed to run.py]

family: one task per (exp, seed, L); phase: one per (exp, seed, L_full, method), looping over --nS, --targets, --arms
and --S inside the task; phase_relearn: one per (exp, seed, L_full).
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functions.array_training import ArgparseArray  # noqa: E402

from experiments.cifar_unlearning.launch import to_argv  # noqa: E402
from experiments.modular_unlearning import run  # noqa: E402


def build_array(stage, exps, seeds, fam_Ls=None, Lfulls=None, methods=None) -> ArgparseArray:
    if stage == "family":
        return ArgparseArray(posarg0="family", exp=list(exps), seed=list(seeds), fam_L=list(fam_Ls or [0]))
    if stage == "phase":
        return ArgparseArray(posarg0="phase", exp=list(exps), seed=list(seeds), Lfull=list(Lfulls or [1800]),
                             methods=list(methods or ["F"]))
    if stage == "phase_relearn":
        return ArgparseArray(posarg0="phase_relearn", exp=list(exps), seed=list(seeds), Lfull=list(Lfulls or [1800]))
    raise ValueError(stage)


def task_count(array) -> int:
    return len(list(itertools.product(*array.array_args.values())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=run.STAGES)
    ap.add_argument("array_id", type=int, nargs="?")
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fam_Ls", type=int, nargs="+", default=None)
    ap.add_argument("--Lfulls", type=int, nargs="+", default=None)
    ap.add_argument("--methods_array", nargs="+", default=None, help="phase: one task per method")
    args, passthrough = ap.parse_known_args(argv)
    array = build_array(args.stage, args.exps, args.seeds, args.fam_Ls, args.Lfulls, args.methods_array)
    n = task_count(array)
    if args.count:
        print(n)
        return
    if args.array_id is None or not 0 <= args.array_id < n:
        raise SystemExit(f"array_id must be in [0, {n})")
    run.main(to_argv(array.get_args(args.array_id)) + passthrough)


if __name__ == "__main__":
    main()
