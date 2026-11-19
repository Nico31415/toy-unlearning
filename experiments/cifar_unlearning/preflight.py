"""Checks run by round.sh before it submits anything. Each exits non-zero on failure.

    python experiments/cifar_unlearning/preflight.py imports
        Every third-party module imported anywhere in experiments/cifar_unlearning or functions/ imports in this Python.
        (A module imported inside a function only fails when that function runs: scipy did that in ph_e4.)

    python experiments/cifar_unlearning/preflight.py train <stage> <launch.py flags...>
        For a stage that trains from scratch (family, pretrain, retrain): task 0 of the array, with the round's exact
        flags, on real CIFAR, for warm-up + 300 steps (the lr schedule unchanged). Fails if the loss is non-finite or
        has not fallen below its starting value. Catches a network that diverges at the round's lr (plain VGG in ph_e4).
"""
from __future__ import annotations

import ast
import importlib
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL = {"experiments", "functions"}


def third_party_imports(paths):
    mods = {}
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for name in names:
                top = name.split(".")[0]
                sibling = path.parent / f"{top}.py"  # a module imported by bare name from the same folder
                if top in LOCAL or top in sys.stdlib_module_names or top == "__future__" or sibling.exists():
                    continue
                mods.setdefault(name, set()).add(str(path.relative_to(REPO_ROOT)))
    return mods


def local_closure():
    """experiments/cifar_unlearning's files plus every repo module they import, transitively."""
    todo = [p for p in (REPO_ROOT / "experiments/cifar_unlearning").glob("*.py")]
    seen = set()
    while todo:
        path = todo.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                if name.split(".")[0] in LOCAL:
                    base = REPO_ROOT.joinpath(*name.split("."))
                    todo += [base.with_suffix(".py"), base / "__init__.py"]
    return sorted(seen)


def check_imports() -> int:
    files = local_closure()
    missing = []
    for name, where in sorted(third_party_imports(files).items()):
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - any failure to import is a failure here
            missing.append(f"  {name} ({type(e).__name__}: {e}) used in {', '.join(sorted(where))}")
    if missing:
        print("preflight imports: FAILED, these do not import in " + sys.executable + ":\n" + "\n".join(missing))
        return 1
    print(f"preflight imports: ok ({len(files)} files)")
    return 0


def check_train(stage: str, launch_argv) -> int:
    import torch
    from experiments.cifar_unlearning import launch, run
    from experiments.cifar_unlearning.data import experiment_name  # noqa: F401  (import check)

    if stage not in ("family", "pretrain", "retrain"):
        print(f"preflight train: nothing to check for stage {stage}")
        return 0
    # Build task 0's run.py argv exactly as launch.py would.
    parser_argv = [stage, "0"] + list(launch_argv)
    captured = {}
    real_main = run.main
    run.main = lambda argv, check_only=False: captured.setdefault("argv", list(argv))
    try:
        launch.main(parser_argv)
    finally:
        run.main = real_main
    args = run.build_parser().parse_args(captured["argv"])
    run.train_module.set_augment(args.augment)
    ctx = run.Context(args)
    cfg = ctx.cfg["pt"]
    steps = (cfg.warmup_steps or 0) + 300
    cfg.max_steps = steps
    if stage == "family":
        split = ctx.split(args.exp, args.seed)
        idx = split.retain_order
    else:
        split = ctx.split(args.exp, args.seed) if args.exp else None
        idx = split.retain_order if split is not None else None
    if idx is None:
        print("preflight train: could not build a training set for this stage; skipped")
        return 0
    x, y = ctx.train_subset(idx)
    model = ctx.new_model(0)
    losses = []
    out = run.train_from_scratch(model, x, y, cfg, seed=0, loss_kind=ctx.loss,
                                 log=lambda s: losses.append(s))
    first = None
    for line in losses:
        if "running loss" in line:
            v = float(line.split("running loss")[1].split()[0])
            first = v if first is None else first
    final = run.train_module.full_loss(model, x[:4096], y[:4096], ctx.loss)
    ok = (not out.get("diverged")) and math.isfinite(final) and (first is None or final < first)
    print(f"preflight train {stage} ({args.arch if hasattr(args, 'arch') else ''}, device {ctx.device}): "
          f"{out['steps']} steps, first-epoch loss {first}, loss now {final:.4f} -> {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("imports", "train"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "imports":
        sys.exit(check_imports())
    sys.exit(check_train(sys.argv[2], sys.argv[3:]))
