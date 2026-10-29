"""Checks that the pipeline does what the experiment design says, not just that it runs."""
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.cifar_unlearning import launch, run
from experiments.cifar_unlearning.data import make_split
from experiments.cifar_unlearning.tests.test_pipeline import SYN

REPO = Path(__file__).resolve().parents[3]
PAIR = "pair-2-7"


def no_timing(obj):
    if isinstance(obj, dict):
        return {k: no_timing(v) for k, v in obj.items() if k != "seconds"}
    if isinstance(obj, list):
        return [no_timing(v) for v in obj]
    return obj


def common(out):
    return SYN + ["--out_root", str(out)]


def unlearn_result(out, exp=PAIR, f="1", mu="0.5", rescale="none"):
    return json.loads((Path(out) / "unlearn" / f"exp={exp}" / "seed=0" / f"mu_f={mu}" / f"rescale={rescale}"
                       / f"f={f}" / "result.json").read_text())


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    out = tmp_path_factory.mktemp("pre")
    run.main(["pretrain", "--seed", "0"] + common(out))
    return out


def fresh_root(tmp_path, pretrained):
    """A new out_root that reuses the module's pretrained model."""
    shutil.copytree(pretrained / "pretrain", tmp_path / "pretrain")
    return tmp_path


def test_runs_are_reproducible(tmp_path, pretrained):
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        out.mkdir()
        fresh_root(out, pretrained)
        run.main(["unlearn_relearn", "--exp", PAIR, "--seed", "0"] + common(out))
    ra, rb = unlearn_result(a), unlearn_result(b)
    for key in ("unlearn", "after_unlearn", "after_relearn"):
        assert json.dumps(no_timing(ra[key]), sort_keys=True) == json.dumps(no_timing(rb[key]), sort_keys=True)

    run.main(["pretrain", "--seed", "1"] + common(a))
    m0 = torch.load(a / "pretrain" / "seed=0" / "model.pt")
    m1 = torch.load(a / "pretrain" / "seed=1" / "model.pt")
    assert not torch.equal(m0["fc.weight"], m1["fc.weight"])


def test_frozen_batchnorm_statistics_survive_unlearning(tmp_path, pretrained):
    out = fresh_root(tmp_path, pretrained)
    run.main(["unlearn_relearn", "--exp", PAIR, "--seed", "0"] + common(out))
    before = torch.load(out / "pretrain" / "seed=0" / "model.pt")
    after = torch.load(out / "unlearn" / f"exp={PAIR}" / "seed=0" / "mu_f=0.5" / "rescale=none" / "f=1"
                       / "model_fp16.pt")
    stats = [k for k in before if k.endswith(("running_mean", "running_var"))]
    assert stats
    for k in stats:
        assert torch.allclose(after[k].float(), before[k], rtol=2e-3, atol=1e-3), k
    assert not torch.allclose(after["fc.weight"].float(), before["fc.weight"], atol=1e-3)


def test_data_routing(tmp_path, pretrained, monkeypatch):
    """Retrain sees exactly D_r; unlearning sees the f-subsets; relearning (of the
    unlearned model and of every reference) sees exactly the same forget images."""
    out = fresh_root(tmp_path, pretrained)
    seen = {"scratch_train": [], "unlearn": [], "relearn": []}
    real_train, real_unlearn, real_relearn = run.train_from_scratch, run.unlearn, run.relearn

    def spy_train(model, x, y, *a, **k):
        seen["scratch_train"].append(x.clone())
        return real_train(model, x, y, *a, **k)

    def spy_unlearn(model, x_r, y_r, x_f, y_f, *a, **k):
        seen["unlearn"].append((x_r.clone(), x_f.clone()))
        return real_unlearn(model, x_r, y_r, x_f, y_f, *a, **k)

    def spy_relearn(model, x, y, *a, **k):
        seen["relearn"].append(x.clone())
        return real_relearn(model, x, y, *a, **k)

    monkeypatch.setattr(run, "train_from_scratch", spy_train)
    monkeypatch.setattr(run, "unlearn", spy_unlearn)
    monkeypatch.setattr(run, "relearn", spy_relearn)

    args = common(out) + ["--fractions", "0.5", "1.0"]
    run.main(["retrain", "--exp", PAIR, "--seed", "0"] + args)
    run.main(["unlearn_relearn", "--exp", PAIR, "--seed", "0"] + args)
    run.main(["references", "--exp", PAIR, "--seed", "0"] + args)

    ctx = run.Context(run.build_parser().parse_args(["retrain", "--exp", PAIR] + args))
    split = ctx.split(PAIR, 0)
    x_all = ctx.x_train
    rows = lambda idx: x_all[torch.as_tensor(idx)]  # noqa: E731

    assert len(seen["scratch_train"]) == 1
    assert torch.equal(seen["scratch_train"][0], rows(split.retain_order))
    forbidden = set(split.forget_all.tolist())
    assert not forbidden & set(split.retain_order.tolist())

    assert len(seen["unlearn"]) == 2 and len(seen["relearn"]) == 2 + 2 * 3
    for i, f in enumerate((0.5, 1.0)):
        x_r, x_f = seen["unlearn"][i]
        assert torch.equal(x_r, rows(split.retain_subset(f)))
        assert torch.equal(x_f, rows(split.forget_subset(f)))
        assert not set(split.forget_subset(f).tolist()) & set(split.probe.tolist())
        assert torch.equal(seen["relearn"][i], x_f)  # relearned on the images it unlearned
        for ref in range(3):  # pretrained, retrain, scratch at this fraction
            assert torch.equal(seen["relearn"][2 + 3 * i + ref], x_f)


def test_partial_resubmission_recomputes_only_missing(tmp_path, pretrained):
    out = fresh_root(tmp_path, pretrained)
    args = ["unlearn_relearn", "--exp", PAIR, "--seed", "0"] + common(out)
    run.main(args)
    base = out / "unlearn" / f"exp={PAIR}" / "seed=0" / "mu_f=0.5" / "rescale=none"
    kept = base / "f=0.1" / "result.json"
    stamp = kept.stat().st_mtime_ns
    first = json.loads((base / "f=1" / "result.json").read_text())
    (base / "f=1" / "result.json").unlink()
    run.main(args)
    assert kept.stat().st_mtime_ns == stamp
    again = json.loads((base / "f=1" / "result.json").read_text())
    assert json.dumps(again["after_relearn"], sort_keys=True) == json.dumps(first["after_relearn"], sort_keys=True)


def test_split_is_saved_and_guarded(tmp_path):
    args = run.build_parser().parse_args(["retrain", "--exp", "random"] + common(tmp_path))
    ctx = run.Context(args)
    first = ctx.split("random", 3)
    saved = tmp_path / "splits" / "exp=random" / "seed=3.npz"
    assert saved.exists()
    again = run.Context(args).split("random", 3)
    for name in ("forget_all", "probe", "forget_pool_order", "retain_order", "test_retain", "test_forget"):
        assert np.array_equal(getattr(first, name), getattr(again, name))
    assert again.kind == "random" and again.pair is None
    fresh = make_split(ctx.dataset, "random", 3, n_forget=100, n_probe=20)
    assert np.array_equal(fresh.retain_order, first.retain_order)
    changed = run.build_parser().parse_args(["retrain", "--exp", "random"] + common(tmp_path) + ["--n_probe", "10"])
    with pytest.raises(SystemExit):
        run.Context(changed).split("random", 3)
    pair_ctx = run.Context(run.build_parser().parse_args(["retrain", "--exp", PAIR] + common(tmp_path)))
    assert pair_ctx.split(PAIR, 0).pair == (2, 7) and pair_ctx.split(PAIR, 0).pair == (2, 7)


def test_failure_propagates_through_launch(tmp_path):
    with pytest.raises(SystemExit) as err:
        launch.main(["pretrain", "0", "--out_root", str(tmp_path)] + SYN
                    + ["--pt_max_final_train_loss", "0", "--pt_epochs", "1"])
    assert err.value.code == 1


def _python_shim(tmp_path):
    """A `python` on PATH that is the interpreter running the tests."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return bindir


def test_launch_sh_end_to_end(tmp_path):
    bindir = _python_shim(tmp_path)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "SLURM_ARRAY_TASK_ID": "0",
           "REQUIRE_GPU": "0", "SLURM_SUBMIT_DIR": str(REPO)}
    script = REPO / "experiments" / "cifar_unlearning" / "launch.sh"
    ok = subprocess.run(["bash", str(script), "pretrain", "--out_root", str(tmp_path / "out")] + SYN,
                        cwd=REPO, env=env, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr[-2000:]
    assert (tmp_path / "out" / "pretrain" / "seed=0" / "result.json").exists()
    bad = subprocess.run(["bash", str(script), "pretrain", "--out_root", str(tmp_path / "bad")] + SYN
                         + ["--pt_max_final_train_loss", "0", "--pt_epochs", "1"],
                         cwd=REPO, env=env, capture_output=True, text=True)
    assert bad.returncode != 0
    if not torch.cuda.is_available():  # the GPU check must stop a task that has no usable GPU
        gpu = subprocess.run(["bash", str(script), "pretrain", "--out_root", str(tmp_path / "gpu")] + SYN,
                             cwd=REPO, env={**env, "REQUIRE_GPU": "1"}, capture_output=True, text=True)
        assert gpu.returncode != 0 and "cannot see a GPU" in gpu.stderr
        assert not (tmp_path / "gpu").exists()
    no_id = subprocess.run(["bash", str(script), "pretrain"], cwd=REPO,
                           env={k: v for k, v in env.items() if k != "SLURM_ARRAY_TASK_ID"},
                           capture_output=True, text=True)
    assert no_id.returncode != 0


def test_readme_submit_block(tmp_path):
    """Run the README's submit commands with a fake sbatch and check the arrays requested."""
    readme = (REPO / "experiments" / "cifar_unlearning" / "README.md").read_text()
    block = re.search(r"```bash\n(mkdir -p logs.*?)```", readme, re.S).group(1)
    block = "\n".join(line for line in block.splitlines() if "run.py download" not in line)
    work = tmp_path / "repo"
    shutil.copytree(REPO / "experiments" / "cifar_unlearning", work / "experiments" / "cifar_unlearning",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "experiments" / "__init__.py", work / "experiments" / "__init__.py")
    shutil.copytree(REPO / "functions", work / "functions", ignore=shutil.ignore_patterns("__pycache__"))
    pairs = work / "data" / "cifar_unlearning" / "ce" / "pilot_pairs.json"
    pairs.parent.mkdir(parents=True)
    pairs.write_text(json.dumps({"pairs": ["pair-3-5", "pair-0-8", "pair-1-9"]}))
    bindir = _python_shim(tmp_path)
    calls = tmp_path / "sbatch_calls.txt"
    sbatch = bindir / "sbatch"
    sbatch.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\necho 1234\n')
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "PY": sys.executable}
    done = subprocess.run(["bash", "-euo", "pipefail", "-c", block], cwd=work, env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 5
    arrays = [re.search(r"--array=(\S+)", line).group(1) for line in lines]
    assert arrays == ["0-4", "0-0", "0-19", "0-59", "0-19"]  # 4 experiments x 5 seeds (x 3 mu_f)
    assert "--dependency=afterok:1234" in lines[1] and "--dependency=afterok:1234" in lines[4]
    assert [line.split()[-1] for line in lines] == ["pretrain", "select_pairs", "retrain", "unlearn_relearn",
                                                    "references"]


def test_changed_hyperparameters_are_not_silently_skipped(tmp_path, pretrained):
    out = fresh_root(tmp_path, pretrained)
    args = ["unlearn_relearn", "--exp", PAIR, "--seed", "0", "--fractions", "1.0"] + common(out)
    run.main(args)
    run.main(args)  # same settings: skipped
    with pytest.raises(SystemExit, match="different 'ul' settings"):
        run.main(args + ["--ul_lr", "0.123"])
    run.main(["retrain", "--exp", PAIR, "--seed", "0"] + common(out))
    run.main(["references", "--exp", PAIR, "--seed", "0", "--fractions", "1.0"] + common(out))
    run.main(["references", "--exp", PAIR, "--seed", "0", "--fractions", "1.0"] + common(out)
             + ["--ul_lr", "0.123"])  # unlearning settings do not affect the references
    with pytest.raises(SystemExit, match="different 'rl' settings"):
        run.main(["references", "--exp", PAIR, "--seed", "0", "--fractions", "1.0"] + common(out)
                 + ["--rl_lr", "0.123"])


def test_pretrain_root_for_calibration(tmp_path, pretrained):
    calib = tmp_path / "calib"
    run.main(["unlearn_relearn", "--exp", PAIR, "--seed", "0", "--fractions", "1.0", "--ul_lr", "0.002"]
             + common(calib) + ["--pretrain_root", str(pretrained)])
    assert not (calib / "pretrain").exists()
    result = unlearn_result(calib)
    assert result["configs"]["ul"]["lr"] == 0.002
