"""End-to-end smoke tests on synthetic data: every stage, aggregation, plots, launch mapping."""
import json

import pandas as pd
import pytest

from experiments.cifar_unlearning import aggregate, launch, plot, run

SYN = ["--synthetic", "true", "--arch", "tiny", "--synthetic_train_per_class", "40",
       "--synthetic_test_per_class", "20", "--n_probe", "20", "--n_forget", "100",
       "--pt_epochs", "3", "--pt_lr", "0.05", "--pt_batch_size", "64", "--pt_max_final_train_loss", "10",
       "--ul_max_steps", "40", "--ul_eval_every", "10", "--ul_batch_size", "32",
       "--rl_max_steps", "40", "--rl_eval_every", "10", "--rl_batch_size", "32",
       "--device", "cpu", "--fractions", "0.1", "1.0"]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("pipeline")
    out = root / "out"
    common = SYN + ["--out_root", str(out)]
    for seed in ("0", "1"):
        run.main(["pretrain", "--seed", seed] + common)
    run.main(["select_pairs", "--seeds", "0", "1", "--n_pairs", "2"] + common)
    pairs = json.loads((out / "pilot_pairs.json").read_text())["pairs"]
    exps = [pairs[0], "random"]
    for exp in exps:
        run.main(["retrain", "--exp", exp, "--seed", "0"] + common)
        run.main(["unlearn_relearn", "--exp", exp, "--seed", "0", "--ul_mu_f", "0.5"] + common)
        run.main(["references", "--exp", exp, "--seed", "0"] + common)
    run.main(["unlearn_relearn", "--exp", pairs[0], "--seed", "0", "--ul_mu_f", "0.5", "--rs_mode", "body_head",
              "--rs_alpha_head", "0.5", "--rs_recalibrate_bn", "true"] + common)
    run.main(["pretrain", "--seed", "0"] + common)  # already done: must be skipped
    csv_dir, fig_dir = root / "csv", root / "fig"
    aggregate.main(["--out_root", str(out), "--dest", str(csv_dir)])
    plot.main(["--src", str(csv_dir), "--dest", str(fig_dir)])
    return {"out": out, "csv": csv_dir, "fig": fig_dir, "pairs": pairs, "exps": exps}


def test_pair_selection(pipeline):
    ranking = json.loads((pipeline["out"] / "pilot_pairs.json").read_text())["ranking"]
    assert len(ranking) == 45
    assert [r["confusion"] for r in ranking] == sorted(r["confusion"] for r in ranking)
    assert pipeline["pairs"] == [ranking[0]["exp"], ranking[-1]["exp"]]


def test_tables(pipeline):
    csv = pipeline["csv"]
    unl = pd.read_csv(csv / "unlearn.csv")
    refs = pd.read_csv(csv / "references.csv")
    assert len(pd.read_csv(csv / "pretrain.csv")) == 2
    assert len(pd.read_csv(csv / "retrain.csv")) == 2
    assert len(unl) == 2 * 2 + 2  # 2 experiments x 2 fractions, plus the rescaled run
    assert set(unl["rescale"]) == {"none", "body_head-b1-h0.5-bnrecal"}
    assert len(refs) == 2 * 2 * 3
    assert set(refs["model"]) == {"pretrained", "retrain", "scratch"}
    for col in [c for c in unl.columns if c.endswith(("A_r", "A_f2", "probe_acc"))]:
        assert unl[col].dropna().between(0, 1).all()
    pair_rows = unl[unl["kind"] == "class"]
    assert pair_rows["ul_A_f2"].notna().all() and unl[unl["kind"] == "random"]["ul_A_f2"].isna().all()
    assert set(unl["n_f"]) == {6, 60, 8, 80}  # class pool 60, random pool 80
    curves = pd.read_csv(csv / "curves.csv")
    assert (curves["step"] == 0).any()


def test_figures(pipeline):
    for exp in pipeline["exps"]:
        assert (pipeline["fig"] / f"after_unlearning_{exp}.png").exists()
        assert (pipeline["fig"] / f"after_relearning_{exp}.png").exists()


def test_failed_check_exits_nonzero(tmp_path):
    with pytest.raises(SystemExit) as err:
        run.main(["pretrain", "--seed", "0", "--out_root", str(tmp_path)]
                 + SYN + ["--pt_max_final_train_loss", "0", "--pt_epochs", "1"])
    assert err.value.code == 1
    assert (tmp_path / "pretrain" / "seed=0" / "failed.json").exists()
    assert not (tmp_path / "pretrain" / "seed=0" / "result.json").exists()


def test_launch_arrays(tmp_path):
    (tmp_path / "pilot_pairs.json").write_text(json.dumps({"pairs": ["pair-3-5", "pair-0-8"]}))
    exps = launch.experiments(tmp_path, None, False)
    assert exps == ["pair-3-5", "pair-0-8", "random"]
    counts = {stage: launch.task_count(launch.build_array(stage, exps)) for stage in launch.ARRAY_STAGES}
    assert counts == {"pretrain": 5, "select_pairs": 1, "retrain": 15, "unlearn_relearn": 45, "references": 15}
    argv = launch.to_argv(launch.build_array("unlearn_relearn", exps).get_args(0))
    assert argv[0] == "unlearn_relearn"
    assert argv[argv.index("--exp") + 1] == "pair-3-5" and argv[argv.index("--ul_mu_f") + 1] == "0.25"
    assert argv[argv.index("--fractions") + 1:argv.index("--fractions") + 6] == ["0.01", "0.03", "0.1", "0.3", "1.0"]
    run.build_parser().parse_args(argv)  # every generated flag is accepted by run.py
    assert len(launch.experiments(tmp_path, None, True)) == 46


def test_resnet18_runs_every_stage(tmp_path):
    """The real architecture through every stage, at toy size."""
    common = list(SYN)
    common[common.index("--arch") + 1] = "resnet18"
    common[common.index("--synthetic_train_per_class") + 1] = "12"
    common[common.index("--synthetic_test_per_class") + 1] = "4"
    common[common.index("--n_probe") + 1] = "4"
    common[common.index("--n_forget") + 1] = "20"
    common[common.index("--pt_epochs") + 1] = "1"
    for flag in ("--ul_max_steps", "--rl_max_steps"):
        common[common.index(flag) + 1] = "4"
    common += ["--out_root", str(tmp_path), "--fractions", "1.0"]
    run.main(["pretrain", "--seed", "0"] + common)
    for stage in ("retrain", "unlearn_relearn", "references"):
        run.main([stage, "--exp", "pair-0-1", "--seed", "0"] + common)
    result = json.loads((tmp_path / "unlearn" / "exp=pair-0-1" / "seed=0" / "mu_f=0.5" / "rescale=none" / "f=1"
                         / "result.json").read_text())
    assert result["n_f"] == 20 and result["n_r"] == 96
