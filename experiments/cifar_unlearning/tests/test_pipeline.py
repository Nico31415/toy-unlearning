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
    run.main(["pretrain_curve", "--seed", "0"] + common)
    run.main(["pretrain_curve", "--seed", "0"] + common)  # second call skips everything
    run.main(["select_pairs", "--seeds", "0", "1", "--n_pairs", "2"] + common)
    pairs = json.loads((out / "pilot_pairs.json").read_text())["pairs"]
    exps = [pairs[0], "random"]
    for exp in exps:
        run.main(["retrain", "--exp", exp, "--seed", "0"] + common)
        run.main(["unlearn_relearn", "--exp", exp, "--seed", "0", "--ul_mu_f", "0.5"] + common)
        run.main(["references", "--exp", exp, "--seed", "0"] + common)
    run.main(["unlearn_relearn", "--exp", pairs[0], "--seed", "0", "--ul_mu_f", "0.5", "--rs_mode", "body_head",
              "--rs_alpha_head", "0.5", "--rs_recalibrate_bn", "true"] + common)
    for exp in exps:
        run.main(["represent", "--exp", exp, "--seed", "0"] + common)
        run.main(["represent", "--exp", exp, "--seed", "0"] + common)  # second call skips everything
    run.main(["pretrain", "--seed", "0"] + common)  # already done: must be skipped
    csv_dir, fig_dir = root / "csv", root / "fig"
    aggregate.main(["--out_root", str(out), "--dest", str(csv_dir)])
    plot.main(["--src", str(csv_dir), "--dest", str(fig_dir)])
    return {"out": out, "csv": csv_dir, "fig": fig_dir, "pairs": pairs, "exps": exps}


def test_sample_efficiency(pipeline):
    curve = pd.read_csv(pipeline["csv"] / "pretrain_curve.csv")
    assert list(curve["fraction"]) == [0.1, 1.0] and list(curve["n_train"]) == [40, 400]
    accs = curve[[f"acc_c{k}" for k in range(10)]]
    assert ((accs >= 0) & (accs <= 1)).all().all()
    pw = curve[[f"pair_{a}_{b}" for a in range(10) for b in range(10) if a != b]]
    assert ((pw >= 0) & (pw <= 1)).all().all() and len(pw.columns) == 90
    assert (curve["test_acc"] - accs.mean(axis=1)).abs().max() < 1e-6  # equal test images per class (test_acc is float32)
    assert (pipeline["fig"] / f"sample_efficiency_{pipeline['pairs'][0]}.png").exists()
    assert not (pipeline["fig"] / "sample_efficiency_random.png").exists()


def test_forget_sizes(tmp_path):
    """--forget_sizes: n forget images with the whole retain set, folders n=<size>, tables and the figure."""
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out)]
    run.main(["pretrain", "--seed", "0"] + common)
    run.main(["retrain", "--exp", "pair-0-1", "--seed", "0"] + common)
    sizes = ["--forget_sizes", "2", "6"]
    for _ in range(2):  # the second call skips everything
        run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", "0.5"] + sizes + common)
        run.main(["references", "--exp", "pair-0-1", "--seed", "0"] + sizes + common)
    assert sorted(p.name for p in out.glob("unlearn/exp=pair-0-1/seed=0/mu_f=0.5/rescale=none/*")) == ["n=2", "n=6"]
    aggregate.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    unl = pd.read_csv(tmp_path / "csv" / "unlearn.csv")
    assert list(unl["n_f"]) == [2, 6] and unl["retain_full"].all()
    assert (unl["n_r"] == 8 * 40).all()  # every kept-class training image, whatever n_f
    refs = pd.read_csv(tmp_path / "csv" / "references.csv")
    assert sorted(set(refs["n_f"])) == [2, 6] and len(refs) == 6
    plot.main(["--src", str(tmp_path / "csv"), "--dest", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "sample_efficiency_pair-0-1.png").exists()
    with pytest.raises(SystemExit):  # larger than the forget pool
        run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--forget_sizes", "9999"] + common)


def test_pretrain_sizes(tmp_path):
    """--pt_sizes: sizes from the untrained network up, with the step floor lengthening small runs."""
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out), "--pt_epochs", "1"]
    sizes = ["--pt_sizes", "0", "10", "100", "--pt_min_steps", "20"]
    for _ in range(2):  # the second call skips everything
        run.main(["pretrain_curve", "--seed", "0"] + sizes + common)
    assert sorted(p.name for p in out.glob("pretrain_curve/seed=0/*")) == ["n=0", "n=10", "n=100"]
    aggregate.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    curve = pd.read_csv(tmp_path / "csv" / "pretrain_curve.csv").sort_values("n_train")
    assert list(curve["n_train"]) == [0, 10, 100]
    assert list(curve["steps"]) == [0, 20, 20]  # n=10: 20 one-step epochs; n=100: 4 steps x 5 epochs
    assert curve["test_acc"].iloc[0] < 0.5  # the untrained network is near chance
    with pytest.raises(SystemExit):  # more images than the training set
        run.main(["pretrain_curve", "--seed", "0", "--pt_sizes", "99999"] + common)


def test_three_stage_figure(tmp_path):
    """Matched retain/forget sizes, a fixed-budget mu_f = 0 run, pretraining sizes, and the three-panel figure."""
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out)]
    run.main(["pretrain", "--seed", "0"] + common)
    run.main(["retrain", "--exp", "pair-0-1", "--seed", "0"] + common)
    run.main(["pretrain_curve", "--seed", "0", "--pt_sizes", "0", "20", "100", "--pt_min_steps", "10"] + common)
    sizes = ["--forget_sizes", "2", "6", "--match_retain"]
    run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", "0.5"] + sizes + common)
    run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", "0"] + sizes + common
             + ["--ul_stop_rule", "none", "--ul_max_steps", "6"])  # last flags win over SYN's defaults
    run.main(["references", "--exp", "pair-0-1", "--seed", "0", "--forget_sizes", "2", "6"] + common)
    assert sorted(p.name for p in out.glob("unlearn/exp=pair-0-1/seed=0/mu_f=0.5/rescale=none/*")) == ["m=2", "m=6"]
    aggregate.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    unl = pd.read_csv(tmp_path / "csv" / "unlearn.csv")
    assert unl["retain_matched"].all() and not unl["retain_full"].any()
    assert (unl["n_r"] == unl["n_f"]).all()  # n retain and n forget images
    budget = unl[unl["mu_f"] == 0]
    assert set(budget["ul_status"]) == {"budget"} and (budget["ul_steps"] == 6).all()
    plot.main(["--src", str(tmp_path / "csv"), "--dest", str(tmp_path / "fig")])
    for forget in ("2way", "10way"):
        assert (tmp_path / "fig" / f"pipeline_pair-0-1_{forget}.png").exists()


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


def test_every_model_is_saved_and_analysed(pipeline):
    out = pipeline["out"]
    unlearn_dirs = sorted(p.parent for p in out.glob("unlearn/exp=*/seed=*/mu_f=*/rescale=*/f=*/result.json"))
    ref_dirs = sorted(p.parent for p in out.glob("references/exp=*/seed=*/model=*/f=*/result.json"))
    assert len(unlearn_dirs) == 6 and len(ref_dirs) == 12
    for d in unlearn_dirs:
        assert (d / "model_fp16.pt").exists() and (d / "model_relearned_fp16.pt").exists()
        rep = json.loads((d / "representation.json").read_text())
        assert {"after_unlearn", "after_relearn"} <= set(rep)
    for d in ref_dirs:
        assert (d / "model_fp16.pt").exists() and (d / "representation.json").exists()
    assert len(list(out.glob("pretrain/seed=*/model.pt"))) == 2 and len(list(out.glob("retrain/*/*/model.pt"))) == 2
    before = json.loads((out / "references" / f"exp={pipeline['pairs'][0]}" / "seed=0"
                         / "representation_before.json").read_text())["metrics"]
    assert before["retrain"]["cka_retrain_r"] == pytest.approx(1.0)
    assert before["pretrained"]["cka_pretrained_f"] == pytest.approx(1.0)
    rep = pd.read_csv(pipeline["csv"] / "representation.csv")
    # per experiment: 3 before + 2 fractions x 2 stages + 2 fractions x 3 references; + rescaled run (2 x 2)
    assert len(rep) == 2 * (3 + 2 * 2 + 2 * 3) + 4
    assert set(rep["when"]) == {"before_relearn", "after_unlearn", "after_relearn"}
    assert rep["cka_retrain_r"].between(0, 1.0 + 1e-9).all()


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
    assert counts == {"pretrain": 5, "pretrain_curve": 5, "select_pairs": 1, "retrain": 15, "unlearn_relearn": 45, "relearn_sweep": 45, "references": 15, "reeval": 15,
                      "represent": 15}
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


def test_augment_batch():
    import torch
    from experiments.cifar_unlearning import train as T
    x = torch.randn(64, 3, 32, 32)
    T.set_augment(False)
    assert T.augment_batch(x) is x
    T.set_augment(True)
    try:
        y = T.augment_batch(x)
        assert y.shape == x.shape and not torch.equal(x, y)
        # every output is a shifted (and maybe flipped) copy: the pixel multiset away from the border is preserved
        assert torch.isclose(y.abs().sum(), x.abs().sum(), rtol=0.25)
        # shift 0 + no flip is possible: with many draws some image must equal its source
        ids = sum(int(torch.equal(T.augment_batch(x[:1])[0], x[0])) for _ in range(2000))
        assert ids > 0
    finally:
        T.set_augment(False)


def test_mse_loss_stop_rule(tmp_path):
    """MSE arm: stop rule 'loss' trains to convergence of the bounded effective-target objective, mu_f = 0 is
    allowed (forget target 0), no false 'collapsed' at large rho, and augmentation runs end to end."""
    import torch
    from experiments.cifar_unlearning import losses, train as T
    # optimum of the effective objective is exactly zero
    y = torch.tensor([1, 2])
    for mu_f in (0.0, 0.75):
        rho = mu_f / (1 - mu_f)
        target_r = (1 + rho) * torch.nn.functional.one_hot(y, 10).float()
        target_f = -rho * torch.nn.functional.one_hot(y, 10).float()
        assert losses.unlearning_retain_loss(target_r, y, 1.0, mu_f, "mse") == 0
        assert losses.unlearning_forget_loss(target_f, y, 1.0, mu_f, "mse") == 0
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out), "--loss", "mse", "--augment", "--pt_lr", "0.02"]
    run.main(["pretrain", "--seed", "0"] + common)
    run.main(["retrain", "--exp", "pair-0-1", "--seed", "0"] + common)
    ul = ["--forget_sizes", "4", "--match_retain", "--ul_stop_rule", "loss", "--ul_max_steps", "200",
          "--ul_eval_every", "10", "--ul_lr", "0.01", "--ul_retain_loss_threshold", "1.0"]
    for mu_f in ("0", "0.75"):
        run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", mu_f] + common + ul)
        r = run.read_json(out / f"unlearn/exp=pair-0-1/seed=0/mu_f={mu_f}/rescale=none/m=4/result.json")
        assert r["augment"] is True
        assert r["unlearn"]["status"] in ("success", "step_cap")  # never "collapsed" for MSE
        h = r["unlearn"]["history"]
        if r["unlearn"]["status"] == "success":
            assert h[-1]["retain_loss"] + h[-1]["forget_loss"] < 1.0
    with pytest.raises(ValueError):  # the loss rule needs a bounded objective
        T.unlearn(None, torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
                  __import__("experiments.cifar_unlearning.config", fromlist=["x"]).UnlearnConfig(stop_rule="loss"), seed=0, loss_kind="ce")
    with pytest.raises(SystemExit):  # an augmented root cannot be resumed without --augment
        run.main(["pretrain", "--seed", "0"] + [a for a in common if a != "--augment"])


def test_theory_mode(tmp_path):
    """--theory: stage 2 on n mixed rows weighted equally, stage 3 the forget-only attack (onehot on forget rows,
    zero on retain rows) on disjoint fresh rows, same rows for the references; folders t=<n>; table and figure."""
    import torch
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out), "--loss", "mse", "--pt_lr", "0.02"]
    th = ["--theory", "--forget_sizes", "2", "12", "--ul_stop_rule", "loss", "--ul_max_steps", "100",
          "--ul_lr", "0.01", "--ul_retain_loss_threshold", "1.0", "--rl_loss_threshold", "1.0"]
    run.main(["pretrain", "--seed", "0"] + common)
    run.main(["retrain", "--exp", "pair-0-1", "--seed", "0"] + common)
    for _ in range(2):  # the second call skips everything
        for mu_f in ("0", "0.5"):
            run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", mu_f] + common + th)
        run.main(["references", "--exp", "pair-0-1", "--seed", "0"] + common + th)
    r = run.read_json(out / "unlearn/exp=pair-0-1/seed=0/mu_f=0.5/rescale=none/t=12/result.json")
    assert r["theory"] and r["n_rows"] == 12 and r["n_r"] + r["n_f"] == 12 and r["n_f"] >= 1
    split = run.load_split(str(out / "splits/exp=pair-0-1/seed=0.npz"))
    a_r, a_f = run.theory_rows(split, 12)
    b_r, b_f = run.theory_rows(split, 12, fresh=True)
    assert not set(a_r) & set(b_r) and not set(a_f) & set(b_f)  # unlearning and relearning rows are disjoint
    assert len(a_f) == round(12 * len(split.forget_pool_order) / (len(split.forget_pool_order) + len(split.retain_order)))
    assert len(run.theory_rows(split, 2)[1]) == 1
    aggregate.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    unl = pd.read_csv(tmp_path / "csv" / "unlearn.csv")
    assert unl["theory"].all() and sorted(set(unl["n_rows"])) == [2, 12]
    refs = pd.read_csv(tmp_path / "csv" / "references.csv")
    assert refs["theory"].all() and len(refs) == 6
    pre = pd.DataFrame()
    assert plot.plot_pipeline("pair-0-1", pre, unl, refs, tmp_path / "fig")
    # the relearning targets: zero vector on retain rows, onehot on forget rows
    from experiments.cifar_unlearning import train as T
    model = torch.nn.Linear(3, 4)
    x = torch.randn(5, 3)
    tg = torch.zeros(5, 4)
    assert abs(T.full_target_loss(model, x, tg) - 0.5 * model(x).pow(2).sum(1).mean().item()) < 1e-6
    with pytest.raises(SystemExit):
        run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--theory"] + SYN
                 + ["--out_root", str(out), "--loss", "mse"])  # needs --forget_sizes


def test_theory_mode_ce(tmp_path):
    """--theory with --loss ce: row-share weights in the signed CE loss, stage 3 a KL loss towards onehot (forget
    rows) and the uniform distribution (retain rows); the whole pipeline runs and the table has the same columns."""
    import torch
    from experiments.cifar_unlearning import losses
    from experiments.cifar_unlearning import train as T
    # target_loss: ce = KL(target || softmax); zero at the target, uniform row = equal logits
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    onehot = torch.nn.functional.one_hot(torch.tensor([0, 0]), 4).float()
    uniform = torch.full((2, 4), 0.25)
    expect = (torch.log(torch.tensor(0.25)) - torch.log_softmax(logits[0], 0).mean()).item()  # row 0; row 1 is 0
    assert losses.target_loss(logits, uniform, "ce").item() == pytest.approx(0.5 * expect, abs=1e-6)
    assert losses.target_loss(logits[1:], uniform[1:], "ce").item() == pytest.approx(0.0, abs=1e-6)
    assert losses.target_loss(logits[1:], onehot[1:], "ce").item() == pytest.approx(
        torch.log(torch.tensor(4.0)).item(), abs=1e-6)
    # row weights scale the two CE terms
    y = torch.tensor([0, 1])
    base = losses.signed_unlearning_loss(logits, y, logits, y, 1.0, 0.5, "ce")
    half = losses.signed_unlearning_loss(logits, y, logits, y, 1.0, 0.5, "ce", w_r=0.5, w_f=0.5)
    assert half.item() == pytest.approx(0.5 * base.item(), abs=1e-6)

    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out), "--loss", "ce", "--pt_lr", "0.02"]
    th = ["--theory", "--forget_sizes", "2", "12", "--ul_stop_rule", "none", "--ul_max_steps", "30",
          "--ul_lr", "0.01", "--ul_forget_clip", "5", "--rl_loss_threshold", "1.0"]
    run.main(["pretrain", "--seed", "0"] + common)
    run.main(["retrain", "--exp", "pair-0-1", "--seed", "0"] + common)
    for mu_f in ("0", "0.5"):
        run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", mu_f] + common + th)
    run.main(["references", "--exp", "pair-0-1", "--seed", "0"] + common + th)
    r = run.read_json(out / "unlearn/exp=pair-0-1/seed=0/mu_f=0.5/rescale=none/t=12/result.json")
    assert r["theory"] and r["n_rows"] == 12 and r["n_r"] + r["n_f"] == 12
    split = run.load_split(str(out / "splits/exp=pair-0-1/seed=0.npz"))
    ctx = run.Context(run.build_parser().parse_args(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0"] + common + th))
    _, _, targets, n_r, n_f = run.theory_relearn_set(ctx, split, 12)
    assert torch.allclose(targets[:n_r], torch.full((n_r, 10), 0.1).to(targets.device))
    assert torch.allclose(targets[n_r:].sum(1), torch.ones(n_f).to(targets.device)) and targets[n_r:].max() == 1.0
    aggregate.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    unl = pd.read_csv(tmp_path / "csv" / "unlearn.csv")
    assert unl["theory"].all() and sorted(set(unl["n_rows"])) == [2, 12] and "rl_A_r8" in unl.columns



def test_relearn_sweep(tmp_path):
    """relearn_sweep: the model unlearned with --ul_from rows, relearned at every size, on the references' rows."""
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out), "--loss", "mse", "--pt_lr", "0.02"]
    th = ["--theory", "--ul_stop_rule", "loss", "--ul_max_steps", "60", "--ul_lr", "0.01",
          "--ul_retain_loss_threshold", "1.0", "--rl_loss_threshold", "1.0"]
    run.main(["pretrain", "--seed", "0"] + common)
    run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", "0.5", "--forget_sizes", "6"]
             + common + th)
    sweep = ["relearn_sweep", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", "0.5", "--ul_from", "6",
             "--forget_sizes", "2", "6", "12"]
    for _ in range(2):  # the second call skips everything
        run.main(sweep + common + th)
    base = out / "unlearn/exp=pair-0-1/seed=0/mu_f=0.5/rescale=none/t=6/relearn_sweep"
    assert sorted(p.name for p in base.iterdir()) == ["t=12", "t=2", "t=6"]
    # the relearned model at n_RL = n_UL matches the one unlearn_relearn made (same rows, fp16-rounded start)
    aggregate.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    sw = pd.read_csv(tmp_path / "csv" / "relearn_sweep.csv")
    assert sorted(sw["n_rows"]) == [2, 6, 12] and (sw["n_ul"] == 6).all()
    with pytest.raises(SystemExit):  # no unlearned model with 7 rows
        run.main(["relearn_sweep", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", "0.5", "--ul_from", "7",
                  "--forget_sizes", "2"] + common + th)
    with pytest.raises(SystemExit):  # different unlearning settings than the saved model
        run.main(sweep + common + th + ["--ul_lr", "0.02"])


def test_reeval_adds_r8(tmp_path):
    """A_r8 (argmax over the kept classes only) is recorded by new runs and added to old records by reeval,
    without changing stored values; the pretraining curve stores it per pair; the figure can use it."""
    import torch
    from experiments.cifar_unlearning import metrics
    logits = torch.tensor([[5.0, 4.0, 0.0], [0.0, 9.0, 1.0]])
    assert metrics.kept_class_accuracy(logits, torch.tensor([1, 1]), (0,)) == 1.0
    assert metrics.kept_class_accuracy(logits, torch.tensor([1, 1]), (2,)) == 0.5
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out), "--loss", "mse", "--pt_lr", "0.02"]
    th = ["--theory", "--forget_sizes", "4", "--ul_stop_rule", "loss", "--ul_max_steps", "40", "--ul_lr", "0.01",
          "--ul_retain_loss_threshold", "1.0", "--rl_loss_threshold", "1.0"]
    run.main(["pretrain", "--seed", "0"] + common)
    run.main(["retrain", "--exp", "pair-0-1", "--seed", "0"] + common)
    run.main(["pretrain_curve", "--seed", "0", "--pt_sizes", "0", "30"] + common)
    run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", "0.5"] + common + th)
    run.main(["references", "--exp", "pair-0-1", "--seed", "0"] + common + th)
    run.main(["relearn_sweep", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", "0.5", "--ul_from", "4"] + common + th)
    res = out / "unlearn/exp=pair-0-1/seed=0/mu_f=0.5/rescale=none/t=4/result.json"
    assert (res.parent / "relearn_sweep/t=4/model_relearned_fp16.pt").exists()
    r = run.read_json(res)
    fresh_r8 = r["after_unlearn"]["A_r8"]
    # simulate a record from before A_r8 existed
    for key in ("after_unlearn", "after_relearn"):
        del r[key]["A_r8"]
    r["after_unlearn"]["A_r"] = 0.123456  # a stored value must survive
    run.write_json(res, r)
    before = out / "references/exp=pair-0-1/seed=0/before.json"
    b = run.read_json(before)
    del b["metrics"]["pretrained"]["A_r8"]
    run.write_json(before, b)
    run.main(["reeval", "--exp", "pair-0-1", "--seed", "0"] + common)
    r = run.read_json(res)
    assert abs(r["after_unlearn"]["A_r8"] - fresh_r8) < 0.05  # fp16 model, same value up to rounding
    assert "A_r8" in r["after_relearn"] and r["after_unlearn"]["A_r"] == 0.123456
    assert "A_r8" in run.read_json(before)["metrics"]["pretrained"]
    aggregate.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    pc = pd.read_csv(tmp_path / "csv" / "pretrain_curve.csv")
    assert "r8_0_1" in pc.columns and pc["r8_0_1"].between(0, 1).all()
    unl = pd.read_csv(tmp_path / "csv" / "unlearn.csv")
    refs = pd.read_csv(tmp_path / "csv" / "references.csv")
    sw = pd.read_csv(tmp_path / "csv" / "relearn_sweep.csv")
    assert {"ul_A_r8", "rl_A_r8"} <= set(unl.columns) and "before_A_r8" in refs.columns and "rl_A_r8" in sw.columns
    paths = plot.plot_pipeline("pair-0-1", pc, unl, refs, tmp_path / "fig", sweep=sw, retain="8way")
    assert paths and paths[0].name == "pipeline_pair-0-1_2way_r8_sweep.png"
    # a table with two unlearning sizes: panel (d) takes the requested one (default the largest), never a mix
    n0 = int(sw["n_ul"].iloc[0])
    two = pd.concat([sw, sw.assign(n_ul=n0 * 10)])
    assert set(plot.select_unlearning_size(two)["n_ul"]) == {n0 * 10}
    assert set(plot.select_unlearning_size(two, n0)["n_ul"]) == {n0}
    assert len(plot.select_unlearning_size(two, 7)) == 0
    assert plot.plot_pipeline("pair-0-1", pc, unl, refs, tmp_path / "fig", sweep=two, retain="8way", n_ul=n0)


def test_figure_title_names_the_loss():
    assert plot._title("pair-1-7", "mse") == "pair-1-7, MSE loss"
    assert plot._title("pair-1-7", "ce") == "pair-1-7, CE loss"
    assert plot._title("pair-1-7") == "pair-1-7"


def test_title_label_and_unfrozen_bn(tmp_path):
    assert plot._title_label("mse", "BN unfrozen") == "MSE loss, BN unfrozen"
    assert plot._title_label("ce") == "CE loss"
    # stages 2 and 3 run with BatchNorm in ordinary train mode and record the setting
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out), "--loss", "mse", "--pt_lr", "0.02"]
    th = ["--theory", "--forget_sizes", "2", "--ul_stop_rule", "loss", "--ul_max_steps", "20", "--ul_lr", "0.01",
          "--ul_retain_loss_threshold", "1.0", "--rl_loss_threshold", "1.0", "--ul_freeze_bn", "false",
          "--rl_freeze_bn", "false"]
    run.main(["pretrain", "--seed", "0"] + common)
    run.main(["unlearn_relearn", "--exp", "pair-0-1", "--seed", "0", "--ul_mu_f", "0.5"] + common + th)
    r = run.read_json(out / "unlearn/exp=pair-0-1/seed=0/mu_f=0.5/rescale=none/t=2/result.json")
    assert r["configs"]["ul"]["freeze_bn"] is False and r["configs"]["rl"]["freeze_bn"] is False


def test_unfrozen_bn_unlearning_uses_one_pass_for_both_batches():
    """With BatchNorm in train mode, retain and forget images share one forward pass (shared batch statistics);
    with frozen BatchNorm each batch has its own pass (the layers act on each image alone, so the result is the same)."""
    import torch
    from experiments.cifar_unlearning import train as T
    from experiments.cifar_unlearning.config import UnlearnConfig
    from experiments.cifar_unlearning.models import build_model

    def passes(freeze_bn: bool):
        model = build_model("tiny", 4)
        sizes = []
        model.register_forward_pre_hook(lambda m, inp: sizes.append(len(inp[0])) if m.training else None)
        torch.manual_seed(0)
        x_r, y_r = torch.randn(6, 3, 8, 8), torch.randint(0, 4, (6,))
        x_f, y_f = torch.randn(2, 3, 8, 8), torch.randint(0, 4, (2,))
        cfg = UnlearnConfig(mu_f=0.5, max_steps=3, eval_every=3, stop_rule="none", freeze_bn=freeze_bn, batch_size=6)
        T.unlearn(model, x_r, y_r, x_f, y_f, cfg, seed=0, loss_kind="mse", row_weights=(0.75, 0.25))
        return sizes
    assert passes(False) == [8, 8, 8]  # 6 retain + 2 forget images in one pass per step
    assert passes(True) == [6, 2, 6, 2, 6, 2]
