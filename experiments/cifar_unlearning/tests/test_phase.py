"""Phase experiments on synthetic data: family, every stage-2 method, target and arm, read-outs, relearning,
aggregation and figures; plus the pieces they rely on (single-class split, homogeneous rescaling, linearisation)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.cifar_unlearning import aggregate_phase, plot_phase, run
from experiments.cifar_unlearning.data import make_split, make_synthetic, parse_experiment
from experiments.cifar_unlearning.models import (build_model, feature_taps, homogeneity_degree, rescale_)
from experiments.cifar_unlearning.probes import (LinearForward, ModelForward, ensd, fit_probe, linear_function,
                                                 participation_ratio, balanced_accuracy)

SYN = ["--synthetic", "true", "--arch", "tiny", "--synthetic_train_per_class", "60",
       "--synthetic_test_per_class", "40", "--n_probe", "40", "--n_forget", "100",
       "--pt_epochs", "4", "--pt_lr", "0.05", "--pt_batch_size", "64", "--pt_max_final_train_loss", "10",
       "--ul_max_steps", "30", "--ul_eval_every", "10", "--ul_batch_size", "16", "--ul_stop_rule", "none",
       "--rl_max_steps", "20", "--rl_eval_every", "10", "--rl_batch_size", "16",
       "--ph_n_attack", "20", "--ph_probe_train_forget", "30", "--ph_probe_train_retain", "60",
       "--ph_probe_eval_retain", "100", "--ph_loss_eval_max", "200", "--device", "cpu", "--loss", "mse"]


def test_single_class_split():
    assert parse_experiment("class-3") == ("class", (3,))
    ds = make_synthetic(20, 10)
    split = make_split(ds, "class-3", 0, n_probe=5)
    assert (ds.y_train.numpy()[split.forget_all] == 3).all() and len(split.forget_all) == 20
    assert len(split.forget_pool_order) == 15 and (ds.y_test.numpy()[split.test_forget] == 3).all()


def test_homogeneous_rescaling():
    torch.manual_seed(0)
    net = build_model("tiny_plain", 10)
    x = torch.randn(4, 3, 32, 32)
    before = net(x)
    rescale_(net, "function", S=0.3)
    assert homogeneity_degree(net) == 3
    assert torch.allclose(net(x), 0.3 * before, atol=1e-5)
    res = build_model("tiny", 10)  # BatchNorm net: "function" scales only the head, which scales the logits too
    res.eval()
    b2 = res(x)
    rescale_(res, "function", S=2.0)
    assert torch.allclose(res(x), 2.0 * b2, atol=1e-5)


def test_linearisation_matches_first_order():
    torch.manual_seed(0)
    net = build_model("tiny", 10).eval()
    x = torch.randn(6, 3, 32, 32)
    theta0 = {n: p.detach().clone() for n, p in net.named_parameters()}
    delta = {n: 1e-3 * torch.randn_like(p) for n, p in theta0.items()}
    lin_logits, lin_feats = LinearForward(net, theta0, delta)(x)
    with torch.no_grad():
        for n, p in net.named_parameters():
            p.add_(delta[n])
    full_logits, full_feats = ModelForward(net)(x)
    assert torch.allclose(lin_logits, full_logits, atol=1e-4)
    for k in lin_feats:
        assert torch.allclose(lin_feats[k], full_feats[k], atol=1e-3)  # second order: ReLU and max-pool kinks
    # differentiable in delta (reverse over forward)
    d = {n: torch.zeros_like(p, requires_grad=True) for n, p in theta0.items()}
    out, _ = linear_function(net, theta0)(d, x)
    out.pow(2).sum().backward()
    assert all(v.grad is not None for v in d.values())


def test_probe_geometry():
    torch.manual_seed(0)
    x = torch.randn(200, 5)
    y = (x[:, 0] > 0).float()
    probe = fit_probe(x, y)
    assert balanced_accuracy(probe.score(x), y) > 0.95
    assert balanced_accuracy(fit_probe(10 * x, y).score(10 * x), y) > 0.95  # scale-blind
    z = torch.randn(300, 8)
    assert abs(ensd(z, z) - participation_ratio(z)) < 1e-8
    assert abs(participation_ratio(torch.randn(500, 1) @ torch.randn(1, 6)) - 1) < 1e-6
    assert list(feature_taps(build_model("tiny", 10))) == ["block1", "block2"]


@pytest.fixture(scope="module")
def phase_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase")
    out = root / "out"
    common = SYN + ["--out_root", str(out)]
    for seed in ("0", "1"):
        run.main(["family", "--exp", "pair-0-1", "--seed", seed, "--fam_L", "0", "6", "10"] + common)
    run.main(["family", "--exp", "pair-0-1", "--seed", "0", "--fam_L", "10"] + common)  # skipped
    run.main(["phase", "--exp", "pair-0-1", "--seed", "0", "--ph_Lfull", "10", "--ph_nS", "4", "10",
              "--ph_methods", "F", "FR", "R", "--ph_targets", "ybar", "neg", "oracle", "approx",
              "--ph_arms", "full", "readout", "linear", "--ph_S", "1", "0.5"] + common)
    run.main(["phase", "--exp", "pair-0-1", "--seed", "1", "--ph_Lfull", "10", "--ph_nS", "4",
              "--ph_methods", "F", "--ph_targets", "ybar"] + common)
    run.main(["phase", "--exp", "pair-0-1", "--seed", "0", "--ph_Lfull", "10", "--ph_nS", "4",
              "--ph_methods", "F", "--ph_targets", "ybar", "--ph_track_every", "10",
              "--out_root", str(root / "track"), "--family_root", str(out)] + SYN)
    run.main(["phase_relearn", "--exp", "pair-0-1", "--seed", "0", "--ph_Lfull", "10", "--ph_nRL", "5", "20"] + common)
    csv = root / "csv"
    aggregate_phase.main(["--out_root", str(out), "--dest", str(csv)])
    aggregate_phase.main(["--out_root", str(root / "track"), "--dest", str(root / "csv_track")])
    plot_phase.main(["--src", str(csv), "--dest", str(root / "fig")])
    return {"out": out, "csv": csv, "fig": root / "fig", "root": root}


def test_phase_tables(phase_run):
    fam = pd.read_csv(phase_run["csv"] / "family.csv")
    assert len(fam) == 6 and {"probe_block1", "probe_block2", "pr_block2", "A_f10"} <= set(fam.columns)
    ph = pd.read_csv(phase_run["csv"] / "phase.csv")
    # seed 0: 2 n_S x (F, FR: 4 targets; R: 1) x 3 arms x 2 S = 2 * 9 * 6 = 108; seed 1: 1
    assert len(ph) == 109
    assert set(ph.method) == {"F", "FR", "R"} and set(ph.arm) == {"full", "readout", "linear"}
    assert set(ph[ph.method == "R"].target) == {"-"}
    for col in ("d_ut2", "d_ul2", "d_ref2", "d_noop_ut2", "A_f10", "A_f10_refit", "probe_block2", "frozen_block2", "cka_ad_block2",
                "ensd_ad_block2", "amp_block2", "rho_amp_block2", "ref_amp_block2_ad"):
        assert ph[col].notna().all(), col
    # n_S = L_full: S = S_full, so A(D minus S) is A(0) and the two references coincide
    full = ph[ph.n_S == 10]
    assert (full.d_ref2.abs() < 1e-6).all() and np.allclose(full.d_ut2, full.d_ul2)
    assert np.allclose(full.d_noop_ut2, 1.0)  # A(D minus S) = A(0): the do-nothing distance is the normaliser
    noise = pd.read_csv(phase_run["csv"] / "noise.csv")
    assert set(noise.L) == {0, 6, 10}
    assert noise[noise.L == 0].d2.isna().all() and noise[noise.L > 0].d2.notna().all()  # A(0) - A(0) normalises to 0/0
    rel = pd.read_csv(phase_run["csv"] / "phase_relearn.csv")
    # every full/readout model (72 of seed 0's 108) x 2 attack sizes, plus A(6), A(0) and scratch x 2
    assert len(rel) == 72 * 2 + 3 * 2
    track = pd.read_csv(phase_run["root"] / "csv_track" / "phase_track.csv")
    assert list(track.step) == [0, 10, 20, 30]


def test_phase_figures(phase_run):
    th = json.loads((phase_run["fig"] / "thresholds.json").read_text())
    assert set(th) == {"pair-0-1"} and th["pair-0-1"]["eta2"] >= 0.15
    assert (phase_run["fig"] / "family.png").exists() and (phase_run["fig"] / "rho_layers.png").exists()
    assert any(p.name.startswith("phase_pair-0-1_F_ybar_full") for p in phase_run["fig"].iterdir())


def test_phase_rejects_bad_settings(phase_run):
    common = SYN + ["--out_root", str(phase_run["out"])]
    with pytest.raises(SystemExit):  # n_S larger than L_full
        run.main(["phase", "--exp", "pair-0-1", "--seed", "0", "--ph_Lfull", "6", "--ph_nS", "7"] + common)
    with pytest.raises(SystemExit):  # family member missing
        run.main(["phase", "--exp", "pair-0-1", "--seed", "0", "--ph_Lfull", "10", "--ph_nS", "3"] + common)
    with pytest.raises(SystemExit):  # unknown method
        run.main(["phase", "--exp", "pair-0-1", "--seed", "0", "--ph_Lfull", "10", "--ph_methods", "X"] + common)


def test_mix_split():
    assert parse_experiment("mix-3-0.25") == ("mix", (3,))
    ds = make_synthetic(60, 10)
    y = ds.y_train.numpy()
    for alpha in (0.0, 0.25, 1.0):
        split = make_split(ds, f"mix-3-{alpha:g}", 0, n_forget=100, n_probe=10, n_attack=8)
        pool = split.forget_pool_order
        assert (y[pool[-8:]] == 3).all() and (y[split.probe] == 3).all()  # attack pool and P_f: class 3 only
        assert not np.isin(y[split.retain_order], 3).any()  # class 3 appears only in the pool
        assert not set(pool) & set(split.retain_order) and not set(pool) & set(split.probe)
        head = pool[:40]
        assert (y[head] == 3).sum() == round(alpha * 40)  # any prefix is an alpha fraction class 3
        assert len(pool) == (60 - 10 if alpha > 0 else 8) + (100 if alpha < 1 else 0)


def test_mix_phase(tmp_path):
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out), "--n_forget", "60", "--n_probe", "20", "--ph_n_attack", "10"]
    run.main(["family", "--exp", "mix-1-0.5", "--seed", "0", "--fam_L", "0", "6", "12"] + common)
    run.main(["phase", "--exp", "mix-1-0.5", "--seed", "0", "--ph_Lfull", "12", "--ph_nS", "6",
              "--ph_methods", "F", "R", "--ph_targets", "ybar", "--ph_arms", "full"] + common)
    run.main(["family", "--exp", "mix-1-0", "--seed", "0", "--fam_L", "0", "6", "12"] + common)
    run.main(["phase", "--exp", "mix-1-0", "--seed", "0", "--ph_Lfull", "12", "--ph_nS", "6",
              "--ph_methods", "R", "--ph_arms", "full"] + common)
    aggregate_phase.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    ph = pd.read_csv(tmp_path / "csv" / "phase.csv")
    half = ph[ph.exp == "mix-1-0.5"]
    assert (half.n_S_cls == 3).all() and (half.n_S_iid == 3).all()
    for col in ("mia_cls", "mia_iid", "ref_ut_mia_iid", "ref_ad_mia_iid", "r_S_ut_iid", "r_S_ut_cls", "gap_S_iid",
                "acc_S_iid"):
        assert half[col].notna().all(), col
    iid = ph[ph.exp == "mix-1-0"]
    assert (iid.n_S_iid == 6).all() and "n_S_cls" not in iid.columns or iid.n_S_cls.isna().all()
    # F's clean retain set keeps the pool's iid images left in D: 6 pool images left, 3 of them iid
    res = json.loads(next((out / "phase").glob("exp=mix-1-0.5/**/m=F/**/result.json")).read_text())
    assert res["n_retain_rows"] == res_retain(out) + 3


def res_retain(out):
    split = np.load(next((out / "splits" / "exp=mix-1-0.5").glob("*.npz")))
    return len(split["retain_order"])


def test_mix_plot(tmp_path):
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out), "--n_forget", "60", "--n_probe", "20", "--ph_n_attack", "10"]
    for exp in ("mix-1-0", "mix-1-0.5"):
        run.main(["family", "--exp", exp, "--seed", "0", "--fam_L", "0", "6", "12"] + common)
        run.main(["phase", "--exp", exp, "--seed", "0", "--ph_Lfull", "12", "--ph_nS", "6",
                  "--ph_methods", "R", "--ph_arms", "full"] + common)
    aggregate_phase.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    plot_phase.main(["--src", str(tmp_path / "csv"), "--dest", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "mix_untraining.png").exists()


def test_relearn_budget(tmp_path):
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out)]
    run.main(["family", "--exp", "pair-0-1", "--seed", "0", "--fam_L", "0", "10"] + common)
    run.main(["phase", "--exp", "pair-0-1", "--seed", "0", "--ph_Lfull", "10", "--ph_nS", "10", "--ph_methods", "R",
              "--ph_arms", "full"] + common)
    run.main(["phase_relearn", "--exp", "pair-0-1", "--seed", "0", "--ph_Lfull", "10", "--ph_nRL", "5",
              "--rl_stop_rule", "budget", "--rl_classes", "all", "--rl_max_steps", "20", "--rl_loss_threshold", "100"]
             + common)
    aggregate_phase.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    rl = pd.read_csv(tmp_path / "csv" / "phase_relearn.csv")
    assert (rl.steps == 20).all() and (rl.status == "budget").all()  # a huge threshold does not stop a budget run
    assert rl["A_f10_at10"].notna().all()


def test_keep_forget_omega(tmp_path):
    out = tmp_path / "out"
    common = SYN + ["--out_root", str(out)]
    run.main(["family", "--exp", "pair-0-1", "--seed", "0", "--fam_L", "0", "4", "10"] + common)
    run.main(["phase", "--exp", "pair-0-1", "--seed", "0", "--ph_Lfull", "10", "--ph_nS", "6", "--ph_methods", "F", "R",
              "--ph_targets", "ybar", "--ph_omegas", "0.5", "1", "--ph_arms", "full"] + common)
    aggregate_phase.main(["--out_root", str(out), "--dest", str(tmp_path / "csv")])
    ph = pd.read_csv(tmp_path / "csv" / "phase.csv")
    f = ph[ph.method == "F"].set_index("target")
    assert set(f.index) == {"ybar", "ybar@w0.5"} and f.loc["ybar@w0.5", "omega"] == 0.5
    assert (ph[ph.method == "R"].omega == 1.0).all() and len(ph[ph.method == "R"]) == 1  # R has no forget term
    assert len(list((out / "phase").glob("**/t=ybar@w0.5/**/result.json"))) == 1
