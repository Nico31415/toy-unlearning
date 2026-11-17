"""The modular-arithmetic pipeline on a tiny setting (CPU, about a minute): data, Fourier tools, every stage, launch
arrays, aggregation and figures."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.modular_unlearning import aggregate, launch, plot, run
from experiments.modular_unlearning.core import (
    Transformer, ablate_frequency, amplitude_and_phase, fourier_coeffs, linear_function, make_setup, model_forward)

TINY = ["--retain_extra", "11", "13", "--d", "32", "--heads", "2", "--d_mlp", "64", "--pt_max_steps", "200",
        "--pt_eval_every", "100", "--ul_max_steps", "40", "--ul_eval_every", "20", "--rl_max_steps", "40",
        "--rl_eval_every", "20", "--n_key", "3", "--device", "cpu"]


def test_setup_splits():
    s = make_setup("p12-q6", [11, 13], 0)
    assert s.moduli == [12, 6, 11, 13] and s.max_p == 13 and s.vocab == 13 + 4 + 1
    f = s.forget
    all_pairs = np.concatenate([f.train, f.attack, f.test_fit, f.test_eval])
    assert len(all_pairs) == 144 and len({tuple(p) for p in all_pairs}) == 144
    assert len(f.train) == 72 and len(f.attack) == 36


def test_fourier_tools():
    p, d = 12, 5
    x = torch.arange(p, dtype=torch.float64)
    emb = torch.zeros(p + 3, d)
    emb[:p, 0] = torch.cos(2 * math.pi * 3 * x / p).float()
    c = fourier_coeffs(emb, p)
    assert c.abs()[2, 0] > 5 and c.abs()[[0, 1, 3, 4, 5], 0].max() < 1e-5  # only k = 3
    assert ablate_frequency(emb, p, 3)[:p].abs().max() < 1e-5
    assert amplitude_and_phase(0.5 * emb, emb, p, [3]) == pytest.approx({"amp": 0.25, "phase": 1.0})


def test_linearisation():
    torch.manual_seed(0)
    m = Transformer(18, 13, d=16, heads=2, d_mlp=32).eval()
    x = torch.randint(0, 13, (5, 4))
    th = {n: p.detach().clone() for n, p in m.named_parameters()}
    de = {n: 1e-3 * torch.randn_like(p) for n, p in th.items()}
    lin, taps = linear_function(m, th)(de, x)
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.add_(de[n])
    full, ftaps = model_forward(m, x)
    assert torch.allclose(lin, full, atol=1e-3) and set(taps) == {"embed", "attn0", "mlp0"}


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    root = tmp_path_factory.mktemp("mod")
    out = root / "out"
    c = ["--exp", "p12-q6", "--out_root", str(out)] + TINY
    for seed in ("0", "1"):
        run.main(["family", "--seed", seed, "--fam_L", "0", "36", "72"] + c)
    run.main(["family", "--fam_L", "72"] + c)  # skipped
    run.main(["phase", "--Lfull", "72", "--nS", "36", "72", "--methods", "F", "FR", "R", "--targets", "ybar", "oracle",
              "--arms", "full", "readout", "linear", "--S", "1", "0.5"] + c)
    run.main(["phase_relearn", "--Lfull", "72", "--nRL", "10", "30"] + c)
    aggregate.main(["--out_root", str(out), "--dest", str(root / "csv")])
    plot.main(["--src", str(root / "csv"), "--dest", str(root / "fig")])
    return root


def test_tables(pipe):
    fam = pd.read_csv(pipe / "csv" / "family.csv")
    assert len(fam) == 6 and fam.overlap.between(0, 1).all()
    ph = pd.read_csv(pipe / "csv" / "phase.csv")
    assert len(ph) == 2 * 5 * 3 * 2  # n_S x (F, FR x 2 targets + R) x arms x S
    for col in ("d_ut2", "d_ul2", "amp", "phase", "a0_amp", "acc_forget", "probe_mlp0", "fourier_r2_mlp0",
                "frozen_r2_mlp0"):
        assert ph[col].notna().all(), col
    full = ph[ph.n_S == 72]
    assert (full.d_ref2.abs() < 1e-6).all()
    rl = pd.read_csv(pipe / "csv" / "phase_relearn.csv")
    assert len(rl) == 2 * 40 + 2 * 3  # 40 non-linear models, plus A(36), A(0) and scratch
    assert (pipe / "fig" / "family.png").exists() and (pipe / "fig" / "hidden.png").exists()
    th = json.loads((pipe / "fig" / "thresholds.json").read_text())
    assert th["p12-q6"]["eta2"] >= 0.15


def test_launch_arrays():
    a = launch.build_array("family", ["p60-q30", "p60-q49"], [0, 1], fam_Ls=[0, 1800])
    assert launch.task_count(a) == 8
    argv = launch.to_argv(a.get_args(7))
    assert argv[0] == "family" and argv[argv.index("--fam_L") + 1] == "1800"
    run.build_parser().parse_args(argv)
    ph = launch.build_array("phase", ["p60-q30"], [0], Lfulls=[1800], methods=["F", "R"])
    assert launch.task_count(ph) == 2
    run.build_parser().parse_args(launch.to_argv(ph.get_args(1)))
