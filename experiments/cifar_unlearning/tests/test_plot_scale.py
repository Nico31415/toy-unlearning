import pandas as pd

from experiments.cifar_unlearning import plot_scale


def test_parse_tag():
    assert plot_scale.parse_tag("none") == (1.0, "stale")
    assert plot_scale.parse_tag("all-S0.125") == (0.125, "stale")
    assert plot_scale.parse_tag("all-S2-bnstat") == (2.0, "scaled")
    assert plot_scale.parse_tag("body_head-b1-h0.5") is None


def test_parse_mode():
    assert plot_scale.parse_mode("none") == ("none", 1.0, "stale")
    assert plot_scale.parse_mode("all-S0.5-bnstat") == ("all", 0.5, "scaled")
    assert plot_scale.parse_mode("body_head-b1-h0.125") == ("head", 0.125, "stale")
    assert plot_scale.parse_mode("body_head-b4-h1-bnstat") == ("conv", 4.0, "scaled")
    assert plot_scale.parse_mode("body_head-b2-h2") is None


def test_scale_figure(tmp_path):
    rows = []
    for tag in ("none", "all-S0.5", "all-S0.5-bnstat", "all-S2", "all-S2-bnstat"):
        for n in (10, 100):
            rows.append(dict(rescale=tag, theory=True, mu_f=0.75, n_rows=n, seed=0, ul_A_r8=0.9, ul_A_f2=0.4,
                             rl_A_r8=0.5, rl_A_f2=0.9, ul_steps=10, ul_status="converged"))
    pd.DataFrame(rows).to_csv(tmp_path / "unlearn.csv", index=False)
    plot_scale.main(["--src", str(tmp_path), "--dest", str(tmp_path / "fig"), "--loss", "MSE"])
    assert (tmp_path / "fig" / "scale_sweep.png").exists() and (tmp_path / "fig" / "scale_table.csv").exists()
