"""
Overlay replica theory curves with empirical imperfect-PT results.
Saves replica_vs_emp_quick.png (2×1 grid — right column only).
  Row 0: omega=1, Row 1: omega=0
  sigma0_PT sweep using alpha_pt=0.95 FP theory curves
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REP_CSV = "replica_quick.csv"
EMP_CSV = "emp_imperfect_pt_quick.csv"
OUT_PNG = "replica_vs_emp_quick.png"

SIGMA0_PT_LIST = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
OMEGAS = [1.0, 0.0]

rep = pd.read_csv(REP_CSV)
emp = pd.read_csv(EMP_CSV)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharey=False)
fig.suptitle("Replica theory vs empirical  —  imperfect PT", fontsize=13)

for row, omega in enumerate(OMEGAS):
    sub_rep = rep[rep["omega"] == omega]
    sub_emp = emp[emp["omega"] == omega]

    ax = axes[row]
    ax.set_title(f"ω={omega}  |  σ₀_PT sweep (α_PT=0.95)", fontsize=10)
    ax.set_xlabel("α_FT")
    ax.set_ylabel("FT MSE")

    # Theory curves: alpha_pt=0.95, sigma0_pt sweep
    for i, s0 in enumerate(SIGMA0_PT_LIST):
        lbl = f"w{omega}_apt0.95_s0{s0}"
        sub = sub_rep[sub_rep["label"] == lbl]
        if len(sub):
            ax.plot(sub["alpha"], sub["mse_best"], color=colors[i],
                    label=f"σ₀={s0}", lw=1.5)

    # Oracle
    oracle = sub_rep[sub_rep["label"] == f"w{omega}_oracle"]
    if len(oracle):
        ax.plot(oracle["alpha"], oracle["mse_best"], "k--", lw=1.2, label="oracle")

    # Empirical: sigma0_pt=0.01 (any alpha_pt, small noise so diff is negligible)
    emp_n01 = sub_emp[(sub_emp["pt_mode"] == "noisy") &
                      (sub_emp["sigma0_pt"].round(4) == 0.01)]
    if len(emp_n01):
        grp = emp_n01.groupby("alpha_ft_req")["ft_param_mse"]
        mu  = grp.mean(); se = grp.sem()
        ax.errorbar(mu.index, mu.values, yerr=2*se.values,
                    fmt="s", color="red", capsize=3, markersize=5,
                    label="σ₀=0.01 (emp)")

    # Empirical: sigma0_pt=0.5, alpha_pt=0.95 (new runs)
    emp_n5 = sub_emp[(sub_emp["pt_mode"] == "noisy") &
                     (sub_emp["sigma0_pt"].round(2) == 0.5) &
                     (sub_emp["alpha_pt"].round(2) == 0.95)]
    if len(emp_n5):
        grp = emp_n5.groupby("alpha_ft_req")["ft_param_mse"]
        mu  = grp.mean(); se = grp.sem()
        ax.errorbar(mu.index, mu.values, yerr=2*se.values,
                    fmt="^", color="purple", capsize=3, markersize=5,
                    label="σ₀=0.5, α_PT=0.95 (emp)")

    ax.set_xlim(0, 0.5)
    ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
