# --- Empirical PT+FT (standalone cell) ---
from pathlib import Path
import sys

# make repo imports work from anywhere
p = Path.cwd().resolve()
for _ in range(12):
    if (p / ".git").exists():
        sys.path.insert(0, str(p))
        break
    p = p.parent

import numpy as np
import pandas as pd
import ptft_empirical_finetune_df as emp

# alpha grid + seeds
alphas = np.linspace(0.01, 0.2, 5)
seeds = [0]
lmdas = [1e-3 * -0.99, 0, 1e-3*00.99]
cs = [1e-3]
omegas = [0.0, 1.0]
gamma_reinits = [0.0, 1.0]


df_empirical_ptft_quickcheck = emp.build_ptft_finetune_curves_dataframe(
    rho_pt=[0.10],
    rho_ft=[0.10],
    omega=omegas,
    a_pt=1.0,
    c_pt=cs,
    lambda_pt=lmdas,
    gamma_reinit=gamma_reinits,
    alphas=alphas.tolist(),
    inp_dim=5000,
    n_test=10_000,
    seeds=seeds,
    lr=0.5,
    epochs=5_000_000,
    test_every_n_epochs=5000,   # eval/stopping cadence
    log_every_n_epochs=50000,   # log cadence
    no_tuning=True,
    threshold=1e-5,
)

df_empirical_ptft_quickcheck 