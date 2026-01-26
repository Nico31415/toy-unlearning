# --- Replica PT+FT (standalone cell) ---
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import ptft_replica_qk as rq

# alpha grid + seeds
alphas = np.linspace(0.01, 0.5, 40)
seeds = [0]
lmdas = [1e-3 * -0.99, 0, 1e-3*00.99]
cs = [1e-3]
omegas = [0.0, 1.0]
gamma_reinits = [0.0, 1.0]

# PT+FT setups (Cartesian product across list-valued args)
df_replica_ptft = rq.build_ptft_curves_dataframe(
    rho_pt=[0.10],
    rho_ft=[0.10],   # change to your FT sparsities
    omega=omegas,   # change overlap list as desired
    c_pt=cs,  # change PT c list as desired
    lambda_pt=lmdas,  # change lambda list as desired
    gamma_reinit=gamma_reinits,      # change if you want reinit noise
    a_pt=1.0,
    alphas=alphas,
    mc=80_000,
    seed=seeds,
    gamma_ext=1e-6,
    tol=1e-6,
    max_iters=900,
    damp=0.25,
)


df_replica_ptft.to_csv("replica_ptft_curves.csv", index=False)