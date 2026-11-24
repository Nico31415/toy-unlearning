import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd", "crit"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import numpy as np, pickle, sys
from dmft2 import run, setup, metrics
rho, aPT, aUL, dt, T2, k, T1 = 0.2, 0.3, 0.2, 0.05, 6.0, 1e-6, 10.0
ts, P, W, A, i1 = setup(aPT, aUL, T1, T2, dt); out = []
for seed in range(10, 17):
    th = run(aPT, P, W, A, ts, k, rho, N=2000, seed=seed); un, ul = metrics(th["Q"], th["M"], i1, len(ts) - 2)
    out.append((un[i1:], ul[i1:])); pickle.dump(out, open("es_avg.pkl", "wb")); print(seed, round(float(un[i1]), 3), round(float(un[i1 + 29]), 3), round(float(un[-1]), 3), flush=True)
print("DONE")
