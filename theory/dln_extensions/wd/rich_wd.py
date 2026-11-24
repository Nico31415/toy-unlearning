import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd", "crit"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, numpy as np, pickle
from more_seeds import jobA, LAMS, rho, aPT, aUL
from se_reuse import stage1_grid, solve_unlearn, solve_untrain
k = 1e-6; R = np.array([jobA((k, s, 800))[3] for s in range(12)]); m, e = R.mean(0), R.std(0, ddof=1) / np.sqrt(12)
st = stage1_grid(aPT, k, rho, n_g=61); zs = []
print("Lam | untraining th/sim±se | unlearning th/sim±se | kept loss th/sim±se")
for i, Lam in enumerate(LAMS):
    r = solve_unlearn(aPT, aUL, k, rho, True, st=st, n_z=61, S=np.exp(-2 * Lam)); u = solve_untrain(aPT, aUL, k, rho, st, r, True, n_e=31); th = [u["un"], r["U2"], r["kept"]]
    zs += [(m[i, j] - th[j]) / max(e[i, j], 1e-12) for j in range(3) if th[j] > 2e-3]
    print(f"{Lam:.2f} | " + " | ".join(f"{th[j]:.4f}/{m[i,j]:.4f}±{e[i,j]:.4f}" for j in range(3)), flush=True)
zs = np.array(zs); print(f"{len(zs)} numbers: |z|<2 in {100*np.mean(np.abs(zs)<2):.0f}%, max |z| = {np.abs(zs).max():.1f}")
