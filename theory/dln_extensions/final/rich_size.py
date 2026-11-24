"""k = 1e-6: does the 11% offset in the untraining distance at the start of stage 2 shrink with system size?"""
import sys, numpy as np; sys.path.insert(0, "../ce")
from multiprocessing import Pool
from dmft2 import run, simulate, setup, metrics
k, T1, dt, rho, aPT, aUL = 1e-6, 10.0, 0.05, 0.2, 0.3, 0.2
ts, P, W, A, i1 = setup(aPT, aUL, T1, 0.1, dt)
def job(a):
    kind, size, seed = a
    if kind == "th": th = run(aPT, P, W, A, ts, k, rho, N=size, seed=seed + 50); return metrics(th["Q"], th["M"], i1, i1)[0][i1]
    r = simulate(aPT, P, W, A, ts, k, rho, d=size, seed=seed + 50); b = r["beta"]; return np.sum((b[0][:, i1] - b[1][:, i1])**2) / r["nb"]
if __name__ == "__main__":
    for kind, size, n in (("si", 800, 12), ("si", 1500, 12), ("si", 4000, 8), ("si", 8000, 6), ("th", 1000, 8), ("th", 2500, 6)):
        with Pool(2) as p: v = np.array(p.map(job, [(kind, size, s) for s in range(n)]))
        print(f"{'simulation d' if kind == 'si' else 'theory N'} = {size:5d}: {v.mean():.4f} ± {v.std(ddof=1)/np.sqrt(n):.4f}  ({n} runs)", flush=True)
