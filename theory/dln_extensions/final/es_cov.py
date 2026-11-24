"""Early-stopped stage 2 with non-zero targets and with an anchor (squared loss): theory (dmft2) vs simulation, and the best loss at each budget.
alpha_PT = 0.3, alpha_UL = 0.2, rho = 0.2. Targets on the deleted samples: b * y. Anchor: kept samples stay in the loss at their labels."""
import sys, numpy as np, pickle; sys.path.insert(0, "/home/claude/out/ce")
from multiprocessing import Pool
from dmft2 import run, simulate, setup, metrics
k, T1 = float(sys.argv[1]), float(sys.argv[2]); dt, T2, rho, aPT, aUL = 0.05, 4.0, 0.2, 0.3, 0.2
CFG = [(0.0, False), (-0.5, False), (0.5, False), (0.0, True), (-0.5, True)]
def job(a):
    kind, b, anc, seed = a; ts, P, W, A, i1 = setup(aPT, aUL, T1, T2, dt, b_target=b, anchor=anc); last = len(ts) - 2
    if kind == "th":
        th = run(aPT, P, W, A, ts, k, rho, N=2000, seed=seed); un, ul = metrics(th["Q"], th["M"], i1, i1); return un[:last + 1], ul[:last + 1]
    r = simulate(aPT, P, W, A, ts, k, rho, d=1500, seed=seed); bb = r["beta"]; return (np.sum((bb[0] - bb[1][:, [i1]])**2, 0) / r["nb"])[:last + 1], (np.sum(bb[0]**2, 0) / r["nb"])[:last + 1]
if __name__ == "__main__":
    nt, nsim = 6, 8; jobs = [(w, b, anc, s) for (b, anc) in CFG for w, n in (("th", nt), ("si", nsim)) for s in range(n)]
    with Pool(2) as p: R = p.map(job, jobs)
    out = {}; i = 0
    for cfg in CFG:
        out[cfg] = dict(th=np.array(R[i:i + nt]), si=np.array(R[i + nt:i + nt + nsim])); i += nt + nsim
    pickle.dump(dict(out=out, dt=dt, i1=int(round(T1 / dt)), k=k), open(f"es_cov_{k:g}.pkl", "wb")); i1 = int(round(T1 / dt))
    print(f"k={k:g}: untraining distance ||beta(t) - A(D\\F)||^2/||beta*||^2, reference = retrained for T1. theory / sim (z)")
    for cfg in CFG:
        row = f"target {cfg[0]:+.1f} y, anchor {str(cfg[1]):5s} |"
        for off in (0, 0.25, 0.5, 1, 2, 3.9):
            j = i1 + int(round(off / dt)); a, b = out[cfg]["th"][:, 0, j], out[cfg]["si"][:, 0, j]
            row += f" t2={off:<4g} {a.mean():.3f}/{b.mean():.3f} ({(a.mean()-b.mean())/np.hypot(a.std(ddof=1)/np.sqrt(nt), b.std(ddof=1)/np.sqrt(nsim)):+.1f}) |"
        print(row, flush=True)
