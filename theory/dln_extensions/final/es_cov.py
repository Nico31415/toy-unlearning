"""Early-stopped stage 2 with non-zero targets and with an anchor (squared loss): theory (dmft2) vs simulation, and the best loss at each budget.
alpha_PT = 0.3, alpha_UL = 0.2, rho = 0.2. Targets on the deleted samples: b * y. Anchor: kept samples stay in the loss at their labels."""
import sys, numpy as np, pickle; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "ce"))
from multiprocessing import Pool
from dmft2 import run, simulate, setup, metrics
k, T1 = float(sys.argv[1]), float(sys.argv[2]); dt, T2, rho, aPT, aUL = (float(sys.argv[3]) if len(sys.argv) > 3 else 0.05), (float(sys.argv[4]) if len(sys.argv) > 4 else 4.0), 0.2, 0.3, 0.2
NT, NSIM, NN = (int(sys.argv[5]) if len(sys.argv) > 5 else 6), (int(sys.argv[6]) if len(sys.argv) > 6 else 8), (int(sys.argv[7]) if len(sys.argv) > 7 else 2000)
CFG = [(0.0, False), (-0.5, False), (0.5, False), (0.0, True), (-0.5, True)]
def job(a):
    kind, b, anc, seed = a; ts, P, W, A, i1 = setup(aPT, aUL, T1, T2, dt, b_target=b, anchor=anc); last = len(ts) - 2
    if kind == "th":
        th = run(aPT, P, W, A, ts, k, rho, N=NN, seed=seed); un, ul = metrics(th["Q"], th["M"], i1, i1); return un[:last + 1], ul[:last + 1]
    r = simulate(aPT, P, W, A, ts, k, rho, d=1500, seed=seed); bb = r["beta"]; return (np.sum((bb[0] - bb[1][:, [i1]])**2, 0) / r["nb"])[:last + 1], (np.sum(bb[0]**2, 0) / r["nb"])[:last + 1]
if __name__ == "__main__":
    nt, nsim = NT, NSIM; jobs = [(w, b, anc, s) for (b, anc) in CFG for w, n in (("th", nt), ("si", nsim)) for s in range(n)]
    with Pool(2) as p: R = p.map(job, jobs)
    out = {}; i = 0
    for cfg in CFG:
        out[cfg] = dict(th=np.array(R[i:i + nt]), si=np.array(R[i + nt:i + nt + nsim])); i += nt + nsim
    pickle.dump(dict(out=out, dt=dt, i1=int(round(T1 / dt)), k=k), open(f"es_cov_{k:g}.pkl", "wb")); i1 = int(round(T1 / dt))
    print(f"k={k:g}: untraining distance ||beta(t) - A(D\\F)||^2/||beta*||^2, reference = retrained for T1. theory / sim (z)")
    for cfg in CFG:
        row = f"target {cfg[0]:+.1f} y, anchor {str(cfg[1]):5s} |"
        for off in [round(x * T2, 3) for x in (0, 0.0625, 0.125, 0.25, 0.5, 0.975)]:
            j = i1 + int(round(off / dt)); a, b = out[cfg]["th"][:, 0, j], out[cfg]["si"][:, 0, j]
            row += f" t2={off:<4g} {a.mean():.3f}/{b.mean():.3f} ({(a.mean()-b.mean())/np.hypot(a.std(ddof=1)/np.sqrt(nt), b.std(ddof=1)/np.sqrt(nsim)):+.1f}) |"
        print(row, flush=True)
