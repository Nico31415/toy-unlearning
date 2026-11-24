import numpy as np, sys, pickle
from multiprocessing import Pool
from dmft_nl import run, simulate
k = float(sys.argv[1]); Tend = float(sys.argv[2]); n = int(sys.argv[3]); al, rho, dt = 0.3, 0.2, 0.05; NN = int(sys.argv[4]) if len(sys.argv) > 4 else 3000; ts = np.arange(0, Tend + dt / 2, dt)
def th(s): return run(al, ts, k, rho, N=NN, seed=s)
def si(s): return simulate(al, ts, k, rho, d=NN, seed=s)
if __name__ == "__main__":
    with Pool(2) as p: TH = p.map(th, range(100, 100 + n)); SI = p.map(si, range(100, 100 + n))
    pickle.dump(dict(ts=ts, TH=TH, SI=SI), open(f"nl_{k:g}_{NN}.pkl", "wb"))
    st = lambda L, key: (np.mean([o[key] for o in L], 0), np.std([o[key] for o in L], 0, ddof=1) / np.sqrt(len(L)))
    print(f"k={k:g}, {n} runs each. theory±se / sim±se, (z)")
    for tt in np.linspace(Tend / 5, Tend - dt, 5):
        i = int(round(tt / dt)) - 1; row = f"t={ts[i]:5.2f} |"
        for key in ("Q", "M", "loss", "acc"):
            (a, ae), (b, be) = st(TH, key), st(SI, key); row += f" {key} {a[i]:.4f}±{ae[i]:.4f} / {b[i]:.4f}±{be[i]:.4f} ({(a[i]-b[i])/np.hypot(ae[i],be[i])+0:.1f}) |"
        print(row)
