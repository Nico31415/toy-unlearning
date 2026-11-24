import numpy as np, pickle
from multiprocessing import Pool
from se_logistic import solve, prox_l1, sim
rho, d, ns = 0.2, 1000, 8; lams = [0.03, 0.06, 0.12, 0.25, 0.5]; fine = np.geomspace(0.6, 0.025, 30)
if __name__ == "__main__":
    out = {}
    for al in (0.3, 1.5):
        th = []; init = (1.0, 0.3, 1.0)
        for l in fine:
            r = solve(al, l, rho, np.inf, prox_l1, init=init); init = (r["Q"], r["M"], r["V"]); th.append([r["Q"], r["M"], r["loss"], r["acc"]])
        with Pool(2) as p: S = {l: np.array(p.map(sim, [(al, l, np.inf, "l1", s, d, rho) for s in range(ns)])) for l in lams}
        out[al] = dict(fine=fine, th=np.array(th), S=S); print(al, "done", flush=True)
    pickle.dump(out, open("lam_sweep.pkl", "wb"))
