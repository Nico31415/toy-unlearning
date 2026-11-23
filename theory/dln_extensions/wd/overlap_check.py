import numpy as np, pickle, zlib
from multiprocessing import Pool
from se_wd import lasso, solve, prox_l1
rho, aPT, aUL, d = 0.2, 0.3, 0.2, 3000; lams = (0.05, 0.15, 0.3)
def job(seed):
    rng = np.random.default_rng(zlib.crc32(repr(("OV", seed)).encode())); bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
    n, f = int(aPT * d), int(aUL * d); X = rng.standard_normal((d, n)) / np.sqrt(d); z = X.T @ bs; t = z.copy(); t[:f] = 0
    return [[(b := lasso(X, t, np.ones(n), lam, iters=60000)) @ bs / d, b @ b / d, bs @ bs / d] for lam in lams]
if __name__ == "__main__":
    with Pool(2) as p: R = np.array(p.map(job, range(40)))
    G = [(aUL / aPT, 1.0, 0.0), (1 - aUL / aPT, 1.0, 1.0)]
    for i, lam in enumerate(lams):
        th = solve(aPT, G, lam, rho); m = R[:, i, 0]; q = R[:, i, 1]
        print(f"lam={lam}: overlap E[b.b*]/d: sim {m.mean():.5f} ± {m.std(ddof=1)/np.sqrt(len(m)):.5f}, theory {th['M']:.5f}, diff {100*(m.mean()/th['M']-1):+.2f}% | norm: sim {q.mean():.5f} ± {q.std(ddof=1)/np.sqrt(len(q)):.5f}, theory {th['Q']:.5f}, diff {100*(q.mean()/th['Q']-1):+.2f}%")
