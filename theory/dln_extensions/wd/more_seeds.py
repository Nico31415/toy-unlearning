"""Tighter checks: larger d, many seeds. End points are computed exactly at finite d (convex solves); the identity
'gradient descent on the weights = this end point' is deterministic and is spot-checked separately (part E)."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, pickle, zlib, time, numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from ce_tests import project
rho, aPT, aUL = 0.2, 0.3, 0.2
LAMS = (0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5); L1 = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0); c0 = 0.05

def teacher(d, rng): return np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)

def jobA(a):                                   # switched-off weight decay in stage 2 (deleted -> 0, no anchor)
    k, seed, d = a; rng = np.random.default_rng(zlib.crc32(repr(("A", seed)).encode())); c = np.sqrt(k) / 2
    bs = teacher(d, rng); n, f = int(aPT * d), int(aUL * d); X = rng.standard_normal((d, n)) / np.sqrt(d); Xf, XK = X[:, :f], X[:, f:]
    A = project(np.zeros(d), c, A_eq=X, b_eq=X.T @ bs)[0]; Au = project(np.zeros(d), c, A_eq=XK, b_eq=XK.T @ bs)[0]; nb = bs @ bs; zk = XK.T @ bs
    out = []
    for Lam in LAMS:
        kap = np.exp(-2 * Lam); U = project(kap * A, kap * c, A_eq=Xf, b_eq=np.zeros(f))[0]
        out.append([np.sum((U - Au)**2) / nb, U @ U / nb, np.mean((XK.T @ U - zk)**2) / np.mean(zk**2)])
    return ("A", k, seed, np.array(out))

def jobB(a):                                   # switched-off one-sided decay in pretraining
    k, seed, d = a; rng = np.random.default_rng(zlib.crc32(repr(("B", seed)).encode())); c = np.sqrt(k) / 2
    bs = teacher(d, rng); X = rng.standard_normal((d, int(aPT * d))) / np.sqrt(d); y = X.T @ bs; out = []
    for Lam in L1:
        cp = c * np.exp(-Lam); b = project(np.full(d, cp * np.sinh(-Lam)), cp, A_eq=X, b_eq=y)[0]; m = b @ bs / (bs @ bs)
        out.append([b @ b / (bs @ bs), m, np.mean(b - m * bs)])
    return ("B", k, seed, np.array(out))

REG = {"hyp": (lambda b: b * np.arcsinh(b / c0) - np.sqrt(b * b + c0 * c0), lambda b: np.arcsinh(b / c0), (0.02, 0.053, 0.103)),
       "bridge": (lambda b: 2 / 3 * np.abs(b)**1.5, lambda b: np.sign(b) * np.sqrt(np.abs(b)), (0.08, 0.29, 0.43)),
       "l1": (None, None, (0.05, 0.15, 0.3))}
def jobC(a):                                   # constant explicit regulariser, two-group problem
    name, seed, d = a; rng = np.random.default_rng(zlib.crc32(repr(("C", seed, d)).encode()))
    bs = teacher(d, rng); n, f = int(aPT * d), int(aUL * d); X = rng.standard_normal((d, n)) / np.sqrt(d); z = X.T @ bs; t = z.copy(); t[:f] = 0
    R0, R1, lams = REG[name]; out = []
    for lam in lams:
        if name == "l1":
            from se_wd import lasso; b = lasso(X, t, np.ones(n), lam, iters=60000)
        else:
            obj = lambda b: (0.5 * np.sum((X.T @ b - t)**2) + lam * np.sum(R0(b)), X @ (X.T @ b - t) + lam * R1(b))
            b = minimize(obj, np.full(d, 1e-3), jac=True, method="L-BFGS-B", options=dict(maxiter=40000, maxfun=80000, ftol=1e-16, gtol=1e-10, maxcor=30)).x
        r = X.T @ b - t; out.append([b @ b / (bs @ bs), b @ bs / (bs @ bs), np.mean(r[f:]**2) / np.mean(z**2)])
    return ("C", (name, d), seed, np.array(out))

def run(j): return {"A": jobA, "B": jobB, "C": jobC}[j[0]](j[1])
if __name__ == "__main__":
    jobs = [("A", (k, s, 1600)) for k in (1e-2, 1.0) for s in range(16)] + [("B", (k, s, 1600)) for k in (1e-2, 1.0) for s in range(24)] \
         + [("C", (nm, s, dd)) for nm in REG for dd in (1500, 6000) for s in range(12 if dd == 1500 else 6)]
    res, t0 = [], time.time()
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(run, jobs)):
            res.append(r); pickle.dump(res, open("more_seeds.pkl", "wb"))
            if i % 20 == 19: print(i + 1, len(jobs), f"{time.time()-t0:.0f}s", flush=True)
    print("DONE")
