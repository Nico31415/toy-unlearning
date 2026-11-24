"""CE toy as a function of k. All coordinates are forget coordinates. Pretraining: CE on teacher probabilities sigma(gamma x^T b*)
(= projection onto logits gamma x^T b*). Stage 2 from A(D) on the deleted samples (reused), hard labels for ascent losses.
Theory (exact, large d) for the uniform-target loss from se_reuse with k_eff = k / gamma^2."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, pickle, zlib, time, warnings
import numpy as np
from scipy.special import expit, log_expit
from multiprocessing import Pool
warnings.simplefilter("ignore")
from ce_tests import project
RHO, GAM, D, APT = 0.2, 3.0, 400, 0.3
KS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4)
AUL = (0.1, 0.2)

def cell(args):
    k, aul, seed = args
    c = np.sqrt(k) / 2; d = D
    rng = np.random.default_rng(zlib.crc32(repr(("KS", aul, seed)).encode()))       # same data for every k
    bs = GAM * np.where(rng.random(d) < RHO, rng.choice([-1, 1], d) * RHO ** -0.5, 0.0)
    n, f = int(APT * d), int(aul * d)
    X = rng.standard_normal((d, n)) / np.sqrt(d); Xd, XK = X[:, :f], X[:, f:]
    A = project(np.zeros(d), c, A_eq=X, b_eq=X.T @ bs)[0]
    Au = project(np.zeros(d), c, A_eq=XK, b_eq=XK.T @ bs)[0]
    nb = bs @ bs
    yd = np.sign(Xd.T @ bs); Zd = Xd * yd; m0 = Zd.T @ A
    Xho = rng.standard_normal((d, 1000)) / np.sqrt(d); yho = np.sign(Xho.T @ bs)
    met = lambda B: np.array([np.sum((B - Au) ** 2) / nb, B @ B / nb, np.mean(Zd.T @ B > 0), np.mean(np.sign(Xho.T @ B) == yho)])
    out = dict(k=k, aul=aul, seed=seed)
    for name, B in (("nothing", A), ("ref_untrain", Au), ("uniform", project(A, c, A_eq=Xd, b_eq=np.zeros(f))[0])):
        out[name] = met(B)
    L2 = np.linalg.norm(Xd, 2) ** 2
    for loss in ("GA", "NPO4"):
        th = np.arcsinh(A / c); t = 0.0; H = []
        TMAX = 8.0 if loss == "GA" else 80.0
        for it in range(200_000):
            B = c * np.sinh(th)
            if it % 3 == 0: H.append(met(B))
            m = Zd.T @ B
            if loss == "GA": dm = expit(-m); cur = 0.25
            else: dm = 2 * expit(4 * (log_expit(m) - log_expit(m0))) * expit(-m); cur = 2.5
            g = Zd @ dm
            mx = np.max(np.sqrt(B ** 2 + c ** 2)); eta = 0.4 / (4 * mx * cur * L2)
            gn = np.max(np.abs(g)) * 4 * eta
            if gn > 0.3: eta *= 0.3 / gn
            th -= 4 * eta * g; t += eta
            if t > TMAX or np.abs(th).max() > 40 or B @ B / nb > 50: break
        H = np.array(H)
        i50 = np.argmax(H[:, 2] <= 0.5) if (H[:, 2] <= 0.5).any() else len(H) - 1
        out[loss] = dict(best_untrain=H[:, 0].min(), best_unlearn=H[:, 1].min(), at50=H[i50], end=H[-1], T=t)
    return out

def theory():
    from se_reuse import stage1_grid, solve_unlearn, solve_untrain
    rows = []
    for k in KS:
        ke = k / GAM ** 2
        st = stage1_grid(APT, ke, RHO, n_g=41)
        nA = float(np.sum(st["W"] * st["A"] ** 2) / np.sum(st["W"] * st["B"] ** 2))
        for aul in AUL:
            r = solve_unlearn(APT, aul, ke, RHO, True, st=st, n_z=41)
            u = solve_untrain(APT, aul, ke, RHO, st, r, True, n_e=25) if r.get("phase") == "ok" else {}
            rows.append(dict(k=k, aul=aul, nothing_unlearn=nA, nothing_untrain=u.get("dn", np.nan),
                             uniform_unlearn=r.get("U2", np.nan), uniform_untrain=u.get("un", np.nan), phase=r.get("phase")))
            print(rows[-1], flush=True)
    pickle.dump(rows, open("ksweep_theory.pkl", "wb"))

if __name__ == "__main__":
    if sys.argv[1] == "theory":
        theory()
    else:
        jobs = [(k, a, s) for s in range(4) for k in KS for a in AUL]
        out = []; t0 = time.time()
        with Pool(2) as p:
            for i, r in enumerate(p.imap_unordered(cell, jobs)):
                out.append(r); pickle.dump(out, open("ksweep_sim.pkl", "wb"))
                if i % 8 == 7: print(i + 1, len(jobs), f"{time.time() - t0:.0f}s", flush=True)
        print("DONE")
