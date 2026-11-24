"""CE toy, industry ascent losses vs target losses. Pretraining: CE on teacher probabilities sigma(gamma x^T b*) -> logits gamma x^T b*
(exactly the MSE toy scaled by gamma). Stage 2 from A(D): hard labels y = sign(x^T b*) on the deleted samples."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import numpy as np, zlib, pickle, sys, time, itertools
from scipy.special import expit, log_expit
from multiprocessing import Pool
from ce_tests import project
RHO, GAM, K = 0.2, 4.0, 1e-6

def cell(args):
    dSh, apt, phi, seed = args
    dR = dF = 100; d = dR + dSh + dF; c = np.sqrt(K) / 2
    rng = np.random.default_rng(zlib.crc32(repr(("G", dSh, apt, phi, seed)).encode()))
    bs = GAM * np.where(rng.random(d) < RHO, rng.choice([-1, 1], d) * RHO ** -0.5, 0.0)     # teacher logits = x^T bs
    nR, nF = int(1.5 * (dR + dSh)), int(round(apt * (dSh + dF)))
    XR = np.zeros((d, nR)); XR[:dR + dSh] = rng.standard_normal((dR + dSh, nR)) / np.sqrt(d)
    XF = np.zeros((d, nF)); XF[dR:] = rng.standard_normal((dSh + dF, nF)) / np.sqrt(d)
    f = int(round(phi * nF)); Xd, XK = XF[:, :f], XF[:, f:]
    P = lambda Xs: project(np.zeros(d), c, A_eq=Xs, b_eq=Xs.T @ bs)[0]
    A, A_unt, A_cls = P(np.hstack([XR, XF])), P(np.hstack([XR, XK])), P(XR)
    nf2 = np.sum(bs[dR:] ** 2)
    yd = np.sign(Xd.T @ bs); Zd = Xd * yd; m0 = Zd.T @ A
    Xho = np.zeros((d, 500)); Xho[dR:] = rng.standard_normal((dSh + dF, 500)) / np.sqrt(d)
    def met(B):
        return np.array([np.sum((B - A_unt) ** 2) / nf2, np.sum((B - A_cls) ** 2) / nf2,
                         np.mean(Zd.T @ B > 0), np.mean(np.sign(Xho.T @ B) == np.sign(Xho.T @ bs))])
    rows = [dict(loss="do nothing", anchor="-", T=0.0, best_untrain=met(A)[0], best_unlearn=met(A)[1], t_untrain=0, t_unlearn=0,
                 acc_del=1.0, acc_ho=met(A)[3])]
    ref = dict(untrain_ref_acc_del=met(A_unt)[2], unlearn_ref_acc_del=met(A_cls)[2], untrain_ref_ho=met(A_unt)[3], unlearn_ref_ho=met(A_cls)[3])
    anchors = {"none": np.zeros((d, 0)), "retain": XR, "all": np.hstack([XR, XK])}
    for loss, an in itertools.product(("GA", "NPO1", "NPO4", "uniform"), anchors):
        Xa = anchors[an]; ta = expit(Xa.T @ bs)
        X = np.hstack([Xd, Xa]); L2 = np.linalg.norm(X, 2) ** 2
        th = np.arcsinh(A / c); t = 0.0; hist = []
        TMAX = 60.0 if loss != "GA" else 6.0
        for it in range(150_000):
            B = c * np.sinh(th)
            if it % 4 == 0: hist.append((t, *met(B)))
            m = Zd.T @ B
            if loss == "GA": dm = expit(-m); cur = 0.25
            elif loss == "uniform": dm = None; cur = 0.25
            else:
                bn = float(loss[3:]); dm = 2 * expit(bn * (log_expit(m) - log_expit(m0))) * expit(-m); cur = 0.5 * (1 + bn)
            g = Zd @ dm if dm is not None else Xd @ (expit(Xd.T @ B) - 0.5)
            if Xa.shape[1]: g = g + Xa @ (expit(Xa.T @ B) - ta)
            mx = np.max(np.sqrt(B ** 2 + c ** 2)); eta = 0.4 / (4 * mx * cur * L2)
            gn = np.max(np.abs(g)) * 4 * eta
            if gn > 0.3: eta *= 0.3 / gn
            th -= 4 * eta * g; t += eta
            if t > TMAX or np.abs(th).max() > 40: break
        H = np.array(hist); iu, il = H[:, 1].argmin(), H[:, 2].argmin()
        rows.append(dict(loss=loss, anchor=an, T=H[-1, 0], best_untrain=H[iu, 1], t_untrain=H[iu, 0], accdel_at_untrain=H[iu, 3],
                         best_unlearn=H[il, 2], t_unlearn=H[il, 0], accdel_at_unlearn=H[il, 3], end_untrain=H[-1, 1], end_unlearn=H[-1, 2]))
    return [dict(dSh=dSh, apt=apt, phi=phi, seed=seed, **ref, **r) for r in rows]

if __name__ == "__main__":
    jobs = [(dSh, a, p, s) for s in range(3) for dSh in (0, 20) for a in (0.3, 0.75) for p in (0.3, 0.6)]
    if sys.argv[1] == "time":
        t0 = time.time(); r = cell(jobs[0]); print(time.time() - t0)
        import pandas as pd; pd.set_option("display.width", 250); print(pd.DataFrame(r).drop(columns=["dSh", "apt", "phi", "seed"]).round(3).to_string()); sys.exit()
    out = []
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(cell, jobs)):
            out += r; pickle.dump(out, open("ce_grid.pkl", "wb")); print(i + 1, len(jobs), flush=True)
    print("DONE")
