"""Are ascent losses 'hidden, not deleted'? All-forget coordinates, CE toy (teacher logits gam*x^T b*), deleted samples reused.
For each method: the model U, its squared logit on fresh inputs (= |U|^2/|bs|^2), held-out accuracy, cosine to the untraining reference,
and a relearning attack: Bregman projection from U onto n_re fresh correctly-labelled samples, then test error."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd", "crit"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, zlib, pickle, numpy as np
from scipy.special import expit, log_expit
from multiprocessing import Pool
from ce_tests import project
rho, GAM, d, aPT, aUL = 0.2, 3.0, 400, 0.3, 0.2
ARE = (0.05, 0.15)

def ascend(A, c, Zd, m0, kind, stops, Xho, yho, tmax):
    """mirror flow of the ascent loss; returns the model at the first time held-out accuracy <= each stop level."""
    th = np.arcsinh(A / c); t = 0.0; L2 = np.linalg.norm(Zd, 2) ** 2; out = {}; todo = sorted(stops.items(), key=lambda kv: -kv[1])
    for it in range(400_000):
        B = c * np.sinh(th)
        if it % 2 == 0:
            acc = np.mean(np.sign(Xho.T @ B) == yho)
            while todo and acc <= todo[0][1]:
                out[todo[0][0]] = B.copy(); todo.pop(0)
            if not todo: break
        m = Zd.T @ B
        if kind == "GA": dm = expit(-m); cur = 0.25
        else: dm = 2 * expit(4 * (log_expit(m) - log_expit(m0))) * expit(-m); cur = 2.5
        g = Zd @ dm; mx = np.max(np.sqrt(B * B + c * c)); eta = 0.4 / (4 * mx * cur * L2); gn = np.max(np.abs(g)) * 4 * eta
        if gn > 0.3: eta *= 0.3 / gn
        th -= 4 * eta * g; t += eta
        if t > tmax or np.abs(th).max() > 40: break
    return out

def cell(a):
    k, seed = a; c = np.sqrt(k) / 2; rng = np.random.default_rng(zlib.crc32(repr(("T", seed)).encode()))
    bs = GAM * np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0); nb = bs @ bs
    n, f = int(aPT * d), int(aUL * d); X = rng.standard_normal((d, n)) / np.sqrt(d); Xd, XK = X[:, :f], X[:, f:]
    A = project(np.zeros(d), c, A_eq=X, b_eq=X.T @ bs)[0]; Au = project(np.zeros(d), c, A_eq=XK, b_eq=XK.T @ bs)[0]
    Xho = rng.standard_normal((d, 4000)) / np.sqrt(d); yho = np.sign(Xho.T @ bs); acc = lambda B: np.mean(np.sign(Xho.T @ B) == yho) if B @ B > 0 else 0.5
    yd = np.sign(Xd.T @ bs); Zd = Xd * yd; m0 = Zd.T @ A
    M = {"do nothing": A, "retrained without F (untraining ref.)": Au, "retrained without the class (unlearning ref.)": np.zeros(d),
         "uniform target, converged": project(A, c, A_eq=Xd, b_eq=np.zeros(f))[0]}
    stops = {"chance": 0.52, "ref": acc(Au)}
    for kind, tmax in (("GA", 12.0), ("NPO", 200.0)):
        o = ascend(A, c, Zd, m0, kind, stops, Xho, yho, tmax)
        for s, B in o.items(): M[f"{kind}, stopped at {'chance accuracy' if s == 'chance' else 'the untraining ref. accuracy'}"] = B
    Xre = rng.standard_normal((d, int(max(ARE) * d))) / np.sqrt(d); rows = {}
    for name, U in M.items():
        r = dict(logit2=U @ U / nb, acc=acc(U), dist_unt=np.sum((U - Au) ** 2) / nb, cos_unt=(U @ Au) / np.sqrt((U @ U) * (Au @ Au) + 1e-300))
        for are in ARE:
            Xr = Xre[:, :int(are * d)]; V = project(U, c, A_eq=Xr, b_eq=Xr.T @ bs)[0]
            r[f"re_err_{are}"] = np.sum((V - bs) ** 2) / nb; r[f"re_acc_{are}"] = acc(V)
        rows[name] = r
    return (k, seed, rows)

if __name__ == "__main__":
    jobs = [(k, s) for k in (1e-6, 1e-2, 1.0) for s in range(10)]; out = []
    with Pool(1) as p:
        for r in p.imap_unordered(cell, jobs):
            out.append(r); pickle.dump(out, open("tamper.pkl", "wb"))
    import pandas as pd; pd.set_option("display.width", 250)
    for k in (1e-6, 1e-2, 1.0):
        rows = [dict(method=nm, **v) for kk, s, R in out if kk == k for nm, v in R.items()]
        df = pd.DataFrame(rows); g = df.groupby("method", sort=False); m = g.mean(); se = g.sem(); cnt = g.size()
        print(f"\nk = {k:g}   (mean over seeds; n = runs that reached the stop)")
        t = pd.DataFrame({"n": cnt, "held-out acc": m["acc"].round(3), "sq. logit": m["logit2"].round(3), "dist to untr. ref": m["dist_unt"].round(3), "cos to untr. ref": m["cos_unt"].round(3),
                          "relearn err a=.05": m["re_err_0.05"].round(3).astype(str) + "±" + se["re_err_0.05"].round(3).astype(str), "relearn err a=.15": m["re_err_0.15"].round(3).astype(str) + "±" + se["re_err_0.15"].round(3).astype(str),
                          "relearn acc a=.15": m["re_acc_0.15"].round(3)})
        print(t.to_string())
