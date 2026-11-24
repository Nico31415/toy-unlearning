"""CE toy: test the margin-Bregman conjecture (Test 0 pretraining, Test 1 stage 2 targets) and NPO-as-margin-shift (Test 2)."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, pickle, time, warnings, zlib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from regpath import regpath
warnings.simplefilter("ignore")

dR, dF, RHO = 100, 100, 0.2
d = dR + dF

def data(seed, apt):
    rng = np.random.default_rng(zlib.crc32(repr(("CE", seed, apt)).encode()))
    amp = RHO ** -0.5
    bs = np.where(rng.random(d) < RHO, rng.choice([-amp, amp], d), 0.0)
    nR, nF = 150, int(round(apt * dF))
    XR = np.zeros((d, nR)); XR[:dR] = rng.standard_normal((dR, nR)) / np.sqrt(d)
    XF = np.zeros((d, nF)); XF[dR:] = rng.standard_normal((dF, nF)) / np.sqrt(d)
    yR, yF = np.sign(XR.T @ bs), np.sign(XF.T @ bs)
    yR[yR == 0] = 1; yF[yF == 0] = 1
    return bs, XR, yR, XF, yF

# ------------------------------------------------------------------ Bregman projection with linear (in)equalities
def project(b0, c, A_eq=None, b_eq=None, A_ge=None, b_ge=None):
    """argmin D_Q(beta, b0) s.t. A_eq^T beta = b_eq, A_ge^T beta >= b_ge.  Q' = arcsinh(beta/c) (up to a constant factor)."""
    th0 = np.arcsinh(b0 / c)
    A = np.hstack([M for M in (A_eq, A_ge) if M is not None])
    b = np.r_[[] if b_eq is None else b_eq, [] if b_ge is None else b_ge]
    ne = 0 if A_eq is None else A_eq.shape[1]
    def h(v):
        th = th0 + A @ v
        th = np.clip(th, -700, 700)
        return np.sum(c * np.cosh(th)) - v @ b, A.T @ (c * np.sinh(th)) - b
    bounds = [(None, None)] * ne + [(0, None)] * (A.shape[1] - ne)
    r = minimize(h, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B", bounds=bounds,
                 options=dict(maxiter=50000, maxfun=100000, ftol=1e-15, gtol=1e-11, maxcor=50))
    beta = c * np.sinh(th0 + A @ r.x)
    viol = A.T @ beta - b
    v = max(np.max(np.abs(viol[:ne])) if ne else 0, -min(0, viol[ne:].min()) if A.shape[1] > ne else 0)
    return beta, v

# ------------------------------------------------------------------ mirror flow, d/dt arcsinh(beta/c) = -4 grad L
_LXX = {}

def flow(b0, c, grad_curv, stop, record, step=0.4, maxsteps=400_000):
    """grad_curv(beta) -> (grad_beta, curvature weight vector w (per sample), X used for curvature).
    stop(beta, t) -> bool. record(beta, t) called every step (cheap)."""
    th = np.arcsinh(b0 / c); t = 0.0
    for it in range(maxsteps):
        beta = c * np.sinh(th)
        g, X, w = grad_curv(beta)
        record(beta, t)
        if stop(beta, t):
            return beta, t, it
        key = id(X)
        if key not in _LXX:
            _LXX[key] = (X, np.linalg.norm(X, 2) ** 2)
        lam = np.max(w) * _LXX[key][1] + 1e-12
        mx = np.max(np.sqrt(beta ** 2 + c ** 2))
        eta = step / (4 * mx * lam)
        gn = np.max(np.abs(g)) * 4 * eta
        if gn > 0.5:                      # cap the change in theta per step
            eta *= 0.5 / gn
        th -= 4 * eta * g
        t += eta
    return c * np.sinh(th), t, maxsteps

def logistic_parts(X, y, beta):
    m = y * (X.T @ beta)
    return m, -(X @ (y * expit(-m))), expit(m) * expit(-m)

def rel(a, b, start):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a - start), 1e-300)

def cos(a, b):
    return a @ b / np.linalg.norm(a) / np.linalg.norm(b)

# ------------------------------------------------------------------ Test 0: pretraining
def pretrain(k, seed, apt, eps_list=(1e-2, 1e-3)):
    c = np.sqrt(k) / 2
    bs, XR, yR, XF, yF = data(seed, apt)
    X, y = np.hstack([XR, XF]), np.r_[yR, yF]
    n = len(y)
    out, todo = [], list(eps_list)
    def gc(beta):
        m, g, w = logistic_parts(X, y, beta)
        return g, X, w
    def rec(beta, t):
        if todo:
            L = np.mean(np.logaddexp(0, -y * (X.T @ beta)))
            if L <= todo[0]:
                eps = todo.pop(0)
                m = y * (X.T @ beta); gam = m.min()
                P, v = project(np.zeros(d), c, A_ge=X * y, b_ge=np.full(n, gam))
                def lossf(b):
                    mm = y * (X.T @ b)
                    return np.mean(np.logaddexp(0, -mm)), -(X @ (y * expit(-mm))) / n
                R, ll = regpath(lossf, np.zeros(d), c, L)
                out.append(dict(k=k, seed=seed, apt=apt, eps=eps, t=t, gamma=gam, err=rel(beta, P, 0 * beta),
                                err_reg=rel(beta, R, 0 * beta), cos=cos(beta, P), viol=v,
                                frac_sv=np.mean(m < gam * 1.05)))
    beta, t, it = flow(np.zeros(d) + 0.0, c, gc, lambda b, t: not todo, rec)
    return beta, out, (bs, XR, yR, XF, yF)

# ------------------------------------------------------------------ Test 1: stage 2, forget -> logit 0 (soft label 1/2)
def test1(k, seed, apt, A, D):
    c = np.sqrt(k) / 2
    bs, XR, yR, XF, yF = D
    rows = []
    for anchor in ("none", "GDR"):
        X = XF if anchor == "none" else np.hstack([XF, XR])
        nF = XF.shape[1]
        def gc(beta):
            fF = XF.T @ beta
            g = XF @ (expit(fF) - 0.5)              # CE to soft label 1/2: d/df = sigma(f) - 1/2
            w = np.r_[expit(fF) * expit(-fF)]
            if anchor == "GDR":
                m, gr, wr = logistic_parts(XR, yR, beta)
                g = g + gr; w = np.r_[w, wr]
            return g, X, w
        marks = list(np.logspace(-1, 4, 11)) if anchor == "none" else None
        L0 = np.mean(np.logaddexp(0, -yR * (XR.T @ A)))
        Lt = [L0 * f for f in (0.5, 0.2, 0.1, 0.03, 0.01, 0.003)]
        def rec(beta, t):
            if anchor == "none":
                if marks and t >= marks[0]:
                    marks.pop(0)
                    P, v = project(A, c, A_eq=XF, b_eq=np.zeros(nF))
                    rows.append(dict(test=1, k=k, seed=seed, apt=apt, anchor=anchor, t=t, err=rel(beta, P, A),
                                     fres=np.max(np.abs(XF.T @ beta)), viol=v))
            else:
                L = np.mean(np.logaddexp(0, -yR * (XR.T @ beta)))
                if Lt and L <= Lt[0]:
                    Lt.pop(0)
                    m = yR * (XR.T @ beta); gam = m.min()
                    P, v = project(A, c, A_eq=XF, b_eq=np.zeros(nF), A_ge=XR * yR, b_ge=np.full(len(yR), gam))
                    P2, v2 = project(A, c, A_eq=np.hstack([XF, XR]), b_eq=np.r_[np.zeros(nF), XR.T @ A])   # retain logits frozen
                    P3, v3 = project(A, c, A_eq=np.hstack([XF, XR * yR]), b_eq=np.r_[np.zeros(nF), np.full(len(yR), gam)])  # all retain at margin
                    def lossf(b):
                        f = XF.T @ b; mm = yR * (XR.T @ b)
                        Lv = np.sum(np.logaddexp(0, f) - f / 2 - np.log(2)) + np.sum(np.logaddexp(0, -mm))
                        return Lv, XF @ (expit(f) - 0.5) - XR @ (yR * expit(-mm))
                    R, _ = regpath(lossf, A, c, lossf(beta)[0])
                    rows.append(dict(test=1, k=k, seed=seed, apt=apt, anchor=anchor, t=t, Lret=L, gamma=gam, err_reg=rel(beta, R, A),
                                     err=rel(beta, P, A), err_frozen=rel(beta, P2, A), err_eq=rel(beta, P3, A),
                                     cos=cos(beta - A, P - A), fres=np.max(np.abs(XF.T @ beta)), viol=v))
        stop = (lambda b, t: not marks) if anchor == "none" else (lambda b, t: not Lt)
        flow(A.copy(), c, gc, stop, rec)
    return rows

# ------------------------------------------------------------------ Test 2: NPO
def test2(k, seed, apt, A, D):
    c = np.sqrt(k) / 2
    bs, XR, yR, XF, yF = D
    mref = yF * (XF.T @ A); lpref = log_expit(mref)
    rows = []
    for bnpo in (1.0, 4.0):
        for anchor in ("none", "GDR"):
            X = XF if anchor == "none" else np.hstack([XF, XR])
            def gc(beta):
                m = yF * (XF.T @ beta)
                r = log_expit(m) - lpref
                dm = 2 * expit(bnpo * r) * expit(-m)          # dL/dm  (>0: descending lowers m)
                g = XF @ (yF * dm)
                w = 2 * (expit(m) * expit(-m) + bnpo * expit(-m) ** 2) + 1e-6
                if anchor == "GDR":
                    mr, gr, wr = logistic_parts(XR, yR, beta)
                    g = g + gr; w = np.r_[w, wr]
                return g, X, w
            targets = [0.25, 0.5, 1, 2, 3, 5]
            hist = []
            def rec(beta, t):
                m = yF * (XF.T @ beta)
                Di = lpref - log_expit(m)
                hist.append((t, Di.mean()))
                if targets and Di.mean() >= targets[0]:
                    targets.pop(0)
                    Dm = Di.mean()
                    # constraint log sigma(m_i) <= log sigma(mref_i) - Dm  <=>  m_i <= logit(sigma(mref_i) e^{-Dm})
                    p = np.exp(lpref - Dm)
                    Mi = np.log(p) - np.log1p(-p)
                    kw = dict(A_ge=-(XF * yF), b_ge=-Mi)
                    kwe = dict(A_eq=XF * yF, b_eq=Mi)
                    if anchor == "GDR":
                        mr = yR * (XR.T @ beta); gam = mr.min()
                        kw = dict(A_ge=np.hstack([-(XF * yF), XR * yR]), b_ge=np.r_[-Mi, np.full(len(yR), gam)])
                        kwe = dict(A_eq=XF * yF, b_eq=Mi, A_ge=XR * yR, b_ge=np.full(len(yR), gam))
                    P, v = project(A, c, **kw)
                    Pe, ve = project(A, c, **kwe)
                    def lossf(b, anchor=anchor):
                        mm = yF * (XF.T @ b); r = log_expit(mm) - lpref
                        Lv = np.sum(2 / bnpo * np.logaddexp(0, bnpo * r)); g = XF @ (yF * 2 * expit(bnpo * r) * expit(-mm))
                        if anchor == "GDR":
                            mr = yR * (XR.T @ b); Lv += np.sum(np.logaddexp(0, -mr)); g = g - XR @ (yR * expit(-mr))
                        return Lv, g
                    R, _ = regpath(lossf, A, c, lossf(beta)[0])
                    rows.append(dict(test=2, k=k, seed=seed, apt=apt, bnpo=bnpo, anchor=anchor, t=t, Delta=Dm,
                                     Dspread=Di.std() / Dm, err=rel(beta, P, A), err_reg=rel(beta, R, A), err_eq=rel(beta, Pe, A),
                                     cos=cos(beta - A, P - A), viol=v, frac_active=np.mean(m > Mi - 0.05 * np.abs(Mi) - 1e-3)))
            flow(A.copy(), c, gc, lambda b, t: not targets, rec, maxsteps=300_000)
            ts, Ds = np.array(hist).T
            rows.append(dict(test="2hist", k=k, seed=seed, apt=apt, bnpo=bnpo, anchor=anchor, t=ts[::50], D=Ds[::50]))
    return rows

def job(args):
    k, seed, apt = args
    t0 = time.time()
    A, r0, D = pretrain(k, seed, apt)
    rows = [dict(test=0, **r) for r in r0]
    rows += test1(k, seed, apt, A, D)
    rows += test2(k, seed, apt, A, D)
    print(f"k={k} seed={seed} apt={apt}: {time.time() - t0:.0f}s", flush=True)
    return rows

if __name__ == "__main__":
    if sys.argv[1] == "one":
        rows = job((float(sys.argv[2]), 0, 0.3))
        import pandas as pd; pd.set_option("display.width", 250)
        df = pd.DataFrame([r for r in rows if r["test"] != "2hist"])
        print(df.drop(columns=[x for x in ("t",) if x in df]).to_string())
        sys.exit()
    from multiprocessing import Pool
    jobs = [(k, s, a) for s in range(5) for k in (1e-6, 1e-2, 1.0) for a in (0.3, 0.75)]
    out = []
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            out += r
            pickle.dump(out, open("ce_tests.pkl", "wb"))
    print("DONE")
