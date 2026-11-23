"""Soft-label CE toy. Pretraining: CE with label smoothing q -> finite targets tau*y, tau=logit(q).
Stage 2 ascent on hard labels. Test: flow vs one-parameter family beta(s) = c sinh(theta0 - s g), g = sum_F y_i x_i."""
import sys, time, zlib, warnings
import numpy as np
from scipy.special import expit, log_expit
from scipy.optimize import brentq
from ce_tests import project, flow, rel, cos, _LXX
warnings.simplefilter("ignore")
RHO, Q = 0.2, 0.9
TAU = np.log(Q / (1 - Q))

def data(seed, apt, blocks, phi=1.0):
    dR, dSh, dF = blocks; d = sum(blocks)
    rng = np.random.default_rng(zlib.crc32(repr(("S", seed, apt, blocks)).encode()))
    amp = RHO ** -0.5
    bs = np.where(rng.random(d) < RHO, rng.choice([-amp, amp], d), 0.0)
    nR, nF = int(1.5 * (dR + dSh)) if dR else 0, int(round(apt * (dSh + dF)))
    XR = np.zeros((d, nR)); XR[:dR + dSh] = rng.standard_normal((dR + dSh, nR)) / np.sqrt(d)
    XF = np.zeros((d, nF)); XF[dR:] = rng.standard_normal((dSh + dF, nF)) / np.sqrt(d)
    yR, yF = np.sign(XR.T @ bs), np.sign(XF.T @ bs)
    nf = int(round(phi * nF))
    return d, bs, XR, yR, XF[:, :nf], yF[:nf], XF[:, nf:], yF[nf:]

def ascent(kind, k, seed, apt, blocks, anchor=False, bnpo=1.0, phi=1.0, drops=(0.5, 1, 2, 4, 8), verbose=True):
    c = np.sqrt(k) / 2
    d, bs, XR, yR, XF, yF, XK, yK = data(seed, apt, blocks, phi)
    Xall, yall = np.hstack([XR, XF, XK]), np.r_[yR, yF, yK]
    A, v = project(np.zeros(d), c, A_eq=Xall, b_eq=TAU * yall)
    assert v < 1e-6, v
    th0 = np.arcsinh(A / c); g = XF @ yF
    Xanc, yanc = np.hstack([XR, XK]), np.r_[yR, yK]
    mref = TAU; lpref = log_expit(TAU)
    X = np.hstack([XF, Xanc]) if anchor else XF
    def gc(beta):
        m = yF * (XF.T @ beta)
        if kind == "GA":
            dm = expit(-m); w = expit(m) * expit(-m)
        else:
            r = log_expit(m) - lpref
            dm = 2 * expit(bnpo * r) * expit(-m); w = 2 * (expit(m) * expit(-m) + bnpo * expit(-m) ** 2)
        grad = XF @ (yF * dm)                      # descending this lowers the margins
        if anchor:                                 # soft-label CE on the anchor samples
            ma = yanc * (Xanc.T @ beta)
            grad = grad + Xanc @ (yanc * (expit(ma) - Q)); w = np.r_[w, expit(ma) * expit(-ma)]
        return grad, X, w + 1e-9
    todo = list(drops); rows = []
    def pred(s):
        b0 = c * np.sinh(th0 - s * g)
        if anchor:
            return project(b0, c, A_eq=Xanc, b_eq=TAU * yanc)[0]
        return b0
    def rec(beta, t):
        m = yF * (XF.T @ beta)
        if todo and TAU - m.mean() >= todo[0]:
            dr = todo.pop(0)
            f = lambda s: np.mean(yF * (XF.T @ pred(s))) - m.mean()
            hi = 1e-3
            while f(hi) > 0: hi *= 2
            s = brentq(f, 0, hi, xtol=1e-12)
            P = pred(s)
            mp = yF * (XF.T @ P)
            row = dict(drop=dr, t=t, s=s, err=rel(beta, P, A), cos=cos(beta - A, P - A), m_sd=m.std(), mP_sd=mp.std())
            if anchor:
                row["anc_res"] = np.max(np.abs(Xanc.T @ beta - TAU * yanc))
            rows.append(row)
    _LXX.clear()
    flow(A.copy(), c, gc, lambda b, t: not todo, rec, maxsteps=300_000)
    return rows

if __name__ == "__main__":
    import pandas as pd; pd.set_option("display.width", 200)
    for blocks in ((0, 0, 200), (80, 40, 80)):
        for k in (1e-6, 1e-2, 1.0):
            for kind, anchor in (("GA", False), ("NPO", False), ("GA", True), ("NPO", True)):
                if anchor and blocks[0] == 0: continue
                t0 = time.time()
                r = ascent(kind, k, 0, 0.3, blocks, anchor)
                print(f"\nblocks={blocks} k={k} {kind} anchor={anchor}  ({time.time()-t0:.0f}s)")
                print(pd.DataFrame(r).round(4).to_string(index=False))
