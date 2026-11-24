"""Regulariser kept on, with feature blocks. Coordinates in blocks b (fraction q_b of d); sample groups a (n_a = al_a d samples,
inputs N(0,1/d) on the blocks in S_a, weight w_a, target coef_a * x^T beta*).
Per block: beta = prox_{lam R/A_b}((B_b beta* + sqrt(C_b) h)/A_b), chi_b = E prox'/A_b,
A_b = sum_{a: b in S_a} al_a w_a/(1+w_a V_a), B_b likewise with coef_a, C_b = sum al_a w_a^2 E_a/(1+w_a V_a)^2,
V_a = sum_{b in S_a} q_b chi_b,  E_a = sum_{b in S_a} q_b E_b (beta - coef_a beta*)^2."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd", "crit"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import sys, zlib, pickle, numpy as np
from se_wd import H, WH, prox_l1, lasso
from multiprocessing import Pool
rho = 0.2

def solve_blocks(q, groups, lam, prox=prox_l1, iters=20000, damp=0.7):
    """q: dict block->fraction. groups: list of (al, w, coef, set of blocks)."""
    amp = rho ** -0.5; B_ = list(q); Q = {b: 0.5 for b in B_}; M = {b: 0.5 for b in B_}; chi = {b: 1.0 for b in B_}
    for it in range(iters):
        V = [sum(q[b] * chi[b] for b in S) for (al, w, cf, S) in groups]
        E = [sum(q[b] * (Q[b] - 2 * cf * M[b] + cf * cf) for b in S) for (al, w, cf, S) in groups]
        err = 0
        for b in B_:
            A = sum(al * w / (1 + w * V[i]) for i, (al, w, cf, S) in enumerate(groups) if b in S)
            Bb = sum(al * w * cf / (1 + w * V[i]) for i, (al, w, cf, S) in enumerate(groups) if b in S)
            C = sum(al * w * w * E[i] / (1 + w * V[i]) ** 2 for i, (al, w, cf, S) in enumerate(groups) if b in S)
            b0, d0 = prox(np.sqrt(C) * H / A, lam / A); b1, d1 = prox((Bb * amp + np.sqrt(C) * H) / A, lam / A)
            Qn = (1 - rho) * WH @ b0**2 + rho * WH @ b1**2; Mn = rho * amp * (WH @ b1); cn = ((1 - rho) * WH @ d0 + rho * WH @ d1) / A
            err += abs(Qn - Q[b]) + abs(Mn - M[b]) + abs(cn - chi[b])
            Q[b], M[b], chi[b] = damp * Q[b] + (1 - damp) * Qn, damp * M[b] + (1 - damp) * Mn, damp * chi[b] + (1 - damp) * cn
        if err < 1e-12: break
    return Q, M, err

BL = {"R": 0.4, "Sh": 0.2, "F": 0.4}
def setting(anchor_kept):
    aR, aF = 1.5 * 0.6, 0.3 * 0.6
    g = [(aR, 1.0, 1.0, {"R", "Sh"}), (aF * 2 / 3, 1.0, 0.0, {"Sh", "F"})]
    if anchor_kept: g.append((aF / 3, 1.0, 1.0, {"Sh", "F"}))
    return g
def sim(a):
    anchor_kept, lam, seed, d = a; rng = np.random.default_rng(zlib.crc32(repr(("BLK", seed, d)).encode()))
    dR, dSh = int(0.4 * d), int(0.2 * d); idx = {"R": slice(0, dR), "Sh": slice(dR, dR + dSh), "F": slice(dR + dSh, d)}
    bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0); Xs, ts = [], []
    for (al, w, cf, S) in setting(anchor_kept):
        n = int(round(al * d)); X = np.zeros((d, n))
        for b in S: X[idx[b]] = rng.standard_normal((idx[b].stop - idx[b].start, n)) / np.sqrt(d)
        Xs.append(X); ts.append(cf * (X.T @ bs))
    X = np.hstack(Xs); t = np.concatenate(ts); b = lasso(X, t, np.ones(len(t)), lam, iters=30000)
    return (anchor_kept, lam, seed, {k: (np.mean(b[s] ** 2), np.mean(b[s] * bs[s])) for k, s in idx.items()})
if __name__ == "__main__":
    jobs = [(ak, lam, s, 1000) for ak in (False, True) for lam in (0.01, 0.05, 0.15) for s in range(10)]
    with Pool(2) as p: res = p.map(sim, jobs)
    pickle.dump(res, open("blocks.pkl", "wb")); zs = []
    for ak in (False, True):
        for lam in (0.01, 0.05, 0.15):
            Q, M, err = solve_blocks(BL, setting(ak), lam); rr = [r[3] for r in res if r[0] == ak and r[1] == lam]; line = f"kept anchored={ak!s:5} lam={lam:<5}"
            for b in ("R", "Sh", "F"):
                for j, (nm, th) in enumerate((("norm", Q[b]), ("overlap", M[b]))):
                    v = np.array([r[b][j] for r in rr]); m, e = v.mean(), v.std(ddof=1) / np.sqrt(len(v)); zs.append((m - th) / max(e, 1e-12))
                    line += f" | {b} {nm}: {th:.4f}/{m:.4f}±{e:.4f}"
            print(line)
    zs = np.array(zs); print(f"{len(zs)} numbers: |z|<2 in {100*np.mean(np.abs(zs)<2):.0f}%, max |z| = {np.abs(zs).max():.1f}")
