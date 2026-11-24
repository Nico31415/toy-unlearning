"""Trajectory equations with blocks of coordinates and groups of samples (two classes), squared loss, two stages.
Sample of group a:   p(t) = zeta(t) + sum_{m<t} [sum_{B in a} q_B R_B(t,m)] g(m),   g = w_a(t) (p - b_a(t) y)
Coordinate of block B: theta' = -4 [ eta + sum_{a touching B} alpha_a ( sum_m E_a[dg(t)/dzeta(m)] beta(m) + E_a[dg(t)/dy] beta* ) ]
A reference model is the same run with other sample weights; it shares the randomness (same seed)."""
import numpy as np, sys, pickle
from multiprocessing import Pool

def layout(blocks, groups, N, rho_strat, rng):
    eb = np.r_[0, np.cumsum([int(round(q * N)) for q, _ in blocks])]; bs = np.zeros(eb[-1])
    for B, (q, rho) in enumerate(blocks):
        m = int(round(rho * (eb[B + 1] - eb[B]) / 2)); sg = np.r_[np.ones(m), -np.ones(m)] if rho_strat else rng.choice([-1, 1], 2 * m); bs[eb[B]:eb[B] + 2 * m] = sg * rho**-0.5
    eg = np.r_[0, np.cumsum([int(round(al * N)) for al, _ in groups])]; mask = np.zeros((eg[-1], eb[-1]), bool)
    for a, (al, sup) in enumerate(groups):
        for B in sup: mask[eg[a]:eg[a + 1], eb[B]:eb[B + 1]] = True
    return eb, eg, bs, mask

def run(blocks, groups, sched, n1, T, dt, k, N=3000, seed=0, refs=(), kappa=1.0):
    """sched(stage) -> (w, b): per-group sample weights and target multiples in stage 0 (pretraining) and stage 1. Returns metrics, or beta at the last step if T == n1 + 1 and refs is None."""
    rng = np.random.default_rng(seed); eb, eg, bs, mask = layout(blocks, groups, N, True, rng); Nc, Ns = eb[-1], eg[-1]; nB, nG = len(blocks), len(groups); c = np.sqrt(k) / 2
    G1 = rng.standard_normal((Ns, Nc)) / np.sqrt(Nc) * mask; G2 = rng.standard_normal((Nc, Ns)) / np.sqrt(Nc) * mask.T; z = G1 @ bs
    q = np.array([(eb[B + 1] - eb[B]) / Nc for B in range(nB)]); al = np.array([(eg[a + 1] - eg[a]) / Nc for a in range(nG)]); va = np.array([sum(q[B] for B in groups[a][1]) for a in range(nG)])
    th = np.zeros(Nc); beta = np.zeros((Nc, T)); D = np.zeros((Nc, T)); J = np.zeros((Nc, T, T)); Jn = np.zeros((Nc, T)); g = np.zeros((Ns, T)); Tg = np.zeros((Ns, T, T))
    RbB = np.zeros((nB, T, T)); MB = np.zeros((nB, T)); out = {key: np.zeros(T) for key in ["Q"] + [f"d{i}" for i in range(len(refs or ()))]}
    for n in range(T):
        cc = c * (kappa if n >= n1 else 1.0); b = cc * np.sinh(th); beta[:, n] = b; D[:, n] = np.sqrt(b * b + cc * cc); J[:, n, :] = Jn
        if refs is None and n == T - 1: return b
        out["Q"][n] = b @ b / Nc
        for i, r in enumerate(refs or ()): out[f"d{i}"][n] = np.sum((b - r)**2) / Nc
        for B in range(nB): sl = slice(eb[B], eb[B + 1]); RbB[B, n, :n] = (D[sl, n] @ Jn[sl, :n]) / (eb[B + 1] - eb[B]); MB[B, n] = b[sl] @ bs[sl] / (eb[B + 1] - eb[B])
        w, bt = sched(int(n >= n1)); Rg = np.zeros((nG, n + 1)); dz = np.zeros(nG)
        for a in range(nG):
            sl = slice(eg[a], eg[a + 1]); Rb = sum(q[B] * RbB[B, n, :n] for B in groups[a][1]); p = G1[sl] @ b + g[sl, :n] @ Rb
            g[sl, n] = w[a] * (p - bt[a] * z[sl]); tp = np.einsum("l,ilm->im", Rb, Tg[sl, :n, :]); tp[:, n] += 1.0; Tg[sl, n, :] = w[a] * tp
            Rg[a] = Tg[sl, n, :n + 1].mean(0); Ma = sum(q[B] * MB[B, :n + 1] for B in groups[a][1]); dz[a] = (np.mean(z[sl] * g[sl, n]) - Rg[a] @ Ma) / va[a]
        Gf = G2 @ g[:, n]; dG = np.zeros((Nc, T))
        for B in range(nB):
            sl = slice(eb[B], eb[B + 1]); ga = [a for a in range(nG) if B in groups[a][1]]; R = sum(al[a] * Rg[a] for a in ga) + np.zeros(n + 1)
            Gf[sl] += beta[sl, :n + 1] @ R + sum(al[a] * dz[a] for a in ga) * bs[sl]; dG[sl] = np.einsum("l,il,ilm->im", R, D[sl, :n + 1], J[sl, :n + 1, :])
        dG[:, n] += 1.0; th = th - 4 * dt * Gf; Jn = Jn - 4 * dt * dG
    return out

def simulate(blocks, groups, sched, n1, T, dt, k, d=3000, seed=0, refs=(), kappa=1.0):
    rng = np.random.default_rng(seed + 10**6); eb, eg, bs, mask = layout(blocks, groups, d, False, rng); c = np.sqrt(k) / 2; X = (rng.standard_normal(mask.shape) / np.sqrt(eb[-1]) * mask).T; y = X.T @ bs
    th = np.zeros(eb[-1]); out = {key: np.zeros(T) for key in ["Q"] + [f"d{i}" for i in range(len(refs or ()))]}; grp = np.repeat(np.arange(len(groups)), np.diff(eg))
    for n in range(T):
        cc = c * (kappa if n >= n1 else 1.0); b = cc * np.sinh(th)
        if refs is None and n == T - 1: return b
        out["Q"][n] = b @ b / eb[-1]
        for i, r in enumerate(refs or ()): out[f"d{i}"][n] = np.sum((b - r)**2) / eb[-1]
        w, bt = sched(int(n >= n1)); th = th - 4 * dt * (X @ (np.array(w)[grp] * (X.T @ b - np.array(bt)[grp] * y)))
    return out

# ----- the two-class setting: blocks retain / shared / forget; groups retain class, forget samples, kept samples of the forget class
BL = [(0.4, 0.2), (0.2, 0.2), (0.4, 0.2)]; GR = [(0.3, [0, 1]), (0.12, [1, 2]), (0.08, [1, 2])]
STAGE2 = {"zero targets": ([0, 1, 0], [1, 0, 1]), "zero targets, retain anchor": ([1, 1, 0], [1, 0, 1]), "zero targets, anchor on all kept": ([1, 1, 1], [1, 0, 1]), "targets -0.5 y, anchor on all kept": ([1, 1, 1], [1, -0.5, 1])}
def job(a):
    kind, name, k, n1, T, dt, seed = a; fn = run if kind == "th" else simulate; size = dict(N=8000) if kind == "th" else dict(d=3000)
    pre = ([1, 1, 1], [1, 1, 1]); mk = lambda w1: (lambda st: (w1, [1, 1, 1]))
    rK = fn(BL, GR, mk([1, 0, 1]), n1, n1 + 1, dt, k, seed=seed, refs=None, **size); rR = fn(BL, GR, mk([1, 0, 0]), n1, n1 + 1, dt, k, seed=seed, refs=None, **size)
    return fn(BL, GR, lambda st: STAGE2[name] if st else pre, n1, T, dt, k, seed=seed, refs=(rK, rR), **size)
if __name__ == "__main__":
    k, T1, T2, nrun = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]); dt = 0.05; n1 = int(round(T1 / dt)); T = n1 + int(round(T2 / dt)); res = {}
    print(f"k={k:g}: two classes, squared loss. d0 = distance to the model retrained without the forget samples, d1 = to the model trained on the retain class only (both trained for T1). theory / sim (z)")
    for name in STAGE2:
        with Pool(2) as p: R = p.map(job, [(w, name, k, n1, T, dt, s) for w in ("th", "si") for s in range(nrun)])
        res[name] = (R[:nrun], R[nrun:]); row = f"{name:36s}|"
        for i in [n1] + [n1 + int(round(x * (T - 1 - n1))) for x in (0.1, 0.25, 0.5, 1.0)]:
            for key in ("d0", "d1"):
                a, b = np.array([o[key][i] for o in R[:nrun]]), np.array([o[key][i] for o in R[nrun:]]); row += f" {a.mean():.3f}/{b.mean():.3f} ({(a.mean()-b.mean())/np.hypot(a.std(ddof=1), b.std(ddof=1))*np.sqrt(nrun):+.1f})"
            row += " |"
        print(row, flush=True)
    pickle.dump(dict(res=res, dt=dt, n1=n1, k=k), open(f"blocks_{k:g}.pkl", "wb"))
