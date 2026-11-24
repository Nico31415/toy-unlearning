"""Trajectory equations with blocks of coordinates and groups of samples: any loss per group and stage, gradient flow or mini-batch SGD.
Loss of a group in a stage: (kind, weight, target multiple)
  'sq'   weight/2 (p - b y)^2 with y = x^T beta*          'ce'   cross-entropy with the class label sign(x^T beta*)
  'ga'   ascent (-cross-entropy)   'flip' cross-entropy with the flipped label   'npo'  2 log(1 + pi_t / pi_T1)   'off'  not used
sgd = None: Euler steps of the flow in theta with step dt.   sgd = (gam, phi): u <- u(1 - 2 gam G), v <- v(1 + 2 gam G), each sample in the batch with probability phi."""
import numpy as np, sys, pickle
from multiprocessing import Pool
from dmft_blocks import layout
from dmft_nl2 import grad

def sample_grad(kind, w, bt, p, z, y, m0):
    """returns dL/dp, d2L/dp2, d(dL/dp)/d(p at T1)"""
    if kind == "off" or w == 0: return 0 * p, 0 * p, 0 * p
    if kind == "sq": return w * (p - bt * z), w + 0 * p, 0 * p
    g1, g2, g3 = grad(kind, y * p, m0 if m0 is not None else y * p); return w * y * g1, w * g2, w * g3

def run(blocks, groups, sched, n1, T, dt, k, N=3000, seed=0, refs=(), sgd=None, forget=1):
    rng = np.random.default_rng(seed); mrng = np.random.default_rng(seed + 7); eb, eg, bs, mask = layout(blocks, groups, N, True, rng); Nc, Ns = eb[-1], eg[-1]; nB, nG = len(blocks), len(groups); c = np.sqrt(k) / 2
    G1 = rng.standard_normal((Ns, Nc)) / np.sqrt(Nc) * mask; G2 = rng.standard_normal((Nc, Ns)) / np.sqrt(Nc) * mask.T; z = G1 @ bs; y = np.sign(z)
    q = np.array([(eb[B + 1] - eb[B]) / Nc for B in range(nB)]); al = np.array([(eg[a + 1] - eg[a]) / Nc for a in range(nG)]); va = np.array([sum(q[B] for B in groups[a][1]) for a in range(nG)])
    th = np.zeros(Nc); Jn = np.zeros((Nc, T)); u = np.full(Nc, np.sqrt(c / 2)); v = u.copy(); du = np.zeros((Nc, T)); dv = np.zeros((Nc, T))
    beta = np.zeros((Nc, T)); JB = np.zeros((Nc, T, T)); g = np.zeros((Ns, T)); Tg = np.zeros((Ns, T, T)); RbB = np.zeros((nB, T, T)); MB = np.zeros((nB, T)); m0 = None; tp0 = [None] * nG
    keys = ["Q", "accF"] + [f"{s}{i}" for i in range(len(refs or ())) for s in ("d", "cos")]; out = {key: np.zeros(T) for key in keys}; slF = slice(eg[forget], eg[forget + 1])
    for n in range(T):
        if sgd is None: b = c * np.sinh(th); jb = np.sqrt(b * b + c * c)[:, None] * Jn
        else: b = u * u - v * v; jb = 2 * (u[:, None] * du - v[:, None] * dv)
        beta[:, n] = b; JB[:, n, :] = jb
        if refs is None and n == T - 1: return b
        out["Q"][n] = b @ b / Nc
        for i, r in enumerate(refs or ()): out[f"d{i}"][n] = np.sum((b - r)**2) / Nc; out[f"cos{i}"][n] = b @ r / np.sqrt((b @ b) * (r @ r) + 1e-300)
        for B in range(nB): sl = slice(eb[B], eb[B + 1]); RbB[B, n, :n] = jb[sl, :n].mean(0); MB[B, n] = b[sl] @ bs[sl] / (eb[B + 1] - eb[B])
        cfg = sched(int(n >= n1)); Rg = np.zeros((nG, n + 1)); dz = np.zeros(nG); wm = (mrng.random(Ns) < sgd[1]) / sgd[1] if sgd is not None else np.ones(Ns)
        if n == n1: m0 = np.zeros(Ns)
        for a in range(nG):
            sl = slice(eg[a], eg[a + 1]); Rb = sum(q[B] * RbB[B, n, :n] for B in groups[a][1]); p = G1[sl] @ b + g[sl, :n] @ Rb
            tp = np.einsum("l,ilm->im", Rb, Tg[sl, :n, :]); tp[:, n] += 1.0
            if n == n1: m0[sl] = y[sl] * p; tp0[a] = tp.copy()
            if a == forget: out["accF"][n] = np.mean(y[sl] * p > 0)
            kind, w, bt = cfg[a]; g1, g2, g3 = sample_grad(kind, w, bt, p, z[sl], y[sl], m0[sl] if m0 is not None else None)
            g[sl, n] = wm[sl] * g1; Tg[sl, n, :] = (wm[sl] * g2)[:, None] * tp + ((wm[sl] * g3)[:, None] * tp0[a] if (kind == "npo" and n >= n1) else 0.0)
            Rg[a] = Tg[sl, n, :n + 1].mean(0); Ma = sum(q[B] * MB[B, :n + 1] for B in groups[a][1]); dz[a] = (np.mean(z[sl] * g[sl, n]) - Rg[a] @ Ma) / va[a]
        Gf = G2 @ g[:, n]; dG = np.zeros((Nc, T))
        for B in range(nB):
            sl = slice(eb[B], eb[B + 1]); ga = [a for a in range(nG) if B in groups[a][1]]; R = sum(al[a] * Rg[a] for a in ga) + np.zeros(n + 1)
            Gf[sl] += beta[sl, :n + 1] @ R + sum(al[a] * dz[a] for a in ga) * bs[sl]; dG[sl] = np.einsum("l,ilm->im", R, JB[sl, :n + 1, :])
        dG[:, n] += 1.0
        if sgd is None: th = th - 4 * dt * Gf; Jn = Jn - 4 * dt * dG
        else:
            gam = sgd[0]; du = du * (1 - 2 * gam * Gf)[:, None] - 2 * gam * u[:, None] * dG; dv = dv * (1 + 2 * gam * Gf)[:, None] + 2 * gam * v[:, None] * dG; u = u * (1 - 2 * gam * Gf); v = v * (1 + 2 * gam * Gf)
    return out

def simulate(blocks, groups, sched, n1, T, dt, k, d=3000, seed=0, refs=(), sgd=None, forget=1):
    rng = np.random.default_rng(seed + 10**6); mrng = np.random.default_rng(seed + 7 + 10**6); eb, eg, bs, mask = layout(blocks, groups, d, False, rng); c = np.sqrt(k) / 2; nG = len(groups)
    X = (rng.standard_normal(mask.shape) / np.sqrt(eb[-1]) * mask).T; z = X.T @ bs; y = np.sign(z); th = np.zeros(eb[-1]); u = np.full(eb[-1], np.sqrt(c / 2)); v = u.copy(); m0 = None
    keys = ["Q", "accF"] + [f"{s}{i}" for i in range(len(refs or ())) for s in ("d", "cos")]; out = {key: np.zeros(T) for key in keys}; slF = slice(eg[forget], eg[forget + 1])
    for n in range(T):
        b = c * np.sinh(th) if sgd is None else u * u - v * v
        if refs is None and n == T - 1: return b
        p = X.T @ b; out["Q"][n] = b @ b / eb[-1]; out["accF"][n] = np.mean(y[slF] * p[slF] > 0)
        for i, r in enumerate(refs or ()): out[f"d{i}"][n] = np.sum((b - r)**2) / eb[-1]; out[f"cos{i}"][n] = b @ r / np.sqrt((b @ b) * (r @ r) + 1e-300)
        if n == n1: m0 = y * p
        cfg = sched(int(n >= n1)); gg = np.zeros(len(p)); wm = (mrng.random(len(p)) < sgd[1]) / sgd[1] if sgd is not None else 1.0
        for a in range(nG): sl = slice(eg[a], eg[a + 1]); gg[sl] = sample_grad(cfg[a][0], cfg[a][1], cfg[a][2], p[sl], z[sl], y[sl], m0[sl] if m0 is not None else None)[0]
        G = X @ (wm * gg)
        if sgd is None: th = th - 4 * dt * G
        else: u, v = u * (1 - 2 * sgd[0] * G), v * (1 + 2 * sgd[0] * G)
    return out

# ----- two classes: blocks retain / shared / forget; groups: retain class, forget samples, kept samples of the forget class
BL = [(0.4, 0.2), (0.2, 0.2), (0.4, 0.2)]; GR = [(0.6, [0, 1]), (0.3, [1, 2]), (0.3, [1, 2])]
import os
if os.environ.get('GRSMALL'): GR = [(0.3, [0, 1]), (0.12, [1, 2]), (0.08, [1, 2])]          # the sample ratios of the squared-loss checks
EXP = {  # name: (pretraining loss kind, stage-2 config per group)
    "labels: NPO + retain term on all kept": ("ce", [("ce", 1, 1), ("npo", 1, 1), ("ce", 1, 1)]),
    "labels: NPO + retain term on retain class": ("ce", [("ce", 1, 1), ("npo", 1, 1), ("off", 0, 1)]),
    "labels: flipped labels, no retain term": ("ce", [("off", 0, 1), ("flip", 1, 1), ("off", 0, 1)]),
    "labels: ascent + retain term on all kept": ("ce", [("ce", 1, 1), ("ga", 1, 1), ("ce", 1, 1)]),
    "squared loss: zero targets, anchor on all kept": ("sq", [("sq", 1, 1), ("sq", 1, 0), ("sq", 1, 1)]),
    "squared loss: zero targets, retain anchor": ("sq", [("sq", 1, 1), ("sq", 1, 0), ("off", 0, 1)]),
}
def job(a):
    kind, name, k, n1, T, dt, seed, sgd, size = a; fn = run if kind == "th" else simulate; sz = dict(N=size) if kind == "th" else dict(d=3000); pk, s2 = EXP[name]
    mk = lambda use: (lambda st: [(pk if use[i] else "off", use[i], 1) for i in range(3)])
    rK = fn(BL, GR, mk([1, 0, 1]), n1, n1 + 1, dt, k, seed=seed, refs=None, sgd=sgd, **sz); rR = fn(BL, GR, mk([1, 0, 0]), n1, n1 + 1, dt, k, seed=seed, refs=None, sgd=sgd, **sz)
    return fn(BL, GR, lambda st: s2 if st else [(pk, 1, 1)] * 3, n1, T, dt, k, seed=seed, refs=(rK, rR), sgd=sgd, **sz)
if __name__ == "__main__":
    mode, k, n1, n2, nrun, size = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]); names = sys.argv[7].split(","); dt = 0.05; T = n1 + n2
    sgd = (float(sys.argv[8]), float(sys.argv[9])) if mode == "sgd" else None; res = {}
    print(f"{mode}, k={k:g}" + (f", step {sgd[0]}, batch fraction {sgd[1]}" if sgd else "") + ": cos0/d0 = to the model retrained without the forget samples, cos1 = to the model trained on the retain class only. theory / sim (z)")
    for name in [nm for nm in EXP if any(nm.startswith(x) or x in nm for x in names)]:
        with Pool(2) as p: R = p.map(job, [(w, name, k, n1, T, dt, s, sgd, size) for w in ("th", "si") for s in range(nrun)])
        res[name] = (R[:nrun], R[nrun:]); print(name)
        for i in [n1] + [n1 + int(round(x * (n2 - 1))) for x in (0.25, 0.5, 1.0)]:
            row = f"   step {i - n1:4d} of stage 2 |"
            for key in ("cos0", "cos1", "d0", "accF"):
                a, b = np.array([o[key][i] for o in R[:nrun]]), np.array([o[key][i] for o in R[nrun:]]); row += f" {key} {a.mean():.3f}/{b.mean():.3f} ({(a.mean()-b.mean())/max(np.hypot(a.std(ddof=1), b.std(ddof=1)), 1e-12)*np.sqrt(nrun):+.1f}) |"
            print(row, flush=True)
    pickle.dump(dict(res=res, n1=n1, k=k, sgd=sgd), open(f"blocks2_{mode}_{k:g}" + ("_small" if os.environ.get("GRSMALL") else "") + f"_{size}.pkl", "wb"))
