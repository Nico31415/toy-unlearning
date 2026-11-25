"""Squared-loss pretraining + finetuning pipeline for beta = u^2 - v^2, simulated by (stochastic) gradient descent on (u, v).

Coordinates: blocks retain / shared / forget with fractions q = (q_r, q_sh, q_f) of d. Retain-class inputs live on retain+shared, forget-class inputs on
shared+forget, entries N(0, 1/d). Labels y = x^T beta* + N(0, sig2).
Pretraining: 1/2 sum (p - y)^2 over R_full and F_full [+ lam/2 (|u|^2 + |v|^2)].
Finetuning (after u, v <- sqrt(S) u, sqrt(S) v): mu_r/2 sum_R (p - y)^2 + forget term on F (or on fresh forget-class samples), for the losses in LOSSES.
Step: u <- u (1 - 2 gam G - gam lam), v <- v (1 + 2 gam G - gam lam), G = d loss / d beta. SGD: each sample in the batch with probability phi, weight 1/phi.
Metrics (MSE): m1 fresh retain-class inputs vs clean teacher; m2 on F vs its pretraining labels; m3 on F_full minus F; m4 fresh forget-class inputs vs clean teacher.
References: the pretraining recipe on R_full + (F_full minus F) ("untrain") and on R_full only ("unlearn")."""
import numpy as np, hashlib, pickle, os

LOSSES = ["retain only", "forget to zero", "forget to noise", "negated target -0.5", "negated target -1", "signed", "pure ascent"]
#          (retain weight on?, sign of forget term, target multiple of y, noise target?)
SPEC = {"retain only": (1, 0, 0, 0), "forget to zero": (1, 1, 0, 0), "forget to noise": (1, 1, 0, 1), "negated target -0.5": (1, 1, -0.5, 0),
        "negated target -1": (1, 1, -1, 0), "signed": (1, -1, 1, 0), "pure ascent": (0, -1, 1, 0)}
BASE = dict(q=(1, 0.4, 1), alpha=0.3, ffrac=0.5, rfrac=1.0, sig2_pt=0.0, sig2_ft=0.0, lam_pt=0.0, lam_ft=0.0, phi_pt=None, phi_ft=None, S=1.0, mu=(1.0, 1.0),
            mu_signed=None, fresh=False, gam=0.01, t1=40.0, t2=40.0, k=1e-5, rho=0.2, d=3000, rec=0.1)

def make(cfg, seed):
    rng = np.random.default_rng(seed); d = cfg["d"]; q = np.array(cfg["q"], float); q /= q.sum(); e = np.r_[0, np.round(np.cumsum(q) * d).astype(int)]
    bs = np.zeros(d)
    for B in range(3):
        nb = e[B + 1] - e[B]; m = int(round(cfg["rho"] * nb)); idx = e[B] + rng.permutation(nb)[:m]; bs[idx] = rng.choice([-1, 1], m) * cfg["rho"]**-0.5
    rowsR, rowsF = slice(e[0], e[2]), slice(e[1], e[3]); nR = int(round(cfg["alpha"] * (e[2] - e[0]))); nFf = int(round(cfg["alpha"] * (e[3] - e[1])))
    XR = rng.standard_normal((e[2] - e[0], nR)) / np.sqrt(d); XF = rng.standard_normal((e[3] - e[1], nFf)) / np.sqrt(d)
    yR = XR.T @ bs[rowsR] + np.sqrt(cfg["sig2_pt"]) * rng.standard_normal(nR); yF = XF.T @ bs[rowsF] + np.sqrt(cfg["sig2_pt"]) * rng.standard_normal(nFf)
    return dict(e=e, bs=bs, rowsR=rowsR, rowsF=rowsF, XR=XR, XF=XF, yR=yR, yF=yF, d=d)

def metrics(D, b, nF):
    """b: (d, L). returns dict of arrays (L,)"""
    e, bs = D["e"], D["bs"][:, None]; err = (b - bs)**2; blk = [err[e[B]:e[B + 1]].sum(0) / D["d"] for B in range(3)]
    pF = D["XF"].T @ b[D["rowsF"]]; rF = (pF - D["yF"][:, None])**2; pR = D["XR"].T @ b[D["rowsR"]]
    return dict(m1=blk[0] + blk[1], m4=blk[1] + blk[2], m2=rF[:nF].mean(0) if nF else np.full(b.shape[1], np.nan), m3=rF[nF:].mean(0) if nF < rF.shape[0] else np.full(b.shape[1], np.nan),
                trainR=((pR - D["yR"][:, None])**2).mean(0), eR=blk[0], eSh=blk[1], eF=blk[2], Q=(b**2).sum(0) / D["d"])

def descend(D, u, v, grad, steps, gam, lam, rec_every, nF, refs=None, cap=None):
    """generic loop; grad(b, step) -> G (d, L). records metrics every rec_every steps. columns that exceed the norm cap are frozen."""
    out = {}; alive = np.ones(u.shape[1], bool); nrec = steps // rec_every + 1
    for n in range(steps + 1):
        b = u * u - v * v
        if n % rec_every == 0:
            m = metrics(D, b, nF)
            if refs is not None:
                for name, r in refs.items(): m["d_" + name] = ((b - r[:, None])**2).sum(0) / D["d"]
            for key, val in m.items(): out.setdefault(key, np.full((nrec, u.shape[1]), np.nan))[n // rec_every] = np.where(alive, val, np.nan)
        if n == steps: break
        if cap is not None: alive &= (b**2).sum(0) < cap
        G = grad(b, n) * alive
        u, v = u * (1 - 2 * gam * G - gam * lam * alive), v * (1 + 2 * gam * G - gam * lam * alive)
    return u, v, out

def pretrain(D, cfg, seed, useF, tag):
    """useF: boolean mask over F_full samples used (R_full always used)."""
    d = D["d"]; c = np.sqrt(cfg["k"]) / 2; u = np.full((d, 1), np.sqrt(c / 2)); v = u.copy(); mr = np.random.default_rng(seed * 1000 + tag); phi = cfg["phi_pt"]
    XF = D["XF"][:, useF]; yF = D["yF"][useF]
    def grad(b, n):
        gR = D["XR"].T @ b[D["rowsR"]] - D["yR"][:, None]; gF = XF.T @ b[D["rowsF"]] - yF[:, None]
        if phi is not None: gR = gR * ((mr.random(len(gR)) < phi) / phi)[:, None]; gF = gF * ((mr.random(len(gF)) < phi) / phi)[:, None]
        G = np.zeros_like(b); G[D["rowsR"]] += D["XR"] @ gR; G[D["rowsF"]] += XF @ gF; return G
    return grad, u, v

def pipeline(cfg, seed, cache=None):
    cfg = {**BASE, **cfg}; D = make(cfg, seed); d = D["d"]; gam = cfg["gam"]; rec = max(1, int(round(cfg["rec"] / gam))); s1 = int(round(cfg["t1"] / gam)); s2 = int(round(cfg["t2"] / gam))
    nFf = D["XF"].shape[1]; nF = int(round(cfg["ffrac"] * nFf)); nR = D["XR"].shape[1]; nRu = int(round(cfg["rfrac"] * nR)); res = dict(cfg=cfg, seed=seed)
    key = lambda name, extra=(): hashlib.md5(repr((name, seed, extra, [cfg[x] for x in ("q", "alpha", "sig2_pt", "lam_pt", "phi_pt", "gam", "t1", "k", "rho", "d", "rec", "ffrac")])).encode()).hexdigest()
    def cached(name, extra, fn):
        f = None if cache is None else os.path.join(cache, key(name, extra) + ".pkl")
        if f and os.path.exists(f): return pickle.load(open(f, "rb"))
        val = fn()
        if f: pickle.dump(val, open(f, "wb"))
        return val
    def stage1(useF, tag):
        grad, u, v = pretrain(D, cfg, seed, useF, tag); u, v, out = descend(D, u, v, grad, s1, gam, cfg["lam_pt"], rec, nF); return u, v, out
    allF = np.ones(nFf, bool); keptF = allF.copy(); keptF[:nF] = False
    u, v, res["pre"] = cached("pre", (), lambda: stage1(allF, 0))
    uU, vU, res["ref_untrain"] = cached("untrain", (cfg["ffrac"],), lambda: stage1(keptF, 1))
    uL, vL, res["ref_unlearn"] = cached("unlearn", (), lambda: stage1(np.zeros(nFf, bool), 2))
    refs = dict(untrain=(uU * uU - vU * vU)[:, 0], unlearn=(uL * uL - vL * vL)[:, 0])
    # re-evaluate m2/m3 of the pretrained and reference end points for this split (cache is split-independent for 'pre' and 'unlearn')
    for nm, (a, b_) in dict(pre=(u, v), ref_untrain=(uU, vU), ref_unlearn=(uL, vL)).items(): res[nm + "_end"] = {k_: x[0] for k_, x in metrics(D, a * a - b_ * b_, nF).items()}
    for nm in ("pre", "ref_unlearn"):  # trajectories of m2, m3 depend on the split: recompute cheaply is not possible without betas, so keep whole-F_full value under mF
        pass
    # ---- finetuning
    rng = np.random.default_rng(seed + 555); L = len(LOSSES); rS = np.sqrt(cfg["S"]); U = np.repeat(u * rS, L, 1); V = np.repeat(v * rS, L, 1)
    if cfg["fresh"]:
        XS = rng.standard_normal((D["XF"].shape[0], nF)) / np.sqrt(d); yS = XS.T @ D["bs"][D["rowsF"]] + np.sqrt(cfg["sig2_ft"]) * rng.standard_normal(nF)
    else: XS = D["XF"][:, :nF]; yS = D["yF"][:nF]
    XRu = D["XR"][:, :nRu]; yRu = D["yR"][:nRu]; xi = rng.standard_normal(nF); mr = np.random.default_rng(seed + 777); phi = cfg["phi_ft"]
    wr = np.array([SPEC[l][0] for l in LOSSES], float); sg = np.array([SPEC[l][1] for l in LOSSES], float); bt = np.array([SPEC[l][2] for l in LOSSES], float); nz = np.array([SPEC[l][3] for l in LOSSES], float)
    mur = np.full(L, cfg["mu"][0]); muf = np.full(L, cfg["mu"][1]); i = LOSSES.index("signed")
    if cfg["mu_signed"] is not None: mur[i], muf[i] = cfg["mu_signed"]
    tS = yS[:, None] * bt[None, :] + xi[:, None] * nz[None, :]
    def grad(b, n):
        gR = (XRu.T @ b[D["rowsR"]] - yRu[:, None]) * (wr * mur); gS = (XS.T @ b[D["rowsF"]] - tS) * (sg * muf)
        if phi is not None: gR = gR * ((mr.random(len(gR)) < phi) / phi)[:, None]; gS = gS * ((mr.random(len(gS)) < phi) / phi)[:, None]
        G = np.zeros_like(b); G[D["rowsR"]] += XRu @ gR; G[D["rowsF"]] += XS @ gS; return G
    cap = 400.0 * (D["bs"]**2).sum()
    _, _, res["ft"] = descend(D, U, V, grad, s2, gam, cfg["lam_ft"], rec, nF, refs=refs, cap=cap)
    res["d_pre"] = {nm: float(((u * u - v * v)[:, 0] * cfg["S"] - r) @ ((u * u - v * v)[:, 0] * cfg["S"] - r) / d) for nm, r in refs.items()}
    return res

if __name__ == "__main__":
    import time, sys
    t = time.time(); r = pipeline(dict(d=int(sys.argv[1]), gam=float(sys.argv[2]), t1=float(sys.argv[3]), t2=float(sys.argv[4])), 0); print("seconds", time.time() - t)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    tt = np.arange(len(r["pre"]["m1"])) * BASE["rec"]
    for i in [0, 10, 20, 50, 100, 200, 400, 800]:
        if i < len(tt): print(f"pre t={tt[i]:5.1f} m1 {r['pre']['m1'][i,0]:.5f} m4 {r['pre']['m4'][i,0]:.5f} trainR {r['pre']['trainR'][i,0]:.2e} Q {r['pre']['Q'][i,0]:.4f}")
    print("ends: pre", {k: round(float(x), 4) for k, x in r["pre_end"].items()}); print("untrain", {k: round(float(x), 4) for k, x in r["ref_untrain_end"].items()}); print("unlearn", {k: round(float(x), 4) for k, x in r["ref_unlearn_end"].items()})
    for key in ("m1", "m2", "m3", "m4", "d_untrain"):
        print(key);
        for i in [0, 5, 10, 20, 50, 100, 200, 400, 800]:
            if i < len(r["ft"][key]): print(f"  t={i*BASE['rec']:5.1f}", r["ft"][key][i])
