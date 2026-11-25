"""Large-d theory for the squared-loss pipeline of mse_sim.py.

Pretraining run to convergence is one static cavity system (slot 0): interpolation of the pretraining labels from beta = 0 at scale c.
Finetuning is the trajectory (DMFT) forward pass started from that slot: states n = 1..T, step dt, updates on (u, v) as in the simulation.
Both are solved on one surrogate of size N with two independent masked Gaussian matrices G1, G2 (random-projection sampling) and tangent propagation.
References (pretraining recipe on other sample sets) are further static systems on the same surrogate.

Groups of samples: 0 R (retain samples used in finetuning), 1 R_full minus R, 2 F, 3 F_full minus F, 4 fresh forget-class samples (finetuning only).
Blocks of coordinates: 0 retain, 1 shared, 2 forget."""
import numpy as np
from mse_sim import LOSSES, SPEC, BASE

def sqrtm(G):
    w, U = np.linalg.eigh((G + G.T) / 2); return (U * np.sqrt(np.clip(w, 0, None))) @ U.T

USE = [[0, 1, 2, 3], [0, 1, 3], [0, 1]]   # static systems: pretrained (all samples), retrained without F, retain class only

def surrogate(cfg, N, seed, c, iters=1500, tol=1e-9, damp=0.3):
    """Draw the surrogate and solve the static systems on it.
    Static cavity (interpolation from zero at scale c), for system s:  sample: zeta_s + V_s g_s = y;   coordinate: theta_s + A_s (beta_s - beta*) = -eta_s, beta_s = c sinh theta_s.
    (z, zeta) are Gaussian with the Gram matrix of (beta*, beta_s); eta is Gaussian with the Gram matrix of the g_s. Solved with fixed standard normals, then the matrices G1, G2
    are drawn conditionally on G1 beta_s = zeta_s, G1 beta* = z, G2 g_s = eta_s, so the later trajectory pass sees the right correlations."""
    rng = np.random.default_rng(seed); q = np.array(cfg["q"], float); q /= q.sum(); eb = np.r_[0, np.round(np.cumsum(q) * N).astype(int)]; Nc = eb[-1]; rho = cfg["rho"]
    bs = np.zeros(Nc)
    for B in range(3):
        nb = eb[B + 1] - eb[B]; m = int(round(rho * nb / 2)); bs[eb[B]:eb[B] + 2 * m] = np.r_[np.ones(m), -np.ones(m)] * rho**-0.5
    cR, cF = q[0] + q[1], q[1] + q[2]; a_ = cfg["alpha"]; nR, nFf = a_ * cR, a_ * cF; nF = cfg["ffrac"] * nFf
    alt = [cfg["rfrac"] * nR, (1 - cfg["rfrac"]) * nR, nF, nFf - nF, nF if cfg["fresh"] else 0.0]; sup = [[0, 1], [0, 1], [1, 2], [1, 2], [1, 2]]
    eg = np.r_[0, np.cumsum([int(round(x * Nc)) for x in alt])]; Ns = eg[-1]; al = np.array([(eg[g + 1] - eg[g]) / Nc for g in range(5)]); nS = len(USE)
    sig = np.sqrt(np.r_[[cfg["sig2_pt"]] * 4, cfg["sig2_ft"]]); eps = np.concatenate([sig[g] * rng.standard_normal(eg[g + 1] - eg[g]) for g in range(5)])
    from scipy.special import ndtri
    lhs = lambda m, k_: np.column_stack([ndtri((rng.permutation(m) + 0.5) / m) for _ in range(k_)]) if m else np.zeros((0, k_))   # stratified standard normals
    om = np.vstack([lhs(eg[g + 1] - eg[g], nS + 1) for g in range(5)]); h = np.zeros((Nc, nS))
    for B in range(3):
        for val in (-1, 0, 1): idx = np.where(np.sign(bs[eb[B]:eb[B + 1]]) == val)[0] + eb[B]; h[idx] = lhs(len(idx), nS)
    if cfg.get("iid"): om = rng.standard_normal((Ns, nS + 1)); h = rng.standard_normal((Nc, nS))
    rows = [slice(eb[0], eb[2]), slice(eb[0], eb[2]), slice(eb[1], eb[3]), slice(eb[1], eb[3]), slice(eb[1], eb[3])]
    grp = [[g for g in range(5) if B in sup[g] and eg[g + 1] > eg[g]] for B in range(3)]
    if cfg.get("layout_only"): return dict(eb=eb, eg=eg, bs=bs, Nc=Nc, Ns=Ns, q=q, al=al)
    lam = cfg["lam_pt"]; t1 = cfg["t1"]
    V = np.full((nS, 5), -1.0); beta = np.zeros((Nc, nS)); th = np.zeros((Nc, nS)); g_ = np.zeros((Ns, nS)); Z = np.zeros((Ns, nS + 1)); eta = np.zeros((Nc, nS)); jb = np.zeros((Nc, nS))
    for it in range(iters):
        Bm = np.c_[bs, beta]
        for g in range(5):
            if eg[g + 1] == eg[g]: continue
            sl = slice(eg[g], eg[g + 1]); Z[sl] = om[sl] @ sqrtm(Bm[rows[g]].T @ Bm[rows[g]] / Nc)
        y = Z[:, 0] + eps; gn = np.zeros((Ns, nS))
        for s in range(nS):
            for g in USE[s]:
                sl = slice(eg[g], eg[g + 1]); gn[sl, s] = (y[sl] - Z[sl, s + 1]) / V[s, g] if lam == 0 else (Z[sl, s + 1] - y[sl]) / (1 - V[s, g])
        g_ = gn if it == 0 else (1 - damp) * g_ + damp * gn; A = np.zeros((Nc, nS))
        for B in range(3):
            sl = slice(eb[B], eb[B + 1]); idx = np.concatenate([np.arange(eg[g], eg[g + 1]) for g in grp[B]]) if grp[B] else np.zeros(0, int)
            eta[sl] = h[sl] @ sqrtm(g_[idx].T @ g_[idx] / Nc)
            for s in range(nS): A[sl, s] = sum(al[g] * (-1 / V[s, g] if lam == 0 else 1 / (1 - V[s, g])) for g in USE[s] if B in sup[g] and eg[g + 1] > eg[g])
        if lam == 0:
            rhs = -eta + A * bs[:, None]
            for _ in range(80):
                f = th + A * c * np.sinh(th) - rhs; st = f / (1 + A * c * np.cosh(th)); th = np.clip(th - st, -60, 60)
                if np.max(np.abs(st)) < 1e-13: break
            new = c * np.sinh(th); D = np.sqrt(new * new + c * c); jb = -D / (1 + A * D)
        else:   # weight decay lam/2 (|u|^2 + |v|^2) is the penalty lam/2 |beta|: beta = soft(beta* - eta / A, lam / (2A))
            As = np.where(A > 0, A, 1.0); x = bs[:, None] - eta / As; new = np.where(A > 0, np.sign(x) * np.maximum(np.abs(x) - lam / (2 * As), 0), 0.0); jb = np.where((new != 0) & (A > 0), -1 / As, 0.0)
        err = np.sqrt(np.mean((new - beta)**2)); beta = new; Vn = V.copy()
        for s in range(nS):
            for g in USE[s]: Vn[s, g] = sum(q[B] * jb[eb[B]:eb[B + 1], s].mean() for B in sup[g] if eb[B + 1] > eb[B])
        V = np.minimum((1 - damp) * V + damp * Vn, -1e-7)
        if err < tol and it > 30: break
    if cfg.get("static_only"): return dict(eb=eb, eg=eg, bs=bs, Nc=Nc, Ns=Ns, it=it, err=err, beta=beta, V=V, q=q, al=al)
    # conditional draw of the matrices
    G1 = np.zeros((Ns, Nc)); G2 = np.zeros((Nc, Ns)); Bm = np.c_[bs, beta]
    for g in range(5):
        if eg[g + 1] == eg[g]: continue
        sl = slice(eg[g], eg[g + 1]); Br = Bm[rows[g]]; Gt = rng.standard_normal((eg[g + 1] - eg[g], Br.shape[0])) / np.sqrt(Nc); G1[sl, rows[g]] = Gt - (Gt @ Br - Z[sl]) @ np.linalg.pinv(Br.T @ Br, rcond=1e-10) @ Br.T
    for B in range(3):
        if not grp[B]: continue
        sl = slice(eb[B], eb[B + 1]); idx = np.concatenate([np.arange(eg[g], eg[g + 1]) for g in grp[B]]); gr = g_[idx]; Gt = rng.standard_normal((eb[B + 1] - eb[B], len(idx))) / np.sqrt(Nc)
        G2[sl, idx] = Gt - (Gt @ gr - eta[sl]) @ np.linalg.pinv(gr.T @ gr, rcond=1e-10) @ gr.T
    S = dict(eb=eb, eg=eg, bs=bs, G1=G1, G2=G2, z=Z[:, 0], y=y, q=q, sup=sup, al=al, Nc=Nc, Ns=Ns, xi=rng.standard_normal(Ns), mrng=np.random.default_rng(seed + 7), it=it, err=err)
    S["slots"] = []
    for s in range(nS):
        b_ = beta[:, s]
        if lam == 0:
            u = np.sqrt(c / 2) * np.exp(th[:, s] / 2); v = np.sqrt(c / 2) * np.exp(-th[:, s] / 2); J0 = jb[:, s] / np.sqrt(b_ * b_ + c * c); du = u / 2 * J0; dv = -v / 2 * J0; dgdz = -1 / V[s]
        else:   # after a time t1 with decay: u v = c exp(-2 lam t1) / 2. Active: |beta| = u^2 or v^2. Inactive: u, v = sqrt(c/2) exp(-(lam +- 2F) t1) with F the stationary gradient.
            ce = c * np.exp(-2 * lam * t1); F = eta[:, s] + A[:, s] * (b_ - bs); act = b_ != 0; big = np.sqrt(np.abs(b_) + 1e-300); small = ce / (2 * big); As = np.where(A[:, s] > 0, A[:, s], 1.0)
            u = np.where(act, np.where(b_ > 0, big, small), np.sqrt(c / 2) * np.exp(np.clip(-(lam + 2 * F) * t1, -700, 0))); v = np.where(act, np.where(b_ > 0, small, big), np.sqrt(c / 2) * np.exp(np.clip(-(lam - 2 * F) * t1, -700, 0)))
            dbig = -1 / (2 * big * As) * np.sign(b_); du = np.where(act, np.where(b_ > 0, dbig, -small / big * dbig), -2 * t1 * u); dv = np.where(act, np.where(b_ > 0, -small / big * dbig, dbig), 2 * t1 * v); dgdz = 1 / (1 - V[s])
        S["slots"].append(dict(beta=b_, g=g_[:, s], jb=jb[:, s], V=V[s], u=u, v=v, du=du, dv=dv, dgdz=dgdz, use=[g for g in USE[s] if eg[g + 1] > eg[g]], zeta=Z[:, s + 1]))
    return S

def metrics(S, b, p, ys):
    eb, eg, bs = S["eb"], S["eg"], S["bs"]; e = [np.sum((b - bs)[eb[B]:eb[B + 1]]**2) / S["Nc"] for B in range(3)]; r = (p - ys)**2
    mean = lambda g: r[eg[g]:eg[g + 1]].mean() if eg[g + 1] > eg[g] else np.nan
    return dict(m1=e[0] + e[1], m4=e[1] + e[2], m2=mean(2), m3=mean(3), eR=e[0], eSh=e[1], eF=e[2], Q=b @ b / S["Nc"])

def finetune(S, P, cfg, loss, T, dt, refs, c):
    """Forward pass from the static slot P for one finetuning loss. State 0 = pretrained model; states 1..T+1 at times 0, dt, ..."""
    eb, eg, bs, G1, G2, z, y, q, sup, al = (S[k] for k in ("eb", "eg", "bs", "G1", "G2", "z", "y", "q", "sup", "al")); Nc, Ns = S["Nc"], S["Ns"]; M = T + 2
    wr, sg, bt, nz = SPEC[loss]; mur, muf = cfg["mu_signed"] if (loss == "signed" and cfg["mu_signed"] is not None) else cfg["mu"]; fg = 4 if cfg["fresh"] else 2
    w = np.zeros(5); w[0] = wr * mur; w[fg] = sg * muf; tgt = y.copy(); sl = slice(eg[fg], eg[fg + 1]); tgt[sl] = bt * y[sl] + nz * S["xi"][sl]; bl = np.ones(Ns); bl[sl] = bt
    return core(S, P, w, tgt, cfg["lam_ft"], cfg["phi_ft"], np.sqrt(cfg["S"]), T, dt, refs, c, bl=bl)

def core(S, P, w, tgt, lam, phi, rS, T, dt, refs, c, bl=None):
    """bl: d target / d label per sample (default 1). The response of the gradients to the label is propagated exactly (Tz), not estimated."""
    eb, eg, bs, G1, G2, z, y, q, sup, al = (S[k] for k in ("eb", "eg", "bs", "G1", "G2", "z", "y", "q", "sup", "al")); Nc, Ns = S["Nc"], S["Ns"]; M = T + 2
    va = np.array([sum(q[B] for B in sup[g]) for g in range(5)])
    beta = np.zeros((Nc, M)); JB = np.zeros((Nc, M, M)); g = np.zeros((Ns, M)); Tg = np.zeros((Ns, M, M)); RbB = np.zeros((3, M, M)); MB = np.zeros((3, M))
    beta[:, 0] = P["beta"]; JB[:, 0, 0] = P["jb"]; g[:, 0] = P["g"]
    Tz = np.zeros((Ns, M)); bl = np.ones(Ns) if bl is None else bl
    for a in P["use"]: Tg[eg[a]:eg[a + 1], 0, 0] = P["dgdz"][a]; Tz[eg[a]:eg[a + 1], 0] = -P["dgdz"][a]
    NB = [B for B in range(3) if eb[B + 1] > eb[B]]
    for B in NB: s_ = slice(eb[B], eb[B + 1]); RbB[B, 0, 0] = P["jb"][s_].mean(); MB[B, 0] = P["beta"][s_] @ bs[s_] / (eb[B + 1] - eb[B])
    u = P["u"] * rS; v = P["v"] * rS; du = np.zeros((Nc, M)); dv = np.zeros((Nc, M)); du[:, 0] = P["du"] * rS; dv[:, 0] = P["dv"] * rS
    out = {}; cap = 400.0 * (bs @ bs) / Nc * 1.0; ypre = y
    for n in range(1, M):
        b = u * u - v * v; jb = 2 * (u[:, None] * du - v[:, None] * dv); beta[:, n] = b; JB[:, n, :] = jb
        for B in NB: s_ = slice(eb[B], eb[B + 1]); RbB[B, n, :n] = jb[s_, :n].mean(0); MB[B, n] = b[s_] @ bs[s_] / (eb[B + 1] - eb[B])
        p = G1 @ b; Rg = np.zeros((5, n + 1)); dz = np.zeros(5); wm = (S["mrng"].random(Ns) < phi) / phi if phi is not None else np.ones(Ns)
        for a in range(5):
            if eg[a + 1] == eg[a]: continue
            s_ = slice(eg[a], eg[a + 1]); Rb = sum(q[B] * RbB[B, n, :n] for B in sup[a]); p[s_] += g[s_, :n] @ Rb
            if w[a] != 0:
                tp = np.tensordot(Rb, Tg[s_, :n, :n + 1], axes=(0, 1)); tp[:, n] += 1.0
                g[s_, n] = wm[s_] * w[a] * (p[s_] - tgt[s_]); Tg[s_, n, :n + 1] = (wm[s_] * w[a])[:, None] * tp
                Tz[s_, n] = wm[s_] * w[a] * (Tz[s_, :n] @ Rb - bl[s_]); Rg[a] = Tg[s_, n, :n + 1].mean(0); dz[a] = Tz[s_, n].mean()
        m = metrics(S, b, p, ypre)
        for name, r in refs.items(): m["d_" + name] = np.sum((b - r)**2) / Nc
        for key, val in m.items(): out.setdefault(key, np.full(T + 1, np.nan))[n - 1] = val
        if n == M - 1 or b @ b / Nc > cap: break
        Gf = G2 @ g[:, n]; dG = np.zeros((Nc, M))
        for B in NB:
            s_ = slice(eb[B], eb[B + 1]); ga = [a for a in range(5) if B in sup[a] and w[a] != 0 and eg[a + 1] > eg[a]]
            if not ga: continue
            R = sum(al[a] * Rg[a] for a in ga); Gf[s_] += beta[s_, :n + 1] @ R + sum(al[a] * dz[a] for a in ga) * bs[s_]; dG[s_, :n + 1] = np.tensordot(R, JB[s_, :n + 1, :n + 1], axes=(0, 1))
        dG[:, n] += 1.0
        du = du * (1 - 2 * dt * Gf - dt * lam)[:, None] - 2 * dt * u[:, None] * dG; dv = dv * (1 + 2 * dt * Gf - dt * lam)[:, None] + 2 * dt * v[:, None] * dG
        u = u * (1 - 2 * dt * Gf - dt * lam); v = v * (1 + 2 * dt * Gf - dt * lam)
    return out

def pretrain_theory(cfg, seed, N=3000, T=200, dt=0.05, used=(0, 1, 2, 3)):
    """Trajectory of pretraining from zero (unconditioned surrogate): weight 1 and the labels on the groups in `used`."""
    cfg = {**BASE, **cfg}; c = np.sqrt(cfg["k"]) / 2; S0 = surrogate({**cfg, "static_only": 1, "layout_only": 1}, N, seed, c); rng = np.random.default_rng(seed + 31); eb, eg, Nc, Ns = S0["eb"], S0["eg"], S0["Nc"], S0["Ns"]
    sup = [[0, 1], [0, 1], [1, 2], [1, 2], [1, 2]]; mask = np.zeros((Ns, Nc), bool)
    for g in range(5):
        for B in sup[g]: mask[eg[g]:eg[g + 1], eb[B]:eb[B + 1]] = True
    G1 = rng.standard_normal((Ns, Nc)) / np.sqrt(Nc) * mask; G2 = rng.standard_normal((Nc, Ns)) / np.sqrt(Nc) * mask.T; z = G1 @ S0["bs"]; sig = np.sqrt(np.r_[[cfg["sig2_pt"]] * 4, cfg["sig2_ft"]])
    y = z + np.concatenate([sig[g] * rng.standard_normal(eg[g + 1] - eg[g]) for g in range(5)])
    S = dict(eb=eb, eg=eg, bs=S0["bs"], G1=G1, G2=G2, z=z, y=y, q=S0["q"], sup=sup, al=S0["al"], Nc=Nc, Ns=Ns, mrng=np.random.default_rng(seed + 7))
    P = dict(beta=np.zeros(Nc), jb=np.zeros(Nc), g=np.zeros(Ns), u=np.full(Nc, np.sqrt(c / 2)), v=np.full(Nc, np.sqrt(c / 2)), du=np.zeros(Nc), dv=np.zeros(Nc), use=[]); w = np.zeros(5); w[list(used)] = 1.0
    return dict(cfg=cfg, seed=seed, dt=dt, pre=core(S, P, w, y, cfg["lam_pt"], cfg["phi_pt"], 1.0, T, dt, {}, c))

def theory(cfg, seed, N=2000, T=200, dt=0.05, losses=LOSSES):
    cfg = {**BASE, **cfg}; c = np.sqrt(cfg["k"]) / 2; S = surrogate(cfg, N, seed, c); P, U, L = S["slots"]; refs = dict(untrain=U["beta"], unlearn=L["beta"]); res = dict(cfg=cfg, seed=seed, conv=(S["it"], S["err"]))
    for nm, X in dict(pre=P, ref_untrain=U, ref_unlearn=L).items(): res[nm + "_end"] = metrics(S, X["beta"], X["zeta"] + X["V"][np.searchsorted(S["eg"], np.arange(S["Ns"]), side="right") - 1] * X["g"], S["y"])
    res["ft"] = {l: finetune(S, P, cfg, l, T, dt, refs, c) for l in losses}; return res

if __name__ == "__main__":
    import sys, time
    N, T = int(sys.argv[1]), int(sys.argv[2]); t = time.time(); r = theory({}, 0, N=N, T=T, losses=LOSSES if len(sys.argv) < 4 else sys.argv[3].split(",")); print("seconds", time.time() - t, "static convergence", r["conv"])
    for nm in ("pre", "ref_untrain", "ref_unlearn"): print(nm, {k: round(float(x), 4) for k, x in r[nm + "_end"].items()})
    for key in ("m1", "m2", "m3", "m4", "d_untrain"):
        print(key)
        for i in [0, 10, 20, 40, 100, 200, 300]:
            if i <= T: print(f"  t={i*0.05:5.1f}", np.round([r["ft"][l][key][i] for l in r["ft"]], 4))
