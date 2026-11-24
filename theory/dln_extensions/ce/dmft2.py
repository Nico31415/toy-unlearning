"""Multi-system two-stage DMFT (see dmft.py for the equations). Several gradient flows share the same data:
system s has per-group weights W[s][a][t] and targets A[s][a][t] * x^T beta*. Used for: main run (pretrain, then stage 2)
and the reference run (kept samples only), so that distances between them are available.
Returns Q[s,s'](t,t') = E beta_s(t) beta_s'(t'), M[s](t) = E beta_s(t) beta*, and per-group residuals."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import numpy as np

def run(delta, P, W, A, ts, k, rho, N=2000, seed=0, dtype=np.float64):
    """P[a]; W, A: arrays [S, ng, T+1]."""
    rng = np.random.default_rng(seed)
    W, A, P = np.asarray(W, float), np.asarray(A, float), np.asarray(P, float)
    S, ng = W.shape[:2]; T = len(ts) - 1; dt = ts[1] - ts[0]; c = np.sqrt(k) / 2
    na = int(round(rho * N / 2)); bs = np.zeros(N); bs[:na] = rho ** -0.5; bs[na:2 * na] = -rho ** -0.5; bs /= np.sqrt(np.mean(bs ** 2))   # exact teacher composition (stratified)
    Gm = [rng.standard_normal((N, N)) / np.sqrt(N) for _ in range(ng)]
    zeta = np.zeros((S, ng, N, T)); R = np.zeros((S, T, T)); K = np.zeros((S, ng, T, T)); res = np.zeros((S, ng, T))
    th = np.zeros((S, N)); beta = np.zeros((S, N, T)); Dm = np.zeros((S, N, T))
    J = np.zeros((S, N, T, T), dtype=dtype); Jn = np.zeros((S, N, T), dtype=dtype)
    for n in range(T):
        for s in range(S):
            b = c * np.sinh(th[s]); beta[s, :, n] = b; Dm[s, :, n] = np.sqrt(b * b + c * c); J[s, :, n, :] = Jn[s]
            R[s, n, :n] = (Dm[s, :, n] @ Jn[s][:, :n]) / N
            eta = np.zeros(N)
            for a in range(ng):
                K[s, a, n, n] = 1.0
                if n: K[s, a, n, :n] = (R[s, n, :n] * W[s, a, :n]) @ K[s, a, :n, :n]
                zeta[s, a, :, n] = Gm[a] @ (b - A[s, a, n] * bs)
                r = zeta[s, a, :, :n + 1] @ K[s, a, n, :n + 1]
                res[s, a, n] = np.mean(r * r)
                eta += np.sqrt(delta * P[a]) * W[s, a, n] * r
            An = delta * (P * W[s, :, n]) @ K[s, :, n, :n + 1]
            Bn = delta * np.sum(P * W[s, :, n] * np.einsum("am,am->a", K[s, :, n, :n + 1], A[s, :, :n + 1]))
            G = eta + beta[s, :, :n + 1] @ An - Bn * bs
            dG = np.einsum("l,il,ilm->im", An, Dm[s, :, :n + 1], J[s, :, :n + 1, :]).astype(dtype)
            dG[:, n] += 1.0
            th[s] = th[s] - 4 * dt * G; Jn[s] = Jn[s] - 4 * dt * dG
    Q = np.einsum("sin,uim->sunm", beta, beta) / N
    M = np.einsum("sin,i->sn", beta, bs) / N
    return dict(t=ts[:T], Q=Q, M=M, res=res)

def simulate(delta, P, W, A, ts, k, rho, d=800, seed=0):
    rng = np.random.default_rng(seed)
    W, A, P = np.asarray(W, float), np.asarray(A, float), np.asarray(P, float)
    S, ng = W.shape[:2]; T = len(ts) - 1; dt = ts[1] - ts[0]; c = np.sqrt(k) / 2
    bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho ** -0.5, 0.0); nb = bs @ bs
    n = int(round(delta * d)); X = rng.standard_normal((d, n)) / np.sqrt(d); z = X.T @ bs; mz = np.mean(z ** 2)
    sizes = np.round(np.cumsum(np.r_[0, P]) * n).astype(int)
    th = np.zeros((S, d)); beta = np.zeros((S, d, T)); res = np.zeros((S, ng, T))
    for nn in range(T):
        for s in range(S):
            b = c * np.sinh(th[s]); beta[s, :, nn] = b; p = X.T @ b; g = np.zeros(d)
            for a in range(ng):
                sl = slice(sizes[a], sizes[a + 1]); r = p[sl] - A[s, a, nn] * z[sl]
                res[s, a, nn] = np.mean(r ** 2) / mz; g += W[s, a, nn] * (X[:, sl] @ r)
            th[s] = th[s] - 4 * dt * g
    return dict(beta=beta, bs=bs, res=res, nb=nb)

def setup(aPT, aUL, T1, T2, dt, b_target=0.0, anchor=False):
    """Groups: 0 = deleted F, 1 = kept K. System 0 = pretrain on all, then stage 2. System 1 = reference trained on K only."""
    ts = np.arange(0, T1 + T2 + dt / 2, dt); s2 = (ts >= T1 - 1e-9).astype(float); one = np.ones_like(ts)
    P = [aUL / aPT, 1 - aUL / aPT]
    W = [[one, 1 - s2 if not anchor else one], [0 * one, one]]
    A = [[1 - s2 + b_target * s2, one], [one, one]]
    return ts, P, W, A, int(round(T1 / dt))

def metrics(Q, M, i1, last):
    """untrain(t) = E (beta_0(t) - beta_1(last))^2 ; unlearn(t) = E beta_0(t)^2  (all-forget: A(D\\F_full) = 0)."""
    T = Q.shape[-1]
    un = np.array([Q[0, 0, t, t] - 2 * Q[0, 1, t, last] + Q[1, 1, last, last] for t in range(T)])
    return un, np.diag(Q[0, 0])

if __name__ == "__main__":
    import sys, time
    k = float(sys.argv[1]); T1 = float(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    dt, T2, rho, aPT, aUL = 0.05, 4.0, 0.2, 0.3, 0.2
    ts, P, W, A, i1 = setup(aPT, aUL, T1, T2, dt)
    t0 = time.time(); th = run(aPT, P, W, A, ts, k, rho, N=N); print(f"dmft {time.time()-t0:.0f}s (N={N})")
    un, ul = metrics(th["Q"], th["M"], i1, len(ts) - 2)
    sims = []
    for s in range(4):
        r = simulate(aPT, P, W, A, ts, k, rho, d=800, seed=s); b = r["beta"]; last = b.shape[2] - 1
        sims.append([np.sum((b[0] - b[1][:, [last]]) ** 2, 0) / r["nb"], np.sum(b[0] ** 2, 0) / r["nb"], r["res"][0, 1], np.sum(b[1] ** 2, 0) / r["nb"]])
    sims = np.array(sims); m, e = sims.mean(0), sims.std(0, ddof=1) / 2
    print("  t-T1 | untraining dist th / sim±se | unlearning dist th / sim±se | kept residual th / sim | ref norm th / sim")
    for off in (-0.05, 0.1, 0.25, 0.5, 1, 2, 3, 3.9):
        i = i1 + int(round(off / dt))
        print(f"{off:6.2f} | {un[i]:.3f} / {m[0][i]:.3f}±{e[0][i]:.3f} | {ul[i]:.3f} / {m[1][i]:.3f}±{e[1][i]:.3f} | {th['res'][0,1,i]:.3f} / {m[2][i]:.3f} | {th['Q'][1,1,i,i]:.3f} / {m[3][i]:.3f}")
