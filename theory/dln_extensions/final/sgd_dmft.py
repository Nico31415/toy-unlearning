"""Exact large-d curves for mini-batch SGD on the diagonal network (squared loss, labels y = x^T beta*), any step size and batch fraction.
One step:  G = (1/phi) X (mask * (X^T beta - y)),  mask_i ~ Bernoulli(phi) fresh each step (phi = b/n),   u <- u (1 - 2 gam G),  v <- v (1 + 2 gam G),  beta = u^2 - v^2.
Cavity:    sample  p[n] = zeta[n] + sum_{m<n} Rb[n,m] g[m],  g[n] = mask[n] (p[n] - z) / phi
           coordinate  G[n] = eta[n] + sum_{l<=n} Rg[n,l] beta[l] + mu[n] beta*,   eta ~ N(0, al E g g'),  Rg = al E dg/dzeta,  Rb = E dbeta/deta."""
import numpy as np, sys, pickle
from multiprocessing import Pool

def run(al, gam, phi, T, k, rho, N=2000, seed=0):
    rng = np.random.default_rng(seed); c = np.sqrt(k) / 2; Ns = int(round(al * N))
    na = int(round(rho * N / 2)); bs = np.zeros(N); bs[:na] = rho**-0.5; bs[na:2 * na] = -rho**-0.5
    G1 = rng.standard_normal((Ns, N)) / np.sqrt(N); G2 = rng.standard_normal((N, Ns)) / np.sqrt(N); z = G1 @ bs
    u = np.full(N, np.sqrt(c / 2)); v = u.copy(); du = np.zeros((N, T)); dv = np.zeros((N, T)); beta = np.zeros((N, T)); J = np.zeros((N, T, T))
    g = np.zeros((Ns, T)); Tg = np.zeros((Ns, T, T)); Rb = np.zeros((T, T)); M = np.zeros(T); out = {key: np.zeros(T) for key in ("Q", "M", "err", "loss", "kap")}
    for n in range(T):
        b = u * u - v * v; beta[:, n] = b; Jn = 2 * (u[:, None] * du - v[:, None] * dv); J[:, n, :] = Jn
        Rb[n, :n] = Jn[:, :n].mean(0); M[n] = b @ bs / N; out["Q"][n] = b @ b / N; out["M"][n] = M[n]; out["err"][n] = np.mean((b - bs)**2); out["kap"][n] = np.mean(np.log(2 * u * v / c))
        p = G1 @ b + g[:, :n] @ Rb[n, :n]; out["loss"][n] = np.mean((p - z)**2)
        w = (rng.random(Ns) < phi) / phi; g[:, n] = w * (p - z)
        tp = np.einsum("l,ilm->im", Rb[n, :n], Tg[:, :n, :]); tp[:, n] += 1.0; Tg[:, n, :] = w[:, None] * tp
        Rgt = Tg[:, n, :n + 1].mean(0); mu = al * (np.mean(z * g[:, n]) - Rgt @ M[:n + 1])
        Gf = G2 @ g[:, n] + al * (beta[:, :n + 1] @ Rgt) + mu * bs
        dG = al * np.einsum("l,ilm->im", Rgt, J[:, :n + 1, :]); dG[:, n] += 1.0
        du = du * (1 - 2 * gam * Gf)[:, None] - 2 * gam * u[:, None] * dG; dv = dv * (1 + 2 * gam * Gf)[:, None] + 2 * gam * v[:, None] * dG
        u = u * (1 - 2 * gam * Gf); v = v * (1 + 2 * gam * Gf)
    return out

def simulate(al, gam, phi, T, k, rho, d=3000, seed=0):
    rng = np.random.default_rng(seed + 10**6); c = np.sqrt(k) / 2; n = int(al * d); bs = np.zeros(d); bs[:int(rho * d)] = rng.choice([-1, 1], int(rho * d)) * rho**-0.5
    X = rng.standard_normal((d, n)) / np.sqrt(d); y = X.T @ bs; u = np.full(d, np.sqrt(c / 2)); v = u.copy(); out = {key: np.zeros(T) for key in ("Q", "M", "err", "loss", "kap")}
    for i in range(T):
        b = u * u - v * v; r = X.T @ b - y; out["Q"][i] = b @ b / d; out["M"][i] = b @ bs / d; out["err"][i] = np.mean((b - bs)**2); out["loss"][i] = np.mean(r * r); out["kap"][i] = np.mean(np.log(2 * u * v / c))
        G = X @ ((rng.random(n) < phi) / phi * r); u = u * (1 - 2 * gam * G); v = v * (1 + 2 * gam * G)
    return out

def job(a): return (run if a[0] == "th" else simulate)(*a[1:])

if __name__ == "__main__":
    k, gam, phi, T, nrun = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); al, rho = 0.3, 0.2
    with Pool(2) as p: R = p.map(job, [(w, al, gam, phi, T, k, rho, 2000 if w == "th" else 3000, s) for w in ("th", "si") for s in range(nrun)])
    TH, SI = R[:nrun], R[nrun:]; pickle.dump(dict(TH=TH, SI=SI, gam=gam, phi=phi), open(f"sgd_{k:g}_{gam:g}_{phi:g}.pkl", "wb"))
    st = lambda L, key: (np.mean([o[key] for o in L], 0), np.std([o[key] for o in L], 0, ddof=1) / np.sqrt(len(L)))
    print(f"k={k:g}, step {gam:g}, batch fraction {phi:g}, {nrun} runs: theory / sim (z)")
    for i in [int(x * (T - 1)) for x in (0.1, 0.25, 0.5, 0.75, 1.0)]:
        row = f"step {i:4d} |"
        for key in ("Q", "err", "loss", "kap"):
            (a, ae), (b, be) = st(TH, key), st(SI, key); row += f" {key} {a[i]:.4f}/{b[i]:.4f} ({(a[i]-b[i])/max(np.hypot(ae[i],be[i]),1e-12):+.1f}) |"
        print(row, flush=True)
