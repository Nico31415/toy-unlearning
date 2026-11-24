"""Exact large-d dynamics for a diagonal network trained by full-batch gradient flow on ANY per-sample loss l(p; y),
here hard-label cross-entropy l = log(1 + exp(-y p)), y = sign(x^T beta*). Mirror form: theta' = -4 X l'(X^T beta), beta = c sinh theta.
Cavity equations (al = n/d):
  sample:      p(t) = zeta(t) + int R_b(t,s) g(s) ds,   g = l'(p),   (z, zeta) Gaussian: Cov(z, zeta(t)) = M(t), Cov(zeta(t), zeta(s)) = Q(t,s)
  coordinate:  theta' = -4 [ eta + int R_g(t,s) beta(s) ds + mu(t) beta* ],   eta ~ GP(0, al E g(t) g(s))
  R_b(t,s) = E d beta(t)/d eta(s),   R_g(t,s) = al E d g(t)/d zeta(s),   mu(t) = al ( E[z g(t)] - int R_g(t,s) M(s) ds / al )
Solved by one causal forward pass over sampled paths; (z, zeta) and eta are sampled by random projections."""
import numpy as np, sys, time
from scipy.special import expit

def run(al, ts, k, rho, N=1500, seed=0):
    rng = np.random.default_rng(seed); T = len(ts) - 1; dt = ts[1] - ts[0]; c = np.sqrt(k) / 2; Ns = int(round(al * N))
    na = int(round(rho * N / 2)); bs = np.zeros(N); bs[:na] = rho**-0.5; bs[na:2 * na] = -rho**-0.5; bs /= np.sqrt(np.mean(bs**2))
    G1 = rng.standard_normal((Ns, N)) / np.sqrt(N); G2 = rng.standard_normal((N, Ns)) / np.sqrt(N)
    z = G1 @ bs; y = np.sign(z)
    th = np.zeros(N); beta = np.zeros((N, T)); D = np.zeros((N, T)); J = np.zeros((N, T, T)); Jn = np.zeros((N, T))
    p = np.zeros((Ns, T)); g = np.zeros((Ns, T)); l2 = np.zeros((Ns, T)); Ts = np.zeros((Ns, T, T))
    Rb = np.zeros((T, T)); Rg = np.zeros((T, T)); M = np.zeros(T); out = dict(Q=np.zeros(T), M=M, loss=np.zeros(T), acc=np.zeros(T))
    for n in range(T):
        b = c * np.sinh(th); beta[:, n] = b; D[:, n] = np.sqrt(b * b + c * c); J[:, n, :] = Jn
        Rb[n, :n] = (D[:, n] @ Jn[:, :n]) / N; M[n] = b @ bs / N; out["Q"][n] = b @ b / N
        zeta = G1 @ b; p[:, n] = zeta + g[:, :n] @ Rb[n, :n]
        g[:, n] = -y * expit(-y * p[:, n]); l2[:, n] = expit(p[:, n]) * expit(-p[:, n])
        out["loss"][n] = np.mean(np.logaddexp(0, -y * p[:, n])); out["acc"][n] = np.mean(y * p[:, n] > 0)
        if n: Ts[:, n, :] = np.einsum("l,il,ilm->im", Rb[n, :n], l2[:, :n], Ts[:, :n, :])
        Ts[:, n, n] += 1.0
        Rgt = (l2[:, n] @ Ts[:, n, :n + 1]) / Ns                               # E d g[n] / d zeta[m]
        Rg[n, :n + 1] = al * Rgt; mu = al * (np.mean(z * g[:, n]) - Rgt @ M[:n + 1])
        Gf = G2 @ g[:, n] + beta[:, :n + 1] @ Rg[n, :n + 1] + mu * bs
        dG = np.einsum("l,il,ilm->im", Rg[n, :n + 1], D[:, :n + 1], J[:, :n + 1, :]); dG[:, n] += 1.0
        th = th - 4 * dt * Gf; Jn = Jn - 4 * dt * dG
    return out

def simulate(al, ts, k, rho, d=2000, seed=0):
    rng = np.random.default_rng(seed); T = len(ts) - 1; dt = ts[1] - ts[0]; c = np.sqrt(k) / 2
    bs = np.zeros(d); bs[:int(rho * d)] = rng.choice([-1, 1], int(rho * d)) * rho**-0.5; X = rng.standard_normal((d, int(al * d))) / np.sqrt(d); y = np.sign(X.T @ bs)
    th = np.zeros(d); out = dict(Q=np.zeros(T), M=np.zeros(T), loss=np.zeros(T), acc=np.zeros(T)); nb = bs @ bs
    for n in range(T):
        b = c * np.sinh(th); p = X.T @ b; out["Q"][n] = b @ b / nb; out["M"][n] = b @ bs / nb; out["loss"][n] = np.mean(np.logaddexp(0, -y * p)); out["acc"][n] = np.mean(y * p > 0)
        th = th - 4 * dt * (X @ (-y * expit(-y * p)))
    return out

if __name__ == "__main__":
    k = float(sys.argv[1]); al, rho, dt, Tend = 0.3, 0.2, 0.05, float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    ts = np.arange(0, Tend + dt / 2, dt); t0 = time.time()
    TH = [run(al, ts, k, rho, N=1500, seed=s) for s in range(3)]; print(f"k={k:g}: 3 theory runs in {time.time()-t0:.0f}s", flush=True)
    SI = [simulate(al, ts, k, rho, d=3000, seed=s) for s in range(6)]
    st = lambda L, key: (np.mean([o[key] for o in L], 0), np.std([o[key] for o in L], 0, ddof=1) / np.sqrt(len(L)))
    print("   t  | norm: theory±se / sim±se | overlap: theory / sim | train loss: theory / sim | train accuracy: theory / sim")
    for tt in (0.5, 1, 2, 4, 7, Tend - dt):
        i = int(round(tt / dt)) if tt < Tend - dt else len(ts) - 2
        row = f"{ts[i]:5.2f} |"
        for key in ("Q", "M", "loss", "acc"):
            (a, ae), (b, be) = st(TH, key), st(SI, key); row += f" {a[i]:.4f}±{ae[i]:.4f} / {b[i]:.4f}±{be[i]:.4f} |"
        print(row)
    import pickle; pickle.dump(dict(ts=ts, TH=TH, SI=SI), open(f"dmft_nl_{k:g}.pkl", "wb"))
