"""Two-stage exact large-d dynamics with hard labels. Stage 1 (t < T1): cross-entropy on all n = al d samples, y = sign(x^T beta*).
Stage 2: the forget fraction f of the samples gets an unlearning loss, the rest weight wr on cross-entropy.
  ga:      -log-likelihood ascent,            dL/dp = -l'(p)
  flip:    cross-entropy on the flipped label, dL/dp = l'(p; -y)
  npo:     (2/bN) log(1 + exp(bN (l0 - l))),   dL/dp = y 2 sigmoid(bN (l0 - l)) sigmoid(-y p),  l0 = loss at the end of stage 1
Same cavity equations as dmft_nl.py; the only change is g(p) and its derivative."""
import numpy as np, sys, pickle
from scipy.special import expit
from multiprocessing import Pool
bN = 1.0

def grad(kind, m, m0):
    """returns dL/dm, d2L/dm2, d(dL/dm)/dm0 with m = y p the margin"""
    s, l2 = expit(-m), expit(m) * expit(-m)
    if kind == "ce":   return -s, l2, 0 * m
    if kind == "ga":   return s, -l2, 0 * m
    if kind == "flip": return expit(m), l2, 0 * m
    if kind == "npo":
        r = np.logaddexp(0, -m0) - np.logaddexp(0, -m); w = expit(bN * r); dw = bN * w * (1 - w)
        return 2 * w * s, 2 * (dw * s * s - w * l2), 2 * dw * (-expit(-m0)) * s
    if kind == "off":  return 0 * m, 0 * m, 0 * m

def metrics(out, n, b, bs, nb, y, p, F, ref=None):
    out["Q"][n] = b @ b / nb; out["M"][n] = b @ bs / nb
    if ref is not None: out["dist"][n] = np.sum((b - ref)**2) / nb; out["cos"][n] = b @ ref / np.sqrt((b @ b) * (ref @ ref) + 1e-300)
    out["lossF"][n] = np.mean(np.logaddexp(0, -y[F] * p[F])); out["lossR"][n] = np.mean(np.logaddexp(0, -y[~F] * p[~F])); out["accF"][n] = np.mean(y[F] * p[F] > 0)

def blank(T): return {k: np.zeros(T) for k in ("Q", "M", "lossF", "lossR", "accF", "dist", "cos")}

def run(al, f, kind, wr, n1, T, dt, k, rho, N=3000, seed=0, ref=None, retrain=False):
    """retrain=True: cross-entropy on the kept samples only, from zero, for n1 steps; returns the weights (same randomness as the main run)."""
    if ref is None and not retrain: ref = run(al, f, kind, wr, n1, n1 + 1, dt, k, rho, N, seed, retrain=True)
    rng = np.random.default_rng(seed); c = np.sqrt(k) / 2; Ns = int(round(al * N)); F = np.arange(Ns) < int(round(f * Ns))
    na = int(round(rho * N / 2)); bs = np.zeros(N); bs[:na] = rho**-0.5; bs[na:2 * na] = -rho**-0.5; bs /= np.sqrt(np.mean(bs**2))
    G1 = rng.standard_normal((Ns, N)) / np.sqrt(N); G2 = rng.standard_normal((N, Ns)) / np.sqrt(N); z = G1 @ bs; y = np.sign(z)
    th = np.zeros(N); beta = np.zeros((N, T)); D = np.zeros((N, T)); J = np.zeros((N, T, T)); Jn = np.zeros((N, T))
    g = np.zeros((Ns, T)); Tg = np.zeros((Ns, T, T)); Rb = np.zeros((T, T)); M = np.zeros(T); out = blank(T); m0 = None; tp0 = None
    for n in range(T):
        b = c * np.sinh(th); beta[:, n] = b; D[:, n] = np.sqrt(b * b + c * c); J[:, n, :] = Jn
        Rb[n, :n] = (D[:, n] @ Jn[:, :n]) / N; M[n] = b @ bs / N
        p = G1 @ b + g[:, :n] @ Rb[n, :n]; m = y * p; metrics(out, n, b, bs, N, y, p, F, ref)
        if retrain and n == n1: return b
        tp = np.einsum("l,ilm->im", Rb[n, :n], Tg[:, :n, :]); tp[:, n] += 1.0          # d p[n] / d zeta[.]
        if n == n1: m0, tp0 = m.copy(), tp.copy()
        if n < n1:
            g1, g2, g3 = grad("ce", m, m)
            if retrain: g1, g2 = np.where(F, 0.0, g1), np.where(F, 0.0, g2)
        else:
            g1, g2, g3 = grad(kind, m, m0); r1, r2, r3 = grad("ce", m, m0)
            g1, g2, g3 = np.where(F, g1, wr * r1), np.where(F, g2, wr * r2), np.where(F, g3, 0.0)
        g[:, n] = y * g1; Tg[:, n, :] = g2[:, None] * tp + (g3[:, None] * tp0 if n >= n1 else 0.0)
        Rgt = Tg[:, n, :n + 1].mean(0); mu = al * (np.mean(z * g[:, n]) - Rgt @ M[:n + 1])
        Gf = G2 @ g[:, n] + al * (beta[:, :n + 1] @ Rgt) + mu * bs
        dG = al * np.einsum("l,il,ilm->im", Rgt, D[:, :n + 1], J[:, :n + 1, :]); dG[:, n] += 1.0
        th = th - 4 * dt * Gf; Jn = Jn - 4 * dt * dG
    return out

def simulate(al, f, kind, wr, n1, T, dt, k, rho, d=3000, seed=0, ref=None, retrain=False):
    if ref is None and not retrain: ref = simulate(al, f, kind, wr, n1, n1 + 1, dt, k, rho, d, seed, retrain=True)
    rng = np.random.default_rng(seed + 10**6); c = np.sqrt(k) / 2; n = int(al * d); F = np.arange(n) < int(round(f * n))
    bs = np.zeros(d); bs[:int(rho * d)] = rng.choice([-1, 1], int(rho * d)) * rho**-0.5; X = rng.standard_normal((d, n)) / np.sqrt(d); y = np.sign(X.T @ bs)
    th = np.zeros(d); out = blank(T); nb = bs @ bs; m0 = None
    for i in range(T):
        b = c * np.sinh(th); p = X.T @ b; m = y * p; metrics(out, i, b, bs, nb, y, p, F, ref)
        if retrain and i == n1: return b
        if i == n1: m0 = m.copy()
        g1 = (np.where(F, 0.0, grad("ce", m, m)[0]) if retrain else grad("ce", m, m)[0]) if i < n1 else np.where(F, grad(kind, m, m0)[0], wr * grad("ce", m, m0)[0])
        th = th - 4 * dt * (X @ (y * g1))
    return out

def job(a): return (run if a[0] == "th" else simulate)(*a[1:])

if __name__ == "__main__":
    kind, k, wr, nrun = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]); al, f, rho, dt = 0.6, 0.5, 0.2, 0.05
    T1, T2 = float(sys.argv[5]), float(sys.argv[6]); n1 = int(round(T1 / dt)); T = n1 + int(round(T2 / dt))
    with Pool(2) as p: R = p.map(job, [(w, al, f, kind, wr, n1, T, dt, k, rho, 3000, s) for w in ("th", "si") for s in range(nrun)])
    TH, SI = R[:nrun], R[nrun:]; pickle.dump(dict(dt=dt, n1=n1, TH=TH, SI=SI), open(f"nl2_{kind}_{k:g}_{wr:g}.pkl", "wb"))
    st = lambda L, key: (np.mean([o[key] for o in L], 0), np.std([o[key] for o in L], 0, ddof=1) / np.sqrt(len(L)))
    print(f"{kind}, k={k:g}, retain weight {wr:g}, {nrun} runs: theory / sim (z)")
    for i in [n1] + [n1 + int(round(x * (T - 1 - n1))) for x in (0.25, 0.5, 0.75, 1.0)]:
        row = f"t2={dt*(i-n1):5.2f} |"
        for key in ("dist", "cos", "accF", "lossR", "Q"):
            (a, ae), (b, be) = st(TH, key), st(SI, key); row += f" {key} {a[i]:.4f}/{b[i]:.4f} ({(a[i]-b[i])/max(np.hypot(ae[i],be[i]),1e-9):+.1f}) |"
        print(row)
