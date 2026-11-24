"""Hard labels: task vector and WHP (generic-target relabelling), exact large-d dynamics vs simulation.
Phase 1 (steps < n1): cross-entropy on all samples. Phase 2 (n1 <= step < n2): 'reinforce' = cross-entropy on the forget samples only.
  task vector: weights  w = w(n1) - lam (w(n2) - w(n1))  on (u, v);  per coordinate u = sqrt(c/2) e^{theta/2}, v = sqrt(c/2) e^{-theta/2}.
  WHP: at n2 the weights are reset to w(n1); phase 3 trains the forget samples to the soft target sigmoid(t),
       t = m(n1) - a max(m(n2) - m(n1), 0)  (m = margin; the reinforced model's extra confidence is subtracted), retain samples keep weight wr.
A reset is a causal map of the history, so the same forward pass applies."""
import numpy as np, sys, pickle
from scipy.special import expit
from multiprocessing import Pool
from dmft_nl2 import grad
LAMS = np.array([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])

def tv_metrics(c, th1, th2, bs, ref):
    out = []
    for lam in LAMS:
        u = np.sqrt(c / 2) * ((1 + lam) * np.exp(th1 / 2) - lam * np.exp(th2 / 2)); v = np.sqrt(c / 2) * ((1 + lam) * np.exp(-th1 / 2) - lam * np.exp(-th2 / 2)); b = u * u - v * v
        out.append([b @ ref / np.sqrt((b @ b) * (ref @ ref)), np.sum((b - ref)**2) / (bs @ bs), 1 - np.arccos(np.clip(b @ bs / np.sqrt((b @ b) * (bs @ bs)), -1, 1)) / np.pi])
    return np.array(out)

def whp_grad(m, m0, mr, a):
    ind = (mr > m0).astype(float); t = m0 - a * np.maximum(mr - m0, 0); st = expit(t) * expit(-t)
    return expit(m) - expit(t), expit(m) * expit(-m), -st * (1 + a * ind), a * st * ind

def run(al, f, a, wr, n1, n2, T, dt, k, rho, N=3000, seed=0, retrain=False):
    rng = np.random.default_rng(seed); c = np.sqrt(k) / 2; Ns = int(round(al * N)); F = np.arange(Ns) < int(round(f * Ns))
    ref = None if retrain else run(al, f, a, wr, n1, n2, n1 + 1, dt, k, rho, N, seed, retrain=True)
    na = int(round(rho * N / 2)); bs = np.zeros(N); bs[:na] = rho**-0.5; bs[na:2 * na] = -rho**-0.5
    G1 = rng.standard_normal((Ns, N)) / np.sqrt(N); G2 = rng.standard_normal((N, Ns)) / np.sqrt(N); z = G1 @ bs; y = np.sign(z)
    th = np.zeros(N); beta = np.zeros((N, T)); D = np.zeros((N, T)); J = np.zeros((N, T, T)); Jn = np.zeros((N, T))
    g = np.zeros((Ns, T)); Tg = np.zeros((Ns, T, T)); Rb = np.zeros((T, T)); M = np.zeros(T); out = {key: np.zeros(T) for key in ("cos", "dist", "acc", "accF")}
    for n in range(T):
        if n == n1: th1, J1 = th.copy(), Jn.copy()
        if n == n2: tv = tv_metrics(c, th1, th, bs, ref)
        b = c * np.sinh(th); beta[:, n] = b; D[:, n] = np.sqrt(b * b + c * c); J[:, n, :] = Jn
        Rb[n, :n] = (D[:, n] @ Jn[:, :n]) / N; M[n] = b @ bs / N
        p = G1 @ b + g[:, :n] @ Rb[n, :n]; m = y * p
        if retrain and n == n1: return b
        if ref is not None: out["cos"][n] = b @ ref / np.sqrt((b @ b) * (ref @ ref) + 1e-300); out["dist"][n] = np.sum((b - ref)**2) / N; out["acc"][n] = 1 - np.arccos(np.clip(M[n] / np.sqrt(b @ b / N + 1e-300), -1, 1)) / np.pi; out["accF"][n] = np.mean(m[F] > 0)
        tp = np.einsum("l,ilm->im", Rb[n, :n], Tg[:, :n, :]); tp[:, n] += 1.0
        if n == n1: m0, tp0 = m.copy(), tp.copy()
        if n == n2: mr, tpr = m.copy(), tp.copy()
        c1, c2, _ = grad("ce", m, m)
        if n < n1: g1, g2, ex = (np.where(F, 0.0, c1), np.where(F, 0.0, c2), 0.0) if retrain else (c1, c2, 0.0)
        elif n < n2: g1, g2, ex = np.where(F, c1, 0.0), np.where(F, c2, 0.0), 0.0
        elif n == n2: g1, g2, ex = 0 * m, 0 * m, 0.0                                   # step n2 only records the reinforced model
        else:
            w1, w2, w3, w4 = whp_grad(m, m0, mr, a); g1, g2 = np.where(F, w1, wr * c1), np.where(F, w2, wr * c2); ex = np.where(F, w3, 0.0)[:, None] * tp0 + np.where(F, w4, 0.0)[:, None] * tpr
        g[:, n] = y * g1; Tg[:, n, :] = g2[:, None] * tp + ex
        Rgt = Tg[:, n, :n + 1].mean(0); mu = al * (np.mean(z * g[:, n]) - Rgt @ M[:n + 1])
        Gf = G2 @ g[:, n] + al * (beta[:, :n + 1] @ Rgt) + mu * bs
        dG = al * np.einsum("l,il,ilm->im", Rgt, D[:, :n + 1], J[:, :n + 1, :]); dG[:, n] += 1.0
        th = th - 4 * dt * Gf; Jn = Jn - 4 * dt * dG
        if n == n2: th, Jn = th1.copy(), J1.copy()                                      # reset to the pretrained weights
    out["tv"] = tv; return out

def simulate(al, f, a, wr, n1, n2, T, dt, k, rho, d=3000, seed=0):
    rng = np.random.default_rng(seed + 10**6); c = np.sqrt(k) / 2; n = int(al * d); F = np.arange(n) < int(round(f * n))
    bs = np.zeros(d); bs[:int(rho * d)] = rng.choice([-1, 1], int(rho * d)) * rho**-0.5; X = rng.standard_normal((d, n)) / np.sqrt(d); y = np.sign(X.T @ bs)
    th = np.zeros(d)
    for i in range(n1): th = th - 4 * dt * (X[:, ~F] @ (y[~F] * grad("ce", y[~F] * (X[:, ~F].T @ (c * np.sinh(th))), 0)[0]))
    ref = c * np.sinh(th); th = np.zeros(d); out = {key: np.zeros(T) for key in ("cos", "dist", "acc", "accF")}
    for i in range(T):
        if i == n1: th1 = th.copy()
        if i == n2: tv = tv_metrics(c, th1, th, bs, ref)
        b = c * np.sinh(th); m = y * (X.T @ b)
        out["cos"][i] = b @ ref / np.sqrt((b @ b) * (ref @ ref) + 1e-300); out["dist"][i] = np.sum((b - ref)**2) / d; out["acc"][i] = 1 - np.arccos(np.clip(b @ bs / np.sqrt((b @ b) * d + 1e-300), -1, 1)) / np.pi; out["accF"][i] = np.mean(m[F] > 0)
        if i == n1: m0 = m.copy()
        if i == n2: mr = m.copy()
        c1 = grad("ce", m, m)[0]
        g1 = c1 if i < n1 else np.where(F, c1, 0.0) if i < n2 else 0 * m if i == n2 else np.where(F, whp_grad(m, m0, mr, a)[0], wr * c1)
        th = th - 4 * dt * (X @ (y * g1))
        if i == n2: th = th1.copy()
    out["tv"] = tv; return out

def job(x): return (run if x[0] == "th" else simulate)(*x[1:])

if __name__ == "__main__":
    k, a, wr, nrun, T1 = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]); al, f, rho, dt, Tr, T3 = 0.6, 0.5, 0.2, 0.05, 1.0, 2.0
    n1 = int(round(T1 / dt)); n2 = n1 + int(round(Tr / dt)); T = n2 + int(round(T3 / dt))
    with Pool(2) as p: R = p.map(job, [(w, al, f, a, wr, n1, n2, T, dt, k, rho, 3000, s) for w in ("th", "si") for s in range(nrun)])
    TH, SI = R[:nrun], R[nrun:]; pickle.dump(dict(dt=dt, n1=n1, n2=n2, TH=TH, SI=SI, lams=LAMS), open(f"whp_{k:g}_{a:g}_{wr:g}.pkl", "wb"))
    ms = lambda V: (np.mean(V, 0), np.std(V, 0, ddof=1) / np.sqrt(len(V))); zz = lambda A, B: (ms(A)[0] - ms(B)[0]) / np.maximum(np.hypot(ms(A)[1], ms(B)[1]), 1e-12)
    A, B = np.array([o["tv"] for o in TH]), np.array([o["tv"] for o in SI]); Z = zz(A, B)
    print(f"k={k:g}. Task vector (reinforce time {Tr}): theory / sim (z)")
    for i, lam in enumerate(LAMS): print(f"  lam={lam:<4g} | cos to retrained {A[:, i, 0].mean():.4f}/{B[:, i, 0].mean():.4f} ({Z[i, 0]:+.1f}) | distance {A[:, i, 1].mean():.3f}/{B[:, i, 1].mean():.3f} ({Z[i, 1]:+.1f}) | test acc {A[:, i, 2].mean():.4f}/{B[:, i, 2].mean():.4f} ({Z[i, 2]:+.1f})")
    print(f"WHP (a={a:g}, retain weight {wr:g}): theory / sim (z)")
    for i in [n2 + int(round(x * (T - 1 - n2))) for x in (0, 0.25, 0.5, 0.75, 1.0)]:
        row = f"  t3={dt*(i-n2):4.2f} |"
        for key in ("cos", "dist", "acc", "accF"):
            a_, b_ = np.array([o[key][i] for o in TH]), np.array([o[key][i] for o in SI]); row += f" {key} {a_.mean():.4f}/{b_.mean():.4f} ({(a_.mean()-b_.mean())/max(np.hypot(a_.std(ddof=1), b_.std(ddof=1))/np.sqrt(nrun),1e-12):+.1f}) |"
        print(row, flush=True)
