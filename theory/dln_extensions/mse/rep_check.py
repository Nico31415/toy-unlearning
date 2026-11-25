"""Direct solution of the two-stage replica equations (Result 15: one block, stage 2 on a reused subset, target b*y), against the cavity solver multi.py and a finite-d projection."""
import numpy as np, sys
sys.path.insert(0, "/home/claude/out/final")
from multi import solve, dist
def two_stage(al, alF, b, S, c, rho=0.2, N=400000, iters=600, damp=0.3, seed=0):
    rng = np.random.default_rng(seed); bs = np.where(rng.random(N) < rho, rng.choice([-1, 1], N) * rho**-0.5, 0.0); h = rng.standard_normal((N, 2))
    chi1, chi2, rr = 1.0, 1.0, 0.0; v1, v2, cv = 1.0, 1.0, 0.0; t1 = np.zeros(N); t2 = np.zeros(N)
    for it in range(iters):
        A1 = al / chi1; A2 = alF / chi2; L = np.linalg.cholesky(np.array([[v1, cv], [cv, v2]]) + 1e-12 * np.eye(2)); e1 = L[0, 0] * h[:, 0]; e2 = L[1, 0] * h[:, 0] + L[1, 1] * h[:, 1]
        for _ in range(60): f = t1 + A1 * (c * np.sinh(t1) - bs) - e1; t1 = t1 - f / (1 + A1 * c * np.cosh(t1))          # asinh(b1/c) + A1 u = eta1
        b1 = c * np.sinh(t1); u = b1 - bs; D1 = np.sqrt(b1**2 + c * c); db1 = D1 / (1 + A1 * D1)                           # d beta1 / d eta1
        for _ in range(60): f = t2 - t1 + A2 * (S * c * np.sinh(t2) - b * bs - rr * u) - e2; t2 = t2 - f / (1 + A2 * S * c * np.cosh(t2))   # asinh(b2/Sc) - asinh(b1/c) + A2 (w - rho u) = eta2
        b2 = S * c * np.sinh(t2); w = b2 - b * bs; D2 = np.sqrt(b2**2 + (S * c)**2); db2 = D2 / (1 + A2 * D2)              # d beta2 / d eta2
        db21 = db2 * (1 / D1 + A2 * rr) * db1                                                                             # d beta2 / d eta1 through beta1
        n1, n2, n12, nr = db1.mean(), db2.mean(), al * np.mean(u * u) / chi1**2, db21.mean() / db1.mean(); e = w - rr * u
        chi1 += damp * (n1 - chi1); chi2 += damp * (n2 - chi2); rr += damp * (nr - rr); v1 += damp * (n12 - v1); v2 += damp * (alF * np.mean(e * e) / chi2**2 - v2); cv += damp * (alF * np.mean(u * e) / (chi1 * chi2) - cv)
    return dict(Q2=np.mean(b2**2), err2=np.mean((b2 - bs)**2), d12=np.mean((b2 - b1)**2), Q1=np.mean(b1**2), rho=rr, left=np.mean(((b2 - bs) - rr * u)**2))
for k, al, alF, b, S in ((1e-2, 0.4, 0.2, 0.0, 1.0), (1e-5, 0.3, 0.15, 0.0, 1.0), (1e-2, 0.4, 0.2, -0.5, 1.0), (1e-2, 0.4, 0.2, 0.0, 3.0)):
    c = np.sqrt(k) / 2; r = two_stage(al, alF, b, S, c); s = solve([(1.0, 0.2)], [(alF, [0]), (al - alF, [0])], [dict(parent=None, c=c, use={0: 1, 1: 1}), dict(parent=1, c=S * c, use={0: b})], N=100000, iters=1500)
    print(f"k={k} alpha={al} alpha_F={alF} b={b} S={S}:  Result 15: |b2|^2 {r['Q2']:.4f} |b2-b*|^2 {r['err2']:.4f} |b2-b1|^2 {r['d12']:.4f} (rho {r['rho']:.3f}, left-over error {r['left']:.4f})   cavity solver: {dist(s, [(1.0, 0.2)], 2, None):.4f} {dist(s, [(1.0, 0.2)], 2, 0):.4f} {dist(s, [(1.0, 0.2)], 2, 1):.4f}", flush=True)
