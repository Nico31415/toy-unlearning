"""Replica / cavity state evolution for full-batch training with an explicit separable regulariser:
  U = argmin_beta  1/2 sum_a w_a sum_{i in a} (x_i^T beta - a_a x_i^T beta*)^2 + lam sum_j R(beta_j),   x_i ~ N(0, I/d), n_a = delta p_a d.
Weight decay on the weights of a diagonal network (beta = u^2 - v^2, penalty (lam_w/2)(|u|^2+|v|^2)) is R = |.| with lam = lam_w/2.
Scalar channel: beta = prox_{lam R / A}((B beta* + sqrt(C) h)/A),  with
  A = delta sum_a p_a w_a/(1+w_a V),  B = delta sum_a p_a w_a a_a/(1+w_a V),  C = delta sum_a p_a w_a^2 (Q - 2 a_a M + a_a^2)/(1+w_a V)^2,
  V = E[prox']/A,  Q = E beta^2,  M = E beta beta*.   Train residual of group a: (Q - 2 a_a M + a_a^2)/(1+w_a V)^2."""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in ("core", "ce", "wd", "sgd"):
    _sys.path.insert(0, _os.path.join(_ROOT, _p))
import numpy as np
H = np.linspace(-9, 9, 6001); WH = np.exp(-H**2/2); WH /= WH.sum()

def prox_l1(x, t): return np.sign(x) * np.maximum(np.abs(x) - t, 0), (np.abs(x) > t).astype(float)
def prox_l2(x, t): return x / (1 + t), np.full_like(x, 1 / (1 + t))          # R = b^2/2

def solve(delta, groups, lam, rho, prox=prox_l1, iters=5000, damp=0.5):
    """groups: list of (p_a, w_a, a_a). Teacher: +-rho^-1/2 w.p. rho, else 0."""
    amp = rho ** -0.5; P, W, Aa = map(np.array, zip(*groups))
    Q, M, V = 0.5, 0.5, 1.0
    for it in range(iters):
        den = 1 + W * V
        A = delta * np.sum(P * W / den); B = delta * np.sum(P * W * Aa / den)
        C = delta * np.sum(P * W**2 * (Q - 2 * Aa * M + Aa**2) / den**2)
        b0, d0 = prox(np.sqrt(C) * H / A, lam / A); b1, d1 = prox((B * amp + np.sqrt(C) * H) / A, lam / A)
        Qn = (1 - rho) * WH @ b0**2 + rho * WH @ b1**2; Mn = rho * amp * (WH @ b1)
        Vn = ((1 - rho) * WH @ d0 + rho * WH @ d1) / A
        err = abs(Qn - Q) + abs(Mn - M) + abs(Vn - V)
        Q, M, V = damp * Q + (1 - damp) * Qn, damp * M + (1 - damp) * Mn, damp * V + (1 - damp) * Vn
        if err < 1e-12: break
    den = 1 + W * V
    return dict(Q=Q, M=M, V=V, res=(Q - 2 * Aa * M + Aa**2) / den**2, err=err)

def lasso(X, t, w, lam, iters=20000):
    """FISTA for 1/2 sum_i w_i (x_i^T b - t_i)^2 + lam |b|_1.  X: d x n."""
    L = np.linalg.norm(X * np.sqrt(w), 2) ** 2; b = np.zeros(X.shape[0]); yk = b.copy(); tk = 1.0
    for _ in range(iters):
        g = X @ (w * (X.T @ yk - t)); bn = yk - g / L; bn = np.sign(bn) * np.maximum(np.abs(bn) - lam / L, 0)
        tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2; yk = bn + (tk - 1) / tn * (bn - b)
        if np.max(np.abs(bn - b)) < 1e-11: b = bn; break
        b, tk = bn, tn
    return b

if __name__ == "__main__":
    rho, d, NS = 0.2, 1500, 4
    def sim(delta, groups, lam, seed, ridge=False):
        rng = np.random.default_rng(seed); bs = np.where(rng.random(d) < rho, rng.choice([-1, 1], d) * rho**-0.5, 0.0)
        n = int(round(delta * d)); X = rng.standard_normal((d, n)) / np.sqrt(d); z = X.T @ bs
        sz = np.round(np.cumsum([0] + [g[0] for g in groups]) * n).astype(int); w = np.zeros(n); t = np.zeros(n)
        for gi, (p, wa, aa) in enumerate(groups): w[sz[gi]:sz[gi+1]] = wa; t[sz[gi]:sz[gi+1]] = aa * z[sz[gi]:sz[gi+1]]
        if ridge: b = np.linalg.solve((X * w) @ X.T + lam * np.eye(d), X @ (w * t))
        else: b = lasso(X, t, w, lam)
        r = X.T @ b - t
        return [b @ b / (bs @ bs), b @ bs / (bs @ bs)] + [np.mean(r[sz[gi]:sz[gi+1]]**2) / np.mean(z**2) for gi in range(len(groups))]
    print("A. pretraining with weight decay (lasso), alpha_PT = 0.3:  lam | |A|^2 th/sim | overlap th/sim | train residual th/sim")
    for lam in (0.001, 0.01, 0.05, 0.2):
        th = solve(0.3, [(1.0, 1.0, 1.0)], lam, rho); S = np.array([sim(0.3, [(1.0, 1.0, 1.0)], lam, s) for s in range(NS)]); m, e = S.mean(0), S.std(0, ddof=1) / 2
        print(f"  {lam:5.3f} | {th['Q']:.4f} / {m[0]:.4f}±{e[0]:.4f} | {th['M']:.4f} / {m[1]:.4f}±{e[1]:.4f} | {th['res'][0]:.4f} / {m[2]:.4f}±{e[2]:.4f}")
    print("B. stage 2 with weight decay: deleted samples (alpha_UL = 0.2) -> target 0, kept samples (0.1) anchored on y:  lam | |U|^2 | overlap | deleted residual | kept residual")
    G = [(2/3, 1.0, 0.0), (1/3, 1.0, 1.0)]
    for lam in (0.001, 0.01, 0.05):
        th = solve(0.3, G, lam, rho); S = np.array([sim(0.3, G, lam, s) for s in range(NS)]); m, e = S.mean(0), S.std(0, ddof=1) / 2
        print(f"  {lam:5.3f} | {th['Q']:.4f} / {m[0]:.4f}±{e[0]:.4f} | {th['M']:.4f} / {m[1]:.4f}±{e[1]:.4f} | {th['res'][0]:.4f} / {m[2]:.4f} | {th['res'][1]:.4f} / {m[3]:.4f}")
    print("C. another regulariser, ridge on beta (R = b^2/2), same two-group problem:  lam | |U|^2 | overlap | deleted residual | kept residual")
    for lam in (0.01, 0.1):
        th = solve(0.3, G, lam, rho, prox=prox_l2); S = np.array([sim(0.3, G, lam, s, ridge=True) for s in range(NS)]); m, e = S.mean(0), S.std(0, ddof=1) / 2
        print(f"  {lam:5.3f} | {th['Q']:.4f} / {m[0]:.4f}±{e[0]:.4f} | {th['M']:.4f} / {m[1]:.4f}±{e[1]:.4f} | {th['res'][0]:.4f} / {m[2]:.4f} | {th['res'][1]:.4f} / {m[3]:.4f}")
    # D. does gradient descent on the diagonal network with weight decay reach the lasso, whatever the start and k?
    d2 = 150; rng = np.random.default_rng(0); bs = np.where(rng.random(d2) < rho, rng.choice([-1, 1], d2) * rho**-0.5, 0.0)
    X = rng.standard_normal((d2, 45)) / np.sqrt(d2); t = X.T @ bs; lam = 0.02; bl = lasso(X, t, np.ones(45), lam)
    print("D. diagonal network + weight decay lam_w = 2 lam, gradient descent to convergence vs the lasso:")
    for k, start in ((1e-4, "zero"), (1.0, "zero"), (1e-2, "random")):
        a = (k / 16) ** 0.25 if True else 0; u = np.full(d2, np.sqrt(np.sqrt(k) / 4 + 1e-12)); v = u.copy()
        if start == "random": u = np.abs(rng.standard_normal(d2)) * 0.7; v = np.abs(rng.standard_normal(d2)) * 0.7
        lr = 0.02
        for it in range(400000):
            g = X @ (X.T @ (u*u - v*v) - t)
            u, v = u - lr * (2 * u * g + 2 * lam * u), v - lr * (-2 * v * g + 2 * lam * v)
        b = u*u - v*v
        print(f"   k={k:g}, start={start}: |beta_GD - beta_lasso| / |beta_lasso| = {np.linalg.norm(b - bl) / np.linalg.norm(bl):.2e}")
