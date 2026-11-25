"""Exact interpolation end point from zero at scale c: beta = c sinh(X lam) with X^T beta = y (Newton on the convex dual)."""
import numpy as np
def proj(X, y, c, iters=500, tol=1e-10):
    lam = np.zeros(X.shape[1]); dual = lambda l: np.sum(c * np.cosh(X @ l)) - l @ y
    for it in range(iters):
        th = X @ lam; g = X.T @ (c * np.sinh(th)) - y
        if np.linalg.norm(g) < tol * (1 + np.linalg.norm(y)): break
        H = X.T @ (X * (c * np.cosh(th))[:, None]); st = np.linalg.solve(H + 1e-12 * np.eye(len(lam)), g); s = 1.0; D0 = dual(lam)
        with np.errstate(over="ignore", invalid="ignore"):
            while s > 1e-14 and not (dual(lam - s * st) <= D0 - 0.25 * s * (g @ st)): s /= 2
        lam = lam - s * st
    return c * np.sinh(X @ lam), it
if __name__ == "__main__":
    import sys; from mse_sim import make, BASE, metrics
    for seed in range(int(sys.argv[1])):
        cfg = {**BASE, "d": int(sys.argv[2])}; D = make(cfg, seed); d = D["d"]; e = D["e"]; c = np.sqrt(cfg["k"]) / 2; nFf = D["XF"].shape[1]; nF = nFf // 2
        def full(useF):
            X = np.zeros((d, D["XR"].shape[1] + int(useF.sum()))); X[D["rowsR"], :D["XR"].shape[1]] = D["XR"]; X[D["rowsF"], D["XR"].shape[1]:] = D["XF"][:, useF]; return X, np.r_[D["yR"], D["yF"][useF]]
        a = np.ones(nFf, bool); k_ = a.copy(); k_[:nF] = False
        for nm, u in (("pre", a), ("untrain", k_), ("unlearn", np.zeros(nFf, bool))):
            X, y = full(u); b, it = proj(X, y, c); m = metrics(D, b[:, None], nF); print(seed, nm, it, {k: round(float(v[0]), 4) for k, v in m.items()}, flush=True)
