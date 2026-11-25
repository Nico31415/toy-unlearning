"""Lazy (linear, l2) finetuning from the exact rich pretrained model: smallest l2 change with X_R^T beta = y_R and X_F^T beta = b y_F. Compared with the rich simulation."""
import numpy as np
from mse_sim import make, BASE, metrics
from endpoint import proj
out = {}
for seed in range(3):
    cfg = {**BASE, "d": 2000}; D = make(cfg, seed); d = D["d"]; c = np.sqrt(cfg["k"]) / 2; nR = D["XR"].shape[1]; nFf = D["XF"].shape[1]; nF = nFf // 2
    def full(useF):
        X = np.zeros((d, nR + int(useF.sum()))); X[D["rowsR"], :nR] = D["XR"]; X[D["rowsF"], nR:] = D["XF"][:, useF]; return X, np.r_[D["yR"], D["yF"][useF]]
    a = np.ones(nFf, bool); k_ = a.copy(); k_[:nF] = False; bP = proj(*full(a), c)[0]; bU = proj(*full(k_), c)[0]; bL = proj(*full(np.zeros(nFf, bool)), c)[0]
    f_ = np.zeros(nFf, bool); f_[:nF] = True; X, y = full(f_)
    for b in (0.0, -0.5, None):
        t = y.copy()
        if b is None: t[nR:] = X[:, nR:].T @ bU      # oracle target: the retrained model's own predictions on F
        else: t[nR:] = b * y[nR:]
        bl = bP + X @ np.linalg.solve(X.T @ X, t - X.T @ bP); m = metrics(D, bl[:, None], nF)
        out.setdefault(b, []).append([m[k][0] for k in ("m1", "m2", "m3", "m4")] + [np.sum((bl - bU)**2) / d, np.sum((bl - bL)**2) / d])
    out.setdefault("pre", []).append([metrics(D, bP[:, None], nF)[k][0] for k in ("m1", "m2", "m3", "m4")] + [np.sum((bP - bU)**2) / d, np.sum((bP - bL)**2) / d]); out.setdefault("untrain ref", []).append([metrics(D, bU[:, None], nF)[k][0] for k in ("m1", "m2", "m3", "m4")] + [0, 0])
print("                               m1     m2     m3     m4   d_untrain d_unlearn")
for k, v in out.items(): print(f"{str(k):28s}", np.mean(v, 0).round(3))
