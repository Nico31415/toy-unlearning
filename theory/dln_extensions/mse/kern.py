"""What decides how forgetting spreads: the metric R_j = (u_j^2 + v_j^2) = sqrt(beta_j^2 + c^2) at the pretrained model (the tangent kernel).
Linearised finetuning: beta = beta_pre + R * (X lam), with X_R^T beta = y_R and X_F^T beta = 0. Compared with the flat metric (l2) and with the participation ratio."""
import numpy as np
from mse_sim import make, BASE, metrics
from endpoint import proj
for ff in (0.1, 0.5, 0.9):
    out = []
    for seed in range(3):
        cfg = {**BASE, "d": 2000, "ffrac": ff}; D = make(cfg, seed); d = D["d"]; c = np.sqrt(cfg["k"]) / 2; nR = D["XR"].shape[1]; nFf = D["XF"].shape[1]; nF = int(round(ff * nFf)); e = D["e"]
        Xa = np.zeros((d, nR + nFf)); Xa[D["rowsR"], :nR] = D["XR"]; Xa[D["rowsF"], nR:] = D["XF"]; bP = proj(Xa, np.r_[D["yR"], D["yF"]], c)[0]
        X = Xa[:, :nR + nF]; t = np.r_[D["yR"], np.zeros(nF)]; R = np.sqrt(bP**2 + c * c); row = []
        for M in (R, np.ones(d)):
            b = bP + M * (X @ np.linalg.solve(X.T @ (X * M[:, None]), t - X.T @ bP)); m = metrics(D, b[:, None], nF); row += [m["m3"][0] / m["m2"][0]]
        Rf = R[e[1]:e[3]]; deff = Rf.sum()**2 / (Rf**2).sum(); row += [nF / deff, deff / (e[3] - e[1]), np.mean(np.abs(bP[e[1]:e[3]]) > 1e-3), nF / (e[3] - e[1])]; out.append(row)
    o = np.mean(out, 0); print(f"|F|/|F_full| = {ff}: left-over damage m3/m2:  linearised with metric R {o[0]:.2f} | flat metric {o[1]:.2f} | n_F / d_eff {o[2]:.2f}   (d_eff / d_class = {o[3]:.2f}, active fraction {o[4]:.2f}, n_F / d_class = {o[5]:.3f})", flush=True)
