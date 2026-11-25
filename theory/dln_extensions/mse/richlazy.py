"""Forget-to-zero from the same rich pretrained model: rich finetuning (simulation, k = 1e-5) against lazy finetuning (smallest l2 change), for two forget fractions."""
import numpy as np
from mse_sim import make, BASE, metrics, pipeline, LOSSES
K = ("m1", "m2", "m3", "m4")
for ff in (0.1, 0.5):
    R, L, P = [], [], []
    for seed in range(3):
        cfg = {**BASE, "d": 1500, "ffrac": ff}; r = pipeline(cfg, seed); R.append([r["ft"][k][-1, 1] for k in K]); P.append([r["pre_end"][k] for k in K])
        # lazy: recover the pretrained beta by rerunning is costly; use the pipeline's reference distances instead -> recompute pretrained beta
        D = make(cfg, seed); from mse_sim import pretrain, descend
        grad, u, v = pretrain(D, cfg, seed, np.ones(D["XF"].shape[1], bool), 0); u, v, _ = descend(D, u, v, grad, 4000, 0.01, 0.0, 4000, 1); bP = (u * u - v * v)[:, 0]
        nR = D["XR"].shape[1]; nF = int(round(ff * D["XF"].shape[1])); X = np.zeros((D["d"], nR + nF)); X[D["rowsR"], :nR] = D["XR"]; X[D["rowsF"], nR:] = D["XF"][:, :nF]; t = np.r_[D["yR"], np.zeros(nF)]
        bl = bP + X @ np.linalg.solve(X.T @ X, t - X.T @ bP); m = metrics(D, bl[:, None], nF); L.append([m[k][0] for k in K])
    P, R, L = (np.mean(x, 0) for x in (P, R, L)); print(f"|F|/|F_full| = {ff}:  pretrained {P.round(3)}   rich {R.round(3)}   lazy {L.round(3)}")
    print(f"     share of the forgetting that generalises, (m4 - m4_pre)/m2:  rich {(R[3] - P[3]) / R[1]:.2f}  lazy {(L[3] - P[3]) / L[1]:.2f};   left-over damage m3/m2: rich {R[2] / R[1]:.2f}  lazy {L[2] / L[1]:.2f}", flush=True)
