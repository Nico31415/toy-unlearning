"""Rescaling identity: finetuning after rescaling by S = S * (finetune the unrescaled model to targets / S). Linearised version with metric R, against the simulations."""
import numpy as np
from mse_sim import make, BASE, metrics
from endpoint import proj
res = {}
for seed in range(3):
    cfg = {**BASE, "d": 2000}; D = make(cfg, seed); d = D["d"]; c = np.sqrt(cfg["k"]) / 2; nR = D["XR"].shape[1]; nFf = D["XF"].shape[1]; nF = nFf // 2
    Xa = np.zeros((d, nR + nFf)); Xa[D["rowsR"], :nR] = D["XR"]; Xa[D["rowsF"], nR:] = D["XF"]; bP = proj(Xa, np.r_[D["yR"], D["yF"]], c)[0]; X = Xa[:, :nR + nF]; R = np.sqrt(bP**2 + c * c)
    for S in (0.1, 0.3, 1.0, 3.0):
        t = np.r_[D["yR"], np.zeros(nF)] / S; b = S * (bP + R * (X @ np.linalg.solve(X.T @ (X * R[:, None]), t - X.T @ bP))); m = metrics(D, b[:, None], nF); res.setdefault(S, []).append([m[k][0] for k in ("m1", "m2", "m3", "m4")])
for S, v in res.items(): v = np.mean(v, 0); print(f"S={S}: linearised m1 {v[0]:.3f} m2 {v[1]:.3f} m3 {v[2]:.3f} m4 {v[3]:.3f}  m3/m2 {v[2] / v[1]:.2f}")
print("simulation (d=3000): m3/m2 = 0.67, 0.56, 0.40, 0.35;  m3 = 0.371, 0.315, 0.231, 0.205;  m1 = 0.451, 0.441, 0.430, 0.424")
