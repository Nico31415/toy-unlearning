"""Runner: every configuration x data setting, simulation (5 seeds, d = 3000) and theory (3 seeds). Resumable: one file per job in results/."""
import numpy as np, pickle, os, sys, time, traceback
from multiprocessing import Pool
DATA = {"C": (1, 0.4, 1), "A": (0, 1, 0), "B": (1, 0, 1), "D": (1, 1, 1)}
CONFIGS = {
    "base": {},
    "alpha=0.8": dict(alpha=0.8, d=2000, t1=20.0, t2=20.0), "alpha=1.5": dict(alpha=1.5, d=2000, t1=20.0, t2=20.0),
    "ffrac=0.1": dict(ffrac=0.1), "ffrac=0.9": dict(ffrac=0.9), "ffrac=1": dict(ffrac=1.0),
    "noise_pt=0.1": dict(sig2_pt=0.1), "noise_pt=0.5": dict(sig2_pt=0.5),
    "fresh": dict(fresh=True), "fresh,noise_ft=0.1": dict(fresh=True, sig2_ft=0.1), "fresh,noise_ft=0.5": dict(fresh=True, sig2_ft=0.5),
    "S=0.1": dict(S=0.1), "S=0.3": dict(S=0.3), "S=3": dict(S=3.0),
    "mu=(1,0.25)": dict(mu=(1.0, 0.25), mu_signed=(1.0, 0.25)), "mu=(1,4)": dict(mu=(1.0, 4.0), mu_signed=(1.0, 4.0)),
    "R=quarter": dict(rfrac=0.25),
    "sgd_ft": dict(phi_ft=0.1), "sgd_pt": dict(phi_pt=0.1),
    "decay_ft=0.1": dict(lam_ft=0.1, t2=120.0), "decay_ft=0.3": dict(lam_ft=0.3),
    "decay_pt=0.1": dict(lam_pt=0.1, t1=120.0), "decay_pt=0.3": dict(lam_pt=0.3),
    "decay_both=0.1": dict(lam_pt=0.1, t1=120.0, lam_ft=0.1, t2=120.0), "decay_both=0.3": dict(lam_pt=0.3, lam_ft=0.3),
    "alpha=0.8,fresh": dict(alpha=0.8, fresh=True, d=2000, t1=20.0, t2=20.0), "alpha=1.5,fresh": dict(alpha=1.5, fresh=True, d=2000, t1=20.0, t2=20.0),
}
for c in CONFIGS.values(): c.setdefault("mu_signed", (1.0, 0.5))
NO_THEORY = ("sgd_pt",)   # SGD pretraining is not a static end point: no finetuning theory from the static slot
def th_grid(cfg):  # (dt, T) for the theory: smaller steps where the curvature is larger
    f = max(cfg.get("S", 1.0), 1.0) * (2.0 if cfg.get("alpha", 0.3) >= 0.8 else 1.0) * (2.0 if cfg.get("mu", (1, 1))[1] > 1 else 1.0)
    if cfg.get("phi_ft"): return (0.01, 120)   # SGD: the theory must use the simulation's step size
    return (0.05, 120) if f == 1 else (0.05 / f, 100)
def job(a):
    cn, dn, kind, seed = a; f = f"results/{cn}__{dn}__{kind}{seed}.pkl"
    if os.path.exists(f): return
    cfg = {**CONFIGS[cn], "q": DATA[dn]}; t = time.time()
    try:
        if kind == "sim":
            from mse_sim import pipeline; r = pipeline(cfg, seed, cache="cache")
        elif kind == "pth":   # theory for the pretraining trajectory
            from mse_theory import pretrain_theory; r = pretrain_theory(cfg, seed + 2000, N=3000, T=200, dt=0.025 if cfg.get("alpha", 0.3) >= 0.8 else 0.05)
        else:
            from mse_theory import theory; dt, T = th_grid(cfg); r = theory(cfg, seed + 1000, N=3000, T=T, dt=dt); r["dt"] = dt
        r["seconds"] = time.time() - t; pickle.dump(r, open(f + ".tmp", "wb")); os.replace(f + ".tmp", f)
    except Exception: open("results/errors.log", "a").write(f"{a}\n{traceback.format_exc()}\n")
    print(time.strftime("%H:%M:%S"), cn, dn, kind, seed, f"{time.time() - t:.0f}s", flush=True)
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True); os.makedirs("cache", exist_ok=True); sel = sys.argv[1].split(";") if len(sys.argv) > 1 else list(CONFIGS); jobs = []
    for cn in sel:
        for dn in DATA:
            jobs += [(cn, dn, "sim", s) for s in range(5)]
            if not cn.startswith(NO_THEORY): jobs += [(cn, dn, "th", s) for s in range(3)]
    PRE = ["base", "alpha=0.8", "alpha=1.5", "noise_pt=0.1", "noise_pt=0.5", "decay_pt=0.1", "decay_pt=0.3"]
    jobs = [(cn, dn, "pth", s) for cn in PRE if cn in sel and cn == "base" for dn in DATA for s in range(2)] + jobs + [(cn, dn, "pth", s) for cn in PRE if cn in sel and cn != "base" for dn in DATA for s in range(2)]
    jobs = [j for j in jobs if not j[0].startswith("alpha")] + [j for j in jobs if j[0].startswith("alpha")]
    with Pool(2) as p: list(p.imap_unordered(job, jobs, chunksize=1))
    print("ALL_DONE", flush=True)
