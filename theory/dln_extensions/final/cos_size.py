"""Class labels, small sample ratios: is the 0.6% cosine offset between theory and simulation a finite-size effect? Cosine to the retrained model before stage 2 and 20 steps in."""
import os, sys, numpy as np; os.environ["GRSMALL"] = "1"
from multiprocessing import Pool
import dmft_blocks2 as m
name = "labels: NPO + retain term on all kept"; n1, T, dt, k = 60, 81, 0.05, 1.0
def job(a):
    kind, size, seed = a; fn = m.run if kind == "th" else m.simulate; sz = dict(N=size) if kind == "th" else dict(d=size); pk, s2 = m.EXP[name]
    rK = fn(m.BL, m.GR, lambda st: [("ce", 1, 1), ("off", 0, 1), ("ce", 1, 1)], n1, n1 + 1, dt, k, seed=seed + 300, refs=None, **sz)
    o = fn(m.BL, m.GR, lambda st: s2 if st else [(pk, 1, 1)] * 3, n1, T, dt, k, seed=seed + 300, refs=(rK,), **sz); return o["cos0"][[n1, n1 + 20]]
if __name__ == "__main__":
    for kind, size, n in (("si", 1500, 12), ("si", 3000, 12), ("si", 6000, 8), ("si", 12000, 6), ("th", 4000, 8), ("th", 8000, 8)):
        with Pool(2) as p: v = np.array(p.map(job, [(kind, size, s) for s in range(n)]))
        print(f"{'simulation d' if kind == 'si' else 'theory N'} = {size:5d}: " + "  ".join(f"{a:.4f} ± {b:.4f}" for a, b in zip(v.mean(0), v.std(0, ddof=1) / np.sqrt(n))), flush=True)
