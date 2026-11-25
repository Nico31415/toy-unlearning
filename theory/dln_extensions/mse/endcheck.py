"""End points of finetuning from the static two-stage equations (multi.py), against the simulations at t = 40."""
import numpy as np, sys
sys.path.insert(0, "/home/claude/out/final")
from multi import solve, dist
c = np.sqrt(1e-5) / 2
for name, q in (("C (1,0.4,1)", (1, 0.4, 1)), ("B (1,0,1)", (1, 0, 1))):
    q = np.array(q, float) / sum(q); bl = [(q[0], 0.2), (q[1], 0.2), (q[2], 0.2)]; bl2 = [b for b in bl if b[0] > 0]; idx = [i for i, b in enumerate(bl) if b[0] > 0]; mp = {i: j for j, i in enumerate(idx)}
    R = [mp[i] for i in (0, 1) if i in mp]; F = [mp[i] for i in (1, 2) if i in mp]; aR = 0.3 * (q[0] + q[1]); aF = 0.3 * (q[1] + q[2])
    gr = [(aR, R), (aF / 2, F), (aF / 2, F)]   # 0 retain, 1 forget F, 2 left-over
    sy = [dict(parent=None, c=c, use={0: 1, 1: 1, 2: 1}), dict(parent=None, c=c, use={0: 1, 2: 1}), dict(parent=None, c=c, use={0: 1}),
          dict(parent=1, c=c, use={0: 1, 1: 0.0}), dict(parent=1, c=c, use={0: 1, 1: -0.5}), dict(parent=1, c=0.3 * c, use={0: 1, 1: 0.0}), dict(parent=1, c=3 * c, use={0: 1, 1: 0.0})]
    out = []
    for seed in range(3):
        s = solve(bl2, gr, sy, N=60000, iters=1500, seed=seed); out.append([dist(s, bl2, a, b) * 1.0 for a in (4, 5, 6, 7) for b in (2, 3)] + [dist(s, bl2, 1, 2), dist(s, bl2, 1, 3)])
    o = np.mean(out, 0); print(name, "theory: f2z (untr, unl) %.3f %.3f | neg-0.5 %.3f %.3f | S=0.3 %.3f %.3f | S=3 %.3f %.3f | pretrained %.3f %.3f" % tuple(o), flush=True)
print("simulation t=40, C: f2z 0.140 0.096 | neg-0.5 0.260 0.249 | S=0.3 0.159 0.071 | S=3 0.152 0.193 | pretrained 0.207 0.290")
print("simulation t=40, B: f2z 0.113 0.012 | S=0.3 0.135 0.002 | S=3 0.107 0.096 | pretrained 0.184 0.222")
