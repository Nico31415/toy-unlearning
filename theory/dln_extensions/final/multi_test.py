import sys, numpy as np; sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "ce"))
from ce_tests import project
from multi import solve, dist
def sim(blocks, groups, systems, d, seed, noise=None):
    rng = np.random.default_rng(seed); edges = np.r_[0, np.cumsum([int(round(q * d)) for q, _ in blocks])]; bs = np.zeros(d)
    for B, (q, rho) in enumerate(blocks): m = int(rho * (edges[B + 1] - edges[B])); bs[edges[B]:edges[B] + m] = rng.choice([-1, 1], m) * rho**-0.5
    X = []
    for al, sup in groups:
        x = np.zeros((d, int(round(al * d))))
        for B in sup: x[edges[B]:edges[B + 1]] = rng.standard_normal((edges[B + 1] - edges[B], x.shape[1])) / np.sqrt(d)
        X.append(x)
    lab = [x.T @ bs + (0.0 if noise is None else noise[a] * rng.standard_normal(x.shape[1])) for a, x in enumerate(X)]; beta = [bs]
    for sy in systems:
        Xu = np.hstack([X[a] for a in sy["use"]]); t = np.concatenate([b * lab[a] for a, b in sy["use"].items()]); p = sy["parent"]
        b0 = np.zeros(d) if p is None else beta[p] * sy["c"] / systems[p - 1]["c"]; beta.append(project(b0, sy["c"], A_eq=Xu, b_eq=t)[0])
    return np.array(beta) @ np.array(beta).T / (bs @ bs)
def compare(name, blocks, groups, systems, pairs, d=1500, ns=8, **kw):
    sol = solve(blocks, groups, systems, **kw); G = np.array([sim(blocks, groups, systems, d, s, kw.get("noise")) for s in range(ns)])
    print(f"{name}: fixed point err {sol['err']:.1e} after {sol['it']} iterations")
    for lab, s, m in pairs:
        th = dist(sol, blocks, s, m); v = G[:, s, s] - (2 * G[:, s, m] - G[:, m, m] if m is not None else 0); print(f"   {lab:42s} theory {th:.4f} | sim {v.mean():.4f} ± {v.std(ddof=1)/np.sqrt(ns):.4f}", flush=True)
if __name__ == "__main__":
    for k in (1.0, 1e-2):
        c = np.sqrt(k) / 2
        compare(f"forget coordinates only, k={k:g}", [(1.0, 0.2)], [(0.2, [0]), (0.1, [0])],
                [dict(parent=None, c=c, use={0: 1, 1: 1}), dict(parent=1, c=c, use={0: 0.0}), dict(parent=None, c=c, use={1: 1})],
                [("pretrained to teacher", 1, 0), ("unlearned norm (dist to unlearning ref)", 2, None), ("unlearned to untraining ref", 2, 3), ("do nothing to untraining ref", 1, 3)])
        compare(f"blocks R/Sh/F with retain anchor, k={k:g}", [(0.4, 0.2), (0.2, 0.2), (0.4, 0.2)], [(0.3, [0, 1]), (0.12, [1, 2]), (0.08, [1, 2])],
                [dict(parent=None, c=c, use={0: 1, 1: 1, 2: 1}), dict(parent=1, c=c, use={1: 0.0, 0: 1}), dict(parent=None, c=c, use={0: 1, 2: 1}), dict(parent=None, c=c, use={0: 1}), dict(parent=1, c=0.3 * c, use={1: 0.0})],
                [("pretrained to teacher", 1, 0), ("unlearned (anchor) to untraining ref", 2, 3), ("unlearned (anchor) to unlearning ref", 2, 4), ("do nothing to untraining ref", 1, 3), ("do nothing to unlearning ref", 1, 4), ("kappa=0.3, no anchor, to unlearning ref", 5, 4)])
