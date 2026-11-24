"""Large-d end points of several coupled trainings on the same data (static cavity), with blocks of coordinates and groups of samples.
System s is a gradient flow to convergence that starts from system parent[s] (or from zero), with scale c_s, on the groups it uses, to targets b * (teacher logit).
KKT: theta_s = theta_parent - X g_s,  X_used^T beta_s = targets,  beta_s = c_s sinh(theta_s).
Sample of group a:   p_s = zeta_s + sum_{m in ancestors-or-self} V_a[s,m] g_m   (zeta_0 = teacher logit; (zeta_0..zeta_S) Gaussian with covariance C_a)
Coordinate of block B: theta_s = theta_parent - [ eta_s + sum_m a_B[s,m] beta_m ],  beta_0 = beta*,  eta ~ N(0, sum_a alpha_a E g g^T),  a_B[s,m] = sum_a alpha_a d g_s / d zeta_m
V_a[s,m] = sum_{B in a} q_B E d beta_s / d eta_m,   C_a[s,m] = sum_{B in a} q_B E beta_s beta_m."""
import numpy as np

def anc(parent, s):
    out = []
    while parent[s] is not None: s = parent[s]; out.append(s)
    return out

def solve(blocks, groups, systems, N=40000, iters=400, damp=0.5, seed=0, tol=1e-7, verbose=False, noise=None):
    """blocks: [(q, rho)], groups: [(alpha, [block ids])], systems: [dict(parent=None|index (1-based), c=..., use={group id: b})] (system ids 1..S).
    noise: per-group standard deviation of label noise (labels = teacher logit + noise; targets are b * label). The noise is one more Gaussian input, index S+1."""
    S = len(systems); nB, nG = len(blocks), len(groups); rng = np.random.default_rng(seed); parent = [None] + [sy["parent"] for sy in systems]
    eps = rng.standard_normal((nB, N, S)); bst = np.zeros((nB, N))
    for B, (q, rho) in enumerate(blocks): m = int(round(rho * N / 2)); bst[B, :m] = rho**-0.5; bst[B, m:2 * m] = -rho**-0.5
    sig = [0.0] * nG if noise is None else list(noise)
    def full(Ca, a): Cf = np.zeros((S + 2, S + 2)); Cf[:S + 1, :S + 1] = Ca; Cf[S + 1, S + 1] = sig[a]**2; return Cf
    C = np.array([np.eye(S + 1) * sum(blocks[B][0] for B in sup) for al, sup in groups]); V = np.array([-np.eye(S + 1) for _ in groups]); mom = None
    for it in range(iters):
        # sample side: g = L w, w = (zeta_0..zeta_S)
        Ls = []
        for a in range(nG):
            L = np.zeros((S + 1, S + 2))
            for s in range(1, S + 1):
                use = systems[s - 1]["use"]
                if a in use:
                    r = -np.eye(S + 2)[s]; r[0] += use[a]; r[S + 1] += use[a]
                    for m in anc(parent, s): r = r - V[a][s, m] * L[m]
                    L[s] = r / V[a][s, s]
            Ls.append(L)
        Cn = np.zeros_like(C); Vn = np.zeros_like(V); mom = []
        for B, (q, rho) in enumerate(blocks):
            gs = [a for a in range(nG) if B in groups[a][1]]
            aB = sum(groups[a][0] * Ls[a][:, :S + 1] for a in gs); Sig = sum(groups[a][0] * Ls[a] @ full(C[a], a) @ Ls[a].T for a in gs)[1:, 1:]
            w, U = np.linalg.eigh((Sig + Sig.T) / 2); eta = eps[B] @ (U * np.sqrt(np.maximum(w, 0))).T
            beta = np.zeros((S + 1, N)); beta[0] = bst[B]; th = np.zeros((S + 1, N)); D = np.ones((S + 1, N)); T = np.zeros((S + 1, S + 1, N))
            for s in range(1, S + 1):
                c = systems[s - 1]["c"]; p = parent[s]; A = anc(parent, s); ass = aB[s, s]
                rhs = (th[p] if p else 0.0) - eta[:, s - 1] - aB[s, 0] * beta[0] - sum(aB[s, m] * beta[m] for m in A)
                lo, hi = -np.abs(rhs), np.abs(rhs)                                   # theta + ass c sinh(theta) = rhs, |theta| <= |rhs|
                for _ in range(60):
                    mid = 0.5 * (lo + hi); f = mid + ass * c * np.sinh(mid) - rhs; lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
                th[s] = 0.5 * (lo + hi); beta[s] = c * np.sinh(th[s]); D[s] = np.sqrt(beta[s]**2 + c * c)
                for m in range(1, S + 1):
                    dr = (T[p, m] / D[p] if p else 0.0) - (1.0 if m == s else 0.0) - sum(aB[s, mm] * T[mm, m] for mm in A)
                    T[s, m] = D[s] * dr / (1 + D[s] * ass)
            MB = beta @ beta.T / N; mom.append(MB); TB = T.mean(-1)
            for a in gs: Cn[a] += q * MB; Vn[a] += q * TB
        for a in range(nG):
            for s in range(S + 1):
                if Vn[a][s, s] == 0: Vn[a][s, s] = -1.0
        err = max(np.max(np.abs(Cn - C)), np.max(np.abs(Vn - V) / (1 + np.abs(V))))
        C = damp * Cn + (1 - damp) * C; V = damp * Vn + (1 - damp) * V
        if verbose and it % 20 == 0: print(it, err, flush=True)
        if err < tol: break
    return dict(mom=np.array(mom), C=C, V=V, err=err, it=it)

def dist(sol, blocks, s, m):
    """||beta_s - beta_m||^2 / ||beta*||^2 (m = None: distance to zero)"""
    q = np.array([b[0] for b in blocks]); M = sol["mom"]
    return float(sum(q[B] * (M[B][s, s] - (2 * M[B][s, m] - M[B][m, m] if m is not None else 0.0)) for B in range(len(q))) / sum(q[B] * M[B][0, 0] for B in range(len(q))))
