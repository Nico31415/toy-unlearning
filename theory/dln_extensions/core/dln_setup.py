"""Finite-d solver for the setup in claude/setup_questions.tex (losses.pdf).

Everything is an exact end point, no training loop:
  * every gradient-flow stage = Bregman projection (proj), geometry c = sqrt(k)/2,
    i.e. Q'(beta) = arcsinh(beta / c) (the paper's 1/2 arcsinh(2 beta / sqrt k) up to a factor that does not move the argmin);
  * rescaling the weights by sqrt(kappa): beta -> kappa beta, c -> kappa c (k -> kappa^2 k);
  * weight decay: weighted LASSO (lasso), independent of the start.

Notation follows the PDF: blocks dR, dSh, dF; R_full, F_full, D; unlearner's R, F; attacker's R_RL, F_re; F_ho = F minus F_re.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ----------------------------------------------------------------------------------------------------------------- data
@dataclass
class Data:
    d: int
    dR: np.ndarray            # coordinate indices of the three blocks
    dSh: np.ndarray
    dF: np.ndarray
    beta_star: np.ndarray
    X_Rfull: np.ndarray       # d x n_R, one column per sample
    X_Ffull: np.ndarray       # d x n_F
    rng: np.random.Generator

    @property
    def y_Rfull(self): return self.X_Rfull.T @ self.beta_star

    @property
    def y_Ffull(self): return self.X_Ffull.T @ self.beta_star

    def fresh(self, n, kind):
        """Fresh samples from P_R ('R') or P_F ('F')."""
        return _draw(self.rng, self.d, n, np.r_[self.dR, self.dSh] if kind == "R" else np.r_[self.dSh, self.dF])


def _draw(rng, d, n, coords):
    X = np.zeros((d, n))
    X[coords] = rng.standard_normal((len(coords), n)) / np.sqrt(d)
    return X


def make_data(d, blocks=(0.4, 0.2, 0.4), rho=0.2, n_R=None, n_F=None, seed=0):
    """blocks = proportions |dR| : |dSh| : |dF| of d. n_R, n_F = |R_full|, |F_full| (ints)."""
    rng = np.random.default_rng(seed)
    sizes = np.floor(np.array(blocks) / sum(blocks) * d).astype(int)
    sizes[0] += d - sizes.sum()
    idx = np.arange(d)
    dR, dSh, dF = idx[:sizes[0]], idx[sizes[0]:sizes[0] + sizes[1]], idx[sizes[0] + sizes[1]:]
    amp = rho ** -0.5
    beta_star = np.where(rng.random(d) < rho, rng.choice([-amp, amp], d), 0.0)
    X_R = _draw(rng, d, n_R, np.r_[dR, dSh])
    X_F = _draw(rng, d, n_F, np.r_[dSh, dF])
    return Data(d, dR, dSh, dF, beta_star, X_R, X_F, rng)


@dataclass
class Split:
    """Index sets into R_full / F_full columns."""
    R: np.ndarray      # unlearner's retain set (subset of R_full)
    F: np.ndarray      # deletion request (subset of F_full)
    F_re: np.ndarray   # attacker's relearning samples (subset of F)
    R_RL: np.ndarray   # attacker's retain set (subset of R_full)

    @property
    def F_ho(self):
        return np.setdiff1d(self.F, self.F_re)


def make_split(data, n_R_ul, n_F, n_re=0, n_R_rl=None, seed=0):
    """R = first n_R_ul retain samples; F = random n_F forget samples; F_re = random n_re of F; R_RL = first n_R_rl
    retain samples (default: all of R_full)."""
    rng = np.random.default_rng(seed + 1)
    nRf, nFf = data.X_Rfull.shape[1], data.X_Ffull.shape[1]
    F = np.sort(rng.choice(nFf, n_F, replace=False))
    F_re = np.sort(rng.choice(F, n_re, replace=False)) if n_re else np.array([], int)
    return Split(np.arange(n_R_ul), F, F_re, np.arange(nRf if n_R_rl is None else n_R_rl))


# ---------------------------------------------------------------------------------------------------- end-point solvers
def c_of(k):
    return np.sqrt(k) / 2


def proj(beta0, c, X, t, iters=300, tol=1e-11):
    """Bregman projection of beta0 onto {X^T beta = t}, geometry Q'(beta) = arcsinh(beta / c). X: d x n.
    Newton on the dual D(lam) = sum c cosh(phi0 + X lam) - lam^T t, with backtracking."""
    if X.shape[1] == 0:
        return beta0.copy()
    check_fittable(X, t)
    phi0 = np.arcsinh(beta0 / c)
    lam = np.zeros(X.shape[1])
    dual = lambda l: np.sum(c * np.cosh(phi0 + X @ l)) - l @ t
    for _ in range(iters):
        phi = phi0 + X @ lam
        g = X.T @ (c * np.sinh(phi)) - t
        if np.linalg.norm(g) < tol * (1 + np.linalg.norm(t)):
            break
        H = X.T @ (X * (c * np.cosh(phi))[:, None])
        step = np.linalg.solve(H + 1e-14 * np.eye(len(lam)), g)
        s, D0, g0 = 1.0, dual(lam), np.linalg.norm(g)
        gnorm = lambda l: np.linalg.norm(X.T @ (c * np.sinh(phi0 + X @ l)) - t)
        with np.errstate(over="ignore", invalid="ignore"):
            # backtrack on the dual; near the optimum the dual is flat to rounding, so also accept a step that
            # shrinks the gradient
            while s > 1e-12 and not (dual(lam - s * step) <= D0 - 0.25 * s * g @ step or gnorm(lam - s * step) < g0):
                s /= 2
        lam -= s * step
    else:
        raise RuntimeError("proj did not converge")
    return c * np.sinh(phi0 + X @ lam)


def check_fittable(X, t, rtol=1e-8):
    """Raise if X^T beta = t has no solution (zero loss unreachable)."""
    sol = np.linalg.lstsq(X.T, t, rcond=None)[0]
    res = np.linalg.norm(X.T @ sol - t)
    if res > rtol * (1 + np.linalg.norm(t)):
        raise ValueError(f"targets cannot be fitted exactly (residual {res:.2e}): zero loss is unreachable")


def lasso(X, t, w, lam, beta_init=None, iters=20000, tol=1e-12):
    """argmin_beta sum_i w_i/2 (x_i^T beta - t_i)^2 + 2 lam ||beta||_1   (weight decay lam ||theta||^2 at the optimum).
    FISTA. w > 0."""
    A = X * np.sqrt(w)[None, :]
    b = t * np.sqrt(w)
    L = np.linalg.eigvalsh(A @ A.T)[-1]
    beta = np.zeros(X.shape[0]) if beta_init is None else beta_init.copy()
    z, q = beta.copy(), 1.0
    for _ in range(iters):
        g = A @ (A.T @ z - b)
        v = z - g / L
        new = np.sign(v) * np.maximum(np.abs(v) - 2 * lam / L, 0)
        q_new = (1 + np.sqrt(1 + 4 * q * q)) / 2
        z = new + (q - 1) / q_new * (new - beta)
        if np.linalg.norm(new - beta) < tol * (1 + np.linalg.norm(beta)):
            beta = new
            break
        beta, q = new, q_new
    return beta


# ------------------------------------------------------------------------------------------------------------- stages
def train_from_zero(data, k, X, y):
    """A(T): the model trained from 0 on (X, y)."""
    return proj(np.zeros(data.d), c_of(k), X, y)


def references(data, split, k):
    """A(D), A(D \\ F) (untraining), A(D \\ F_full) (unlearning)."""
    keepF = np.setdiff1d(np.arange(data.X_Ffull.shape[1]), split.F)
    XD = np.c_[data.X_Rfull, data.X_Ffull]
    A_D = train_from_zero(data, k, XD, XD.T @ data.beta_star)
    X_noF = np.c_[data.X_Rfull, data.X_Ffull[:, keepF]]
    A_UT = train_from_zero(data, k, X_noF, X_noF.T @ data.beta_star)
    A_UL = train_from_zero(data, k, data.X_Rfull, data.y_Rfull)
    return A_D, A_UT, A_UL


def beta_parts(data):
    """Workshop split: beta_r = beta* on dR u dSh, beta_f = beta* on dF."""
    br, bf = np.zeros(data.d), np.zeros(data.d)
    rs = np.r_[data.dR, data.dSh]
    br[rs] = data.beta_star[rs]
    bf[data.dF] = data.beta_star[data.dF]
    return br, bf


def effective_teachers(loss, data, beta_g=None, mu_r=1.0, mu_d=1.0, mu_a=0.0):
    """Return (beta_eff_R, beta_eff_F, c_R, c_F): the per-sample loss is c/2 (f_i - x_i^T beta_eff)^2 + const.
    loss in {'relabel', 'anchored', 'workshop'}. beta_g: surrogate teacher (0 = label mean, or A(D \\ F_full))."""
    bs = data.beta_star
    bg = np.zeros(data.d) if beta_g is None else beta_g
    if loss == "relabel":
        return bs, bg, mu_r, mu_d
    if loss == "anchored":
        if not mu_d > mu_a:
            raise ValueError("anchored ascent is unbounded unless mu_d > mu_a")
        return bs, (mu_d * bg - mu_a * bs) / (mu_d - mu_a), mu_r, mu_d - mu_a
    if loss == "workshop":
        if not mu_r > mu_a:
            raise ValueError("workshop loss is unbounded unless mu_r > mu_a")
        br, bf = beta_parts(data)
        be = (mu_r * br - mu_a * bf) / (mu_r - mu_a)
        return be, be, mu_r - mu_a, mu_r - mu_a
    raise ValueError(loss)


def unlearn(data, split, k, kappa, A_D, loss, weight_decay=0.0, **loss_kw):
    """U: start kappa A(D), geometry kappa^2 k, fit R to beta_eff_R and F to beta_eff_F.
    With weight_decay > 0: the weighted LASSO end point (independent of the start)."""
    bR, bF, cR, cF = effective_teachers(loss, data, **loss_kw)
    XR, XF = data.X_Rfull[:, split.R], data.X_Ffull[:, split.F]
    X = np.c_[XR, XF]
    t = np.r_[XR.T @ bR, XF.T @ bF]
    if weight_decay > 0:
        w = np.r_[np.full(XR.shape[1], cR), np.full(XF.shape[1], cF)]
        return lasso(X, t, w, weight_decay)
    return proj(kappa * A_D, kappa * c_of(k), X, t)


def attack(data, split, M, k_M):
    """Attacker: from M (geometry k_M), fit R_RL and F_re with true labels."""
    X = np.c_[data.X_Rfull[:, split.R_RL], data.X_Ffull[:, split.F_re]]
    return proj(M, c_of(k_M), X, X.T @ data.beta_star)


# ------------------------------------------------------------------------------------------------------------ measures
def dist(data, a, b):
    """||a - b|| / ||beta* on dF||."""
    return np.linalg.norm(a - b) / np.linalg.norm(data.beta_star[data.dF])


def retain_error(data, beta):
    """Population MSE on fresh P_R inputs / that of beta = 0 (exact for isotropic Gaussian inputs)."""
    rs = np.r_[data.dR, data.dSh]
    return np.sum((beta - data.beta_star)[rs] ** 2) / np.sum(data.beta_star[rs] ** 2)


def forget_error_pop(data, beta):
    """Population MSE on fresh P_F inputs / that of beta = 0."""
    fs = np.r_[data.dSh, data.dF]
    return np.sum((beta - data.beta_star)[fs] ** 2) / np.sum(data.beta_star[fs] ** 2)


def forget_error_ho(data, split, beta):
    """MSE on the held-out forget samples F_ho = F \\ F_re / that of beta = 0."""
    X = data.X_Ffull[:, split.F_ho]
    y = X.T @ data.beta_star
    return np.mean((X.T @ beta - y) ** 2) / np.mean(y ** 2)


def first_crossing(xs, ys, thr):
    """Smallest x with y <= thr (xs increasing); nan if never."""
    for x, y in zip(xs, ys):
        if y <= thr:
            return x
    return np.nan


# --------------------------------------------------------------------------------- gradient descent on the network
def gd_network(beta0, c, terms, weight_decay=0.0, steps=200000, eps=0.01, tol=1e-10):
    """Plain GD on balanced weights beta = a^2 - b^2 (a = u+ = v+, b = u- = v-), starting from beta0 with
    a*b = c/2 (so the mirror geometry is c). terms: list of (mu, X, t) with SIGNED mu: L = sum mu/2 ||X^T beta - t||^2.
    weight_decay lam adds 2 lam (a^2 + b^2) = lam ||theta||^2. Used only to check the closed forms."""
    s = np.sqrt(beta0 ** 2 + c ** 2)            # a^2 + b^2, with a^2 - b^2 = beta0 and 2ab = c
    a, b = np.sqrt((s + beta0) / 2), np.sqrt((s - beta0) / 2)
    Lam = np.linalg.eigvalsh(sum(abs(mu) * X @ X.T for mu, X, t in terms))[-1]
    for i in range(steps):
        beta = a * a - b * b
        g = sum(mu * X @ (X.T @ beta - t) for mu, X, t in terms)
        ga = 2 * a * g + 4 * weight_decay * a
        gb = -2 * b * g + 4 * weight_decay * b
        # step: at most a relative change eps per weight, and never above the stability limit of the loss
        # (an uncapped relative step grows as the gradient vanishes and makes GD wander along the zero-loss set)
        lr_stable = 0.25 / (4 * np.max(a * a + b * b) * Lam + 4 * weight_decay + 1e-30)
        lr = min(lr_stable,
                 eps / (np.max(np.abs(ga) / np.maximum(a, 1e-300)) + np.max(np.abs(gb) / np.maximum(b, 1e-300)) + 1e-30))
        a, b = a - lr * ga, b - lr * gb
        if i % 500 == 0 and max(np.max(np.abs(ga)), np.max(np.abs(gb))) < tol:
            break
    return a * a - b * b
