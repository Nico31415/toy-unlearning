import numpy as np
from scipy.optimize import minimize

def psi(b, c):
    return np.sum(b * np.arcsinh(b / c) - np.sqrt(b * b + c * c))

def D(b, b0, c):
    return psi(b, c) - psi(b0, c) - np.arcsinh(b0 / c) @ (b - b0)

def solve_lam(loss, b0, c, lam, x0):
    th0 = np.arcsinh(b0 / c)
    def f(b):
        L, g = loss(b)
        return L + D(b, b0, c) / lam, g + (np.arcsinh(b / c) - th0) / lam
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10, maxcor=50))
    return r.x

def regpath(loss, b0, c, Ltarget, lo=-6, hi=12, it=40):
    """beta(lam) = argmin loss + D(beta,b0)/lam, with lam set so that loss(beta) = Ltarget (bisection in log lam)."""
    x = b0.copy()
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        b = solve_lam(loss, b0, c, 10 ** mid, x)
        if loss(b)[0] > Ltarget:
            lo = mid
        else:
            hi = mid
        x = b
    return solve_lam(loss, b0, c, 10 ** (0.5 * (lo + hi)), x), 0.5 * (lo + hi)
