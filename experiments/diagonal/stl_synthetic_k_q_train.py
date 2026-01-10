#!/usr/bin/env python3
"""
Single-task synthetic-k experiment: solve the separable q-penalized interpolation problem

    beta_hat ∈ argmin_beta  sum_i  sqrt_k_i * q( 2 beta_i / sqrt_k_i )
    s.t.  X beta = y

where q(z) = 2 - sqrt(4+z^2) + z asinh(z/2).

This is a *synthetic k* experiment: we impose sqrt_k_i directly (heterogeneous if desired),
instead of inducing it via PT→FT.

Outputs a run directory containing:
  - metrics.json
  - k_r_arrays.npz  (beta_hat, beta_star, sqrt_k, k, r_theory)
and appends a row to repo-root experiment_results_stl_synthk.csv.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: torch.\n"
        "This repo's diagonal experiments require PyTorch; please activate the conda environment "
        "from `environment.yml` (or otherwise install torch) and re-run.\n"
        f"Original error: {e}"
    )
import fcntl


def make_deterministic(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def circular_sample(shape, *, generator: torch.Generator):
    W = torch.randn(*shape, generator=generator)
    return W / torch.sqrt((W**2).mean(dim=-1, keepdim=True))


def sample_sparse_teacher_beta(inp_dim: int, active_dim: int, *, generator: torch.Generator) -> torch.Tensor:
    idx = torch.randperm(inp_dim, generator=generator)[:active_dim]
    V = torch.sign(torch.rand((active_dim,), generator=generator) - 0.5)
    V[V == 0] = 1.0
    V = V / torch.sqrt(torch.tensor(float(active_dim)))
    beta = torch.zeros(inp_dim)
    beta[idx] = V
    return beta


def q(z: torch.Tensor) -> torch.Tensor:
    return 2.0 - torch.sqrt(4.0 + z**2) + z * torch.arcsinh(z / 2.0)


def objective(beta: torch.Tensor, sqrt_k: torch.Tensor) -> torch.Tensor:
    # sum_i sqrt_k_i * q( 2 beta_i / sqrt_k_i )
    return (sqrt_k * q(2.0 * beta / sqrt_k)).sum()


def safe_csv_append(csv_path: str, new_row: dict, max_retries: int = 5, base_delay: float = 0.1) -> bool:
    lock_file_path = f"{csv_path}.lock"
    for attempt in range(max_retries):
        try:
            with open(lock_file_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if os.path.exists(csv_path):
                    try:
                        import pandas as pd
                        existing = pd.read_csv(csv_path)
                    except Exception:
                        existing = None
                else:
                    existing = None
                import pandas as pd
                new_df = pd.DataFrame([new_row])
                if existing is None:
                    combined = new_df
                else:
                    all_cols = sorted(set(existing.columns).union(set(new_df.columns)))
                    combined = pd.concat(
                        [existing.reindex(columns=all_cols), new_df.reindex(columns=all_cols)],
                        ignore_index=True,
                    )
                combined.to_csv(csv_path, index=False)
                return True
        except (BlockingIOError, OSError):
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(delay)
                continue
            return False
        except Exception:
            return False
    return False


def summarize_1d(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x).reshape(-1).astype(np.float64)
    qs = np.array([0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    qv = np.quantile(x, qs)
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std(ddof=0)),
        f"{prefix}_min": float(qv[0]),
        f"{prefix}_p10": float(qv[3]),
        f"{prefix}_p50": float(qv[5]),
        f"{prefix}_p90": float(qv[7]),
        f"{prefix}_max": float(qv[10]),
    }


def build_sqrt_k(inp_dim: int, *, pattern: str, k_scale: float, k_spread: float, seed: int) -> torch.Tensor:
    """
    Return sqrt_k (positive) with mean scale controlled by k_scale.
    - pattern='uniform': sqrt_k_i = k_scale
    - pattern='logspace': sqrt_k_i spans [k_scale/k_spread, k_scale*k_spread] geometrically across i
    """
    if pattern == "uniform":
        sqrt_k = torch.full((inp_dim,), float(k_scale))
    elif pattern == "logspace":
        lo = float(k_scale) / float(k_spread)
        hi = float(k_scale) * float(k_spread)
        # deterministic ordering across dims
        vals = torch.logspace(np.log10(lo), np.log10(hi), steps=inp_dim)
        sqrt_k = vals
    else:
        raise ValueError(f"Unknown k pattern: {pattern}")
    # avoid zeros
    sqrt_k = torch.clamp(sqrt_k, min=1e-16)
    return sqrt_k


def solve_penalty_method(
    X: torch.Tensor,
    y: torch.Tensor,
    sqrt_k: torch.Tensor,
    *,
    seed: int,
    max_outer: int = 3,
    lbfgs_max_iter: int = 300,
):
    """
    Enforce interpolation via quadratic penalty: minimize
        objective(beta) + rho/2 ||X beta - y||^2
    with increasing rho and L-BFGS inner solves.
    """
    inp_dim = X.shape[1]
    beta = torch.zeros(inp_dim, requires_grad=True)

    # Increasing rho enforces interpolation more tightly (slower but more faithful).
    rhos = [1e2, 1e4, 1e6, 1e8][:max_outer]
    history = []

    for rho in rhos:
        opt = torch.optim.LBFGS([beta], lr=1.0, max_iter=int(lbfgs_max_iter), line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            resid = X @ beta - y
            loss = objective(beta, sqrt_k) + 0.5 * float(rho) * (resid @ resid)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            train_mse = F.mse_loss(X @ beta, y).item()
            obj = float(objective(beta, sqrt_k).item())
            history.append({"rho": rho, "train_mse": train_mse, "obj": obj})

    return beta.detach(), history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_folder", type=str, required=True)

    p.add_argument("--inp_dim", type=int, default=1000)
    p.add_argument("--active_dim", type=int, default=40)
    p.add_argument("--n_train", type=int, default=256)
    p.add_argument("--n_test", type=int, default=10000)

    p.add_argument("--k_pattern", type=str, default="logspace", choices=["uniform", "logspace"])
    p.add_argument("--k_scale", type=float, default=1e-3, help="Global scale for sqrt(k).")
    p.add_argument("--k_spread", type=float, default=1e3, help="For logspace pattern: multiplicative half-range.")
    # Defaults intentionally a bit heavier so results are stable; override for quick smoke tests.
    p.add_argument("--max_outer", type=int, default=4, help="Penalty-method outer iterations (smaller=faster).")
    p.add_argument("--lbfgs_max_iter", type=int, default=500, help="LBFGS max iterations per outer step (smaller=faster).")

    args = p.parse_args()
    make_deterministic(args.seed)
    os.makedirs(args.save_folder, exist_ok=True)

    gen_x = torch.Generator(device="cpu").manual_seed(args.seed + 0)
    gen_teacher = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    gen_test = torch.Generator(device="cpu").manual_seed(args.seed + 2)

    X = circular_sample((args.n_train, args.inp_dim), generator=gen_x)
    Xtest = circular_sample((args.n_test, args.inp_dim), generator=gen_test)
    beta_star = sample_sparse_teacher_beta(args.inp_dim, args.active_dim, generator=gen_teacher)
    y = X @ beta_star
    ytest = Xtest @ beta_star

    sqrt_k = build_sqrt_k(args.inp_dim, pattern=args.k_pattern, k_scale=args.k_scale, k_spread=args.k_spread, seed=args.seed)

    t0 = time.time()
    beta_hat, hist = solve_penalty_method(
        X, y, sqrt_k, seed=args.seed, max_outer=int(args.max_outer), lbfgs_max_iter=int(args.lbfgs_max_iter)
    )
    wall = time.time() - t0

    with torch.no_grad():
        train_mse = F.mse_loss(X @ beta_hat, y).item()
        test_mse = F.mse_loss(Xtest @ beta_hat, ytest).item()

    k = (sqrt_k**2).numpy()
    r_theory = (2.0 * torch.abs(beta_hat) / sqrt_k).numpy()

    arrays_path = os.path.join(args.save_folder, "k_r_arrays.npz")
    np.savez_compressed(
        arrays_path,
        beta_hat=beta_hat.numpy(),
        beta_star=beta_star.numpy(),
        sqrt_k=sqrt_k.numpy(),
        k=k,
        r_theory=r_theory,
        k_pattern=np.array(args.k_pattern),
        k_scale=np.array(float(args.k_scale), dtype=np.float64),
        k_spread=np.array(float(args.k_spread), dtype=np.float64),
    )

    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "save_folder": args.save_folder,
        "seed": args.seed,
        "inp_dim": args.inp_dim,
        "active_dim": args.active_dim,
        "n_train": args.n_train,
        "n_test": args.n_test,
        # User-facing convention: delta = d/n (so smaller delta = more samples)
        "delta": float(args.inp_dim) / float(args.n_train),
        "k_pattern": args.k_pattern,
        "k_scale": float(args.k_scale),
        "k_spread": float(args.k_spread),
        "train_mse": float(train_mse),
        "final_val_mse": float(test_mse),
        "wall_s": float(wall),
        "k_r_arrays_path": arrays_path,
        "solver_history": hist,
        **summarize_1d(k, "k"),
        **summarize_1d(np.sqrt(k), "sqrt_k"),
        **summarize_1d(r_theory, "r_theory"),
    }

    with open(os.path.join(args.save_folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Append to CSV
    csv_path = os.path.abspath("experiment_results_stl_synthk.csv")
    ok = safe_csv_append(csv_path, {k: v for k, v in metrics.items() if k != "solver_history"})
    if not ok:
        raise RuntimeError(f"Failed to append to {csv_path}")

    print(f"Done. train_mse={train_mse:.3e} test_mse={test_mse:.3e} wall_s={wall:.2f}")
    print(f"Wrote: {arrays_path}")


if __name__ == "__main__":
    main()


