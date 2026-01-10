#!/usr/bin/env python3
"""
Postprocess a *single-task* diagonal pretraining run to compute a (currently uniform) k and r.

Why "uniform" k?
  In the single-task pretraining code, initialization parameters are identical across coordinates
  (set by c and lmda in complex init), so the induced scale is the same for all i unless we
  introduce heterogeneous init (not in the existing code).

We still log k/r because:
  - sweeping (c, lmda) changes the global scale
  - sweeping teacher sparsity (active_dim) changes beta structure
  - this provides a clean STL baseline before PT→FT experiments

Outputs (per run directory):
  - k_r_arrays.npz
Appends to (repo root):
  - experiment_results_st_k.csv
"""

from __future__ import annotations

import argparse
import os
import re
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: torch.\n"
        "This postprocessor loads the trained diagonal model checkpoint to extract beta_hat, "
        "so it requires PyTorch. Activate the repo conda env from `environment.yml`.\n"
        f"Original error: {e}"
    )

import fcntl


def safe_csv_append(csv_path: str, new_row_data: dict, max_retries: int = 5, base_delay: float = 0.1) -> bool:
    lock_file_path = f"{csv_path}.lock"
    for attempt in range(max_retries):
        try:
            with open(lock_file_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if os.path.exists(csv_path):
                    try:
                        existing_df = pd.read_csv(csv_path)
                    except Exception:
                        existing_df = pd.DataFrame()
                else:
                    existing_df = pd.DataFrame()

                new_df = pd.DataFrame([new_row_data])
                all_columns = sorted(set(list(existing_df.columns) + list(new_df.columns)))
                existing_df = existing_df.reindex(columns=all_columns)
                new_df = new_df.reindex(columns=all_columns)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
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


def parse_params_from_save_folder(save_folder: str) -> dict:
    """
    For PT runs, save_folder naming is like:
      seed=0--active_dim=40--c=1.0e-03--lmda=0.0000000000--init_method=complex/
    """
    dirname = os.path.basename(save_folder.rstrip("/"))
    param_pattern = r"(\w+)=((?:(?!--).)+?)(?=--|$)"
    matches = re.findall(param_pattern, dirname)
    params: dict[str, object] = {}
    for key, value in matches:
        try:
            if value.lower() in ["true", "false"]:
                params[key] = value.lower() == "true"
            elif re.match(r"^-?\d+$", value):
                params[key] = int(value)
            elif re.match(r"^-?\d*\.?\d+(?:[eE][+-]?\d+)?$", value):
                params[key] = float(value)
            else:
                params[key] = value
        except Exception:
            params[key] = value
    return params


def summarize_1d(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x).reshape(-1).astype(np.float64)
    qs = np.array([0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    qv = np.quantile(x, qs)
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_std": float(x.std(ddof=0)),
        f"{prefix}_min": float(qv[0]),
        f"{prefix}_p01": float(qv[1]),
        f"{prefix}_p05": float(qv[2]),
        f"{prefix}_p10": float(qv[3]),
        f"{prefix}_p25": float(qv[4]),
        f"{prefix}_p50": float(qv[5]),
        f"{prefix}_p75": float(qv[6]),
        f"{prefix}_p90": float(qv[7]),
        f"{prefix}_p95": float(qv[8]),
        f"{prefix}_p99": float(qv[9]),
        f"{prefix}_max": float(qv[10]),
    }


def summarize_regimes(r: np.ndarray, prefix: str, thresholds=(0.1, 1.0, 10.0)) -> dict:
    r = np.asarray(r).reshape(-1)
    out = {}
    for t in thresholds:
        out[f"{prefix}_frac_le_{t:g}"] = float(np.mean(r <= t))
    return out


class DiagonalNetPT(nn.Module):
    """
    Minimal clone of the PT diagonal net definition (so we don't depend on importing the PT script).
    """
    def __init__(self, inp_dim: int, scaling: float = 1.0, lmda: float = 0.0, c: float = 1e-3, init_method: str = "complex"):
        super().__init__()
        if init_method == "simple":
            self.w_pos = nn.Parameter(scaling * torch.ones(inp_dim))
            self.v_pos = nn.Parameter(scaling * torch.ones(inp_dim))
            self.v_neg = nn.Parameter(scaling * torch.ones(inp_dim))
            self.w_neg = nn.Parameter(scaling * torch.ones(inp_dim))
        elif init_method == "complex":
            if c * c < lmda * lmda:
                raise ValueError("Require c^2 >= lmda^2")
            v = np.sqrt((c + lmda) / 2)
            u = np.sqrt((c - lmda) / 2)
            w_pos, w_neg, v_pos, v_neg = v, v, u, u
            self.w_pos = nn.Parameter(w_pos * torch.ones(inp_dim))
            self.v_pos = nn.Parameter(v_pos * torch.ones(inp_dim))
            self.v_neg = nn.Parameter(v_neg * torch.ones(inp_dim))
            self.w_neg = nn.Parameter(w_neg * torch.ones(inp_dim))
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

    def beta(self):
        return self.w_pos * self.v_pos - self.w_neg * self.v_neg


def sample_teacher_beta(inp_dim: int, active_dim: int, seed: int) -> np.ndarray:
    """
    Mirror PT teacher sampling:
      torch.manual_seed(seed); W = one_hot(randperm[:active_dim]); V = sign(rand - 0.5)/sqrt(active_dim)
    Returns beta* in R^inp_dim.
    """
    torch.manual_seed(seed)
    idx = torch.randperm(inp_dim)[:active_dim]
    V = torch.sign(torch.rand((active_dim,)) - 0.5).float() / float(np.sqrt(active_dim))
    beta = torch.zeros(inp_dim)
    beta[idx] = V
    return beta.numpy().astype(np.float64)


def postprocess_run(save_folder: str, *, write_csv: bool = True) -> dict:
    save_folder = save_folder.rstrip("/") + "/"
    params = parse_params_from_save_folder(save_folder)

    df_path = os.path.join(save_folder, "df.feather")
    model_path = os.path.join(save_folder, "model.pt")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Missing {df_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}")

    df = pd.read_feather(df_path)
    final_train_mse = float(df[df["split"] == "train"]["loss"].iloc[-1])
    final_val_mse = float(df[df["split"] == "val"]["loss"].iloc[-1])

    seed = int(params.get("seed", 0))
    inp_dim = int(params.get("inp_dim", 1000))  # inp_dim isn't always in folder name
    active_dim = int(params.get("active_dim", params.get("active_dim_1", 0)))
    c = float(params.get("c", 1e-3))
    lmda = float(params.get("lmda", 0.0))
    init_method = str(params.get("init_method", "complex"))
    scaling = float(params.get("scaling", 1e-3))

    # Load learned beta
    net = DiagonalNetPT(inp_dim=inp_dim, scaling=scaling, lmda=lmda, c=c, init_method=init_method)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    beta_hat = net.beta().detach().cpu().numpy().astype(np.float64)

    # Teacher beta*
    beta_star = sample_teacher_beta(inp_dim=inp_dim, active_dim=active_dim, seed=seed) if active_dim > 0 else None

    # Define STL "k" (uniform): sqrt(k) := c, k := c^2
    # (Explicitly labeled STL-init-induced; not the PT→FT induced k.)
    sqrt_k = np.full_like(beta_hat, fill_value=c, dtype=np.float64)
    k = sqrt_k ** 2
    r_theory = 2.0 * np.abs(beta_hat) / sqrt_k
    r_code = np.abs(beta_hat) / sqrt_k

    # Save arrays
    arrays_path = os.path.join(save_folder, "k_r_arrays.npz")
    np.savez_compressed(
        arrays_path,
        beta_hat=beta_hat,
        beta_star=(beta_star if beta_star is not None else np.array([], dtype=np.float64)),
        k=k,
        sqrt_k=sqrt_k,
        r_theory=r_theory,
        r_code=r_code,
        c=np.array(c, dtype=np.float64),
        lmda=np.array(lmda, dtype=np.float64),
    )

    row = {
        **params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "save_folder": save_folder,
        "final_train_mse": final_train_mse,
        "final_val_mse": final_val_mse,
        "c": c,
        "lmda": lmda,
        "stl_k_definition": "sqrt_k=c, k=c^2 (uniform; init-induced STL scale)",
        "k_r_arrays_path": arrays_path,
    }
    row.update(summarize_1d(k, "k"))
    row.update(summarize_1d(sqrt_k, "sqrt_k"))
    row.update(summarize_1d(r_theory, "r_theory"))
    row.update(summarize_regimes(r_theory, "r_theory"))

    if write_csv:
        csv_path = os.path.abspath("experiment_results_st_k.csv")
        ok = safe_csv_append(csv_path, row)
        if not ok:
            raise RuntimeError(f"Failed to append to {csv_path}")

    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_folder", type=str, required=True)
    p.add_argument("--no_csv", action="store_true")
    args = p.parse_args()
    postprocess_run(args.save_folder, write_csv=(not args.no_csv))
    print("Postprocessed STL run.")


if __name__ == "__main__":
    main()


