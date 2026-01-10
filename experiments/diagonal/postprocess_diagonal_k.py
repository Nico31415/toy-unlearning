#!/usr/bin/env python3
"""
Postprocess a diagonal PT→FT run directory to compute induced k_i and r_i,
plus sparsity/overlap metrics, without modifying training code.

Assumes the run directory contains:
  - df.feather (train/val history)
  - weights_df.feather (must have been saved; contains beta vectors per task)

Outputs:
  - k_r_arrays.npz in the run directory
  - Appends a row to experiment_results_k.csv (repo root) with summary stats.
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

import fcntl


def parse_params_from_save_path(save_path: str) -> dict:
    """
    Parse key=value pairs from the terminal directory name of save_path.
    Expected format: key=value--key=value--...
    """
    dirname = os.path.basename(save_path.rstrip("/"))
    # stop at '--' but allow '-' in values
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


def l0_l1_l2(x: np.ndarray, thresh: float) -> dict:
    x = np.asarray(x).reshape(-1)
    supp = np.abs(x) > thresh
    return {
        "l0": int(supp.sum()),
        "l1": float(np.sum(np.abs(x))),
        "l2": float(np.sqrt(np.sum(x * x))),
    }, supp


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def reconstruct_teacher_beta_with_pretrain_overlap(
    *,
    pretrained_beta: np.ndarray,
    inp_dim: int,
    active_dim_2: int,
    pretrain_overlap: int,
    seed: int,
    same_signs: bool,
    active_threshold: float = 1e-4,
) -> np.ndarray:
    """
    Mirror `sample_finetuning_teacher_with_pretrain_overlap` from the finetune script,
    but implemented in numpy for postprocessing reproducibility.

    Uses the same generator convention as finetune: gen4 seeded with seed+3.
    """
    rng = np.random.default_rng(seed + 3)
    abs_beta = np.abs(pretrained_beta)
    above = int(np.sum(abs_beta > active_threshold))
    k = max(above, active_dim_2)
    k = min(k, inp_dim)
    active_indices = np.argsort(abs_beta)[::-1][:k]
    pretrain_active_dims = active_indices.tolist()

    if pretrain_overlap > min(len(pretrain_active_dims), active_dim_2):
        raise ValueError("pretrain_overlap too large for available PT active dims / active_dim_2")

    # choose overlap dims (random subset)
    if pretrain_overlap > 0:
        overlap_idx = rng.permutation(len(pretrain_active_dims))[:pretrain_overlap]
        overlap_dims = [pretrain_active_dims[i] for i in overlap_idx]
    else:
        overlap_dims = []

    remaining = active_dim_2 - pretrain_overlap
    if remaining > 0:
        inactive_dims = [i for i in range(inp_dim) if i not in pretrain_active_dims]
        if len(inactive_dims) < remaining:
            raise ValueError("not enough inactive dims to sample new teacher dims")
        new_idx = rng.permutation(len(inactive_dims))[:remaining]
        new_dims = [inactive_dims[i] for i in new_idx]
    else:
        new_dims = []

    finetune_active_dims = overlap_dims + new_dims

    V = np.zeros(active_dim_2, dtype=np.float64)
    if pretrain_overlap > 0 and same_signs:
        for i, dim in enumerate(overlap_dims):
            s = np.sign(pretrained_beta[dim])
            V[i] = 1.0 if s == 0 else s
    elif pretrain_overlap > 0:
        V[:pretrain_overlap] = np.sign(rng.random(pretrain_overlap) - 0.5)
        V[:pretrain_overlap][V[:pretrain_overlap] == 0] = 1.0

    if remaining > 0:
        tmp = np.sign(rng.random(remaining) - 0.5)
        tmp[tmp == 0] = 1.0
        V[pretrain_overlap:] = tmp

    V = V / np.sqrt(active_dim_2)

    beta_teacher = np.zeros(inp_dim, dtype=np.float64)
    for i, dim in enumerate(finetune_active_dims):
        beta_teacher[dim] = V[i]
    return beta_teacher


def postprocess_run(save_path: str, *, write_csv: bool = True) -> dict:
    save_path = save_path.rstrip("/") + "/"
    params = parse_params_from_save_path(save_path)

    df_path = os.path.join(save_path, "df.feather")
    weights_path = os.path.join(save_path, "weights_df.feather")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Missing {df_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing {weights_path} (run finetune with --save_weights)")

    df = pd.read_feather(df_path)
    wdf = pd.read_feather(weights_path)

    # extract betas (one-task PT+FT convention: task '1' is beta_PT, task '2' is beta_FT)
    beta_pt = wdf[wdf["task"] == "1"].sort_values("dim")["value"].to_numpy(dtype=np.float64)
    beta_ft = wdf[wdf["task"] == "2"].sort_values("dim")["value"].to_numpy(dtype=np.float64)

    inp_dim = int(params.get("inp_dim", len(beta_pt)))
    if len(beta_pt) != len(beta_ft):
        raise ValueError("beta_pt and beta_ft length mismatch")

    gamma = float(params.get("gamma", params.get("scaling", np.nan)))
    if not np.isfinite(gamma):
        # fall back: try parse from directory names like gamma=1.0e-03
        gamma_match = re.search(r"gamma=([\d\.\-eE]+)", os.path.basename(save_path.rstrip("/")))
        if gamma_match:
            gamma = float(gamma_match.group(1))
        else:
            gamma = float(params.get("scaling", 1.0))

    # induced k
    sqrt_k = np.abs(beta_pt) + gamma**2
    k = sqrt_k**2
    r_theory = 2.0 * np.abs(beta_ft) / sqrt_k
    r_code = np.abs(beta_ft) / sqrt_k

    # losses
    final_train_mse = float(df[df["split"] == "train"]["loss"].iloc[-1])
    final_val_mse = float(df[df["split"] == "val"]["loss"].iloc[-1])

    # teacher beta reconstruction (if possible)
    seed = int(params.get("seed", 0))
    active_dim_2 = int(params.get("active_dim_2", 0))
    pretrain_overlap = int(params.get("pretrain_overlap", params.get("overlap", 0)))
    same_signs = bool(params.get("same_signs", True))

    teacher_beta = None
    try:
        if active_dim_2 > 0:
            teacher_beta = reconstruct_teacher_beta_with_pretrain_overlap(
                pretrained_beta=beta_pt,
                inp_dim=inp_dim,
                active_dim_2=active_dim_2,
                pretrain_overlap=pretrain_overlap,
                seed=seed,
                same_signs=same_signs,
            )
    except Exception:
        teacher_beta = None

    # sparsity / overlap / cosines
    active_thresh = float(params.get("active_threshold", 1e-6))
    pt_stats, pt_supp = l0_l1_l2(beta_pt, thresh=active_thresh)
    ft_stats, ft_supp = l0_l1_l2(beta_ft, thresh=active_thresh)
    row = {
        **params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "save_path": save_path,
        "final_train_mse": final_train_mse,
        "final_val_mse": final_val_mse,
        "gamma": float(gamma),
        "beta_pt_l0": pt_stats["l0"],
        "beta_pt_l1": pt_stats["l1"],
        "beta_pt_l2": pt_stats["l2"],
        "beta_ft_l0": ft_stats["l0"],
        "beta_ft_l1": ft_stats["l1"],
        "beta_ft_l2": ft_stats["l2"],
        "cos_beta_pt_beta_ft": cos_sim(beta_pt, beta_ft),
    }

    if teacher_beta is not None:
        teacher_stats, teacher_supp = l0_l1_l2(teacher_beta, thresh=1e-12)
        inter_pt_teacher = int(np.sum(pt_supp & teacher_supp))
        inter_ft_teacher = int(np.sum(ft_supp & teacher_supp))
        row.update({
            "teacher_ft_l0": teacher_stats["l0"],
            "teacher_ft_l1": teacher_stats["l1"],
            "teacher_ft_l2": teacher_stats["l2"],
            "supp_overlap_pt_teacher": inter_pt_teacher,
            "supp_overlap_ft_teacher": inter_ft_teacher,
            "supp_overlap_pt_teacher_frac_of_teacher": float(inter_pt_teacher) / float(teacher_stats["l0"]) if teacher_stats["l0"] else float("nan"),
            "supp_overlap_ft_teacher_frac_of_teacher": float(inter_ft_teacher) / float(teacher_stats["l0"]) if teacher_stats["l0"] else float("nan"),
            "cos_beta_pt_teacher": cos_sim(beta_pt, teacher_beta),
            "cos_beta_ft_teacher": cos_sim(beta_ft, teacher_beta),
        })

    # summaries
    row.update(summarize_1d(k, "k"))
    row.update(summarize_1d(sqrt_k, "sqrt_k"))
    row.update(summarize_1d(r_theory, "r_theory"))
    row.update(summarize_regimes(r_theory, "r_theory"))
    row.update(summarize_1d(r_code, "r_code"))
    row.update(summarize_regimes(r_code, "r_code"))

    # save arrays to run dir
    arrays_path = os.path.join(save_path, "k_r_arrays.npz")
    np.savez_compressed(
        arrays_path,
        beta_pt=beta_pt,
        beta_ft=beta_ft,
        k=k,
        sqrt_k=sqrt_k,
        r_theory=r_theory,
        r_code=r_code,
        gamma=np.array(float(gamma), dtype=np.float64),
    )
    row["k_r_arrays_path"] = arrays_path

    if write_csv:
        csv_path = os.path.abspath("experiment_results_k.csv")
        ok = safe_csv_append(csv_path, row)
        if not ok:
            raise RuntimeError(f"Failed to append to {csv_path}")

    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save_path", type=str, required=True, help="Run directory produced by diagonal finetuning.")
    p.add_argument("--no_csv", action="store_true", help="Do not append to experiment_results_k.csv")
    args = p.parse_args()
    row = postprocess_run(args.save_path, write_csv=(not args.no_csv))
    print("Postprocessed run. Wrote k_r_arrays.npz and updated CSV.")
    print(f"save_path={row.get('save_path')}")


if __name__ == "__main__":
    main()




















