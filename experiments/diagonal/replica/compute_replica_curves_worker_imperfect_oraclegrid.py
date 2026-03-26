import argparse
import itertools
import math
from pathlib import Path
from typing import Dict, List, Tuple
 
import numpy as np
import pandas as pd
 
import ptft_replica_imperfect_pt as rip
 
 
def _now_tag() -> str:
    import time
 
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())
 
 
def _fmt_float(x: float) -> str:
    # Stable-ish filename-friendly float
    s = ("%.6g" % float(x)).replace("+", "")
    return s.replace(".", "p").replace("-", "m")


def _atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)


def _safe_master_append(master_csv: Path, row: Dict, key_cols: List[str]) -> None:
    """
    Append row to master_csv under a file lock iff key not already present.
    Safe under SLURM-array concurrency.
    """
    import fcntl
    import random
    import time

    master_csv = Path(master_csv)
    lock_path = Path(str(master_csv) + ".lock")
    master_csv.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(60):
        try:
            with open(lock_path, "w") as lock_f:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

                if master_csv.exists():
                    try:
                        df = pd.read_csv(master_csv)
                    except Exception:
                        df = pd.DataFrame()
                else:
                    df = pd.DataFrame()

                if df.empty:
                    pd.DataFrame([row]).to_csv(master_csv, index=False)
                    return

                for k in key_cols:
                    if k not in df.columns:
                        df[k] = np.nan

                mask = np.ones(len(df), dtype=bool)
                for k in key_cols:
                    mask &= (df[k].astype(str) == str(row.get(k)))
                if bool(mask.any()):
                    return

                new_df = pd.DataFrame([row])
                all_cols = sorted(set(df.columns.tolist() + new_df.columns.tolist()))
                df = df.reindex(columns=all_cols)
                new_df = new_df.reindex(columns=all_cols)
                out = pd.concat([df, new_df], ignore_index=True)
                out.to_csv(master_csv, index=False)
                return
        except Exception:
            if attempt == 59:
                raise
            time.sleep(0.1 * (2 ** min(attempt, 6)) + random.uniform(0, 0.05))
 
 
def _dedup_configs(configs: List[Dict[str, float]]) -> List[Dict[str, float]]:
    seen = set()
    out = []
    for c in configs:
        key = tuple(sorted((k, float(v)) for k, v in c.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
 
 
def _oracle_ptft_configs_omega01() -> List[Dict[str, float]]:
    """
    Replica-side reimplementation of the PT+FT oracle config grid from
    experiments/diagonal/custom_oracle_sweep_driver.py, restricted to omega ∈ {0,1}.
 
    We intentionally avoid importing custom_oracle_sweep_driver.py here because it
    eagerly loads Torch-based experiment modules at import time.
    """
    cfgs = []
 
    a_pt = 1.0
 
    # ---- Family A: full-overlap (omega=1), dense PT rho_pt=0.9999 ----
    rho_pt_dense = 0.9999
    omega_full = 1.0
 
    # A1) rho_ft=0.1, lambda sweep (gamma=0, c_pt=0.001)
    for lambda_pt in [-0.00099, -0.0005, 0.0, 0.0005, 0.00099]:
        cfgs.append(
            dict(
                rho_pt=rho_pt_dense,
                rho_ft=0.1,
                omega=omega_full,
                a_pt=a_pt,
                c_pt=0.001,
                lambda_pt=float(lambda_pt),
                gamma_reinit=0.0,
            )
        )
 
    # A2) rho_ft=0.1, gamma sweep (lambda=0, c_pt=0.001)
    for gamma_reinit in [0.0, 0.1, 1.0]:
        cfgs.append(
            dict(
                rho_pt=rho_pt_dense,
                rho_ft=0.1,
                omega=omega_full,
                a_pt=a_pt,
                c_pt=0.001,
                lambda_pt=0.0,
                gamma_reinit=float(gamma_reinit),
            )
        )
 
    # A3) rho_ft=0.1, c_pt sweep (lambda=0, gamma=0)
    for c_pt in [1e-6, 1e-3, 1e-1, 1.0]:
        cfgs.append(
            dict(
                rho_pt=rho_pt_dense,
                rho_ft=0.1,
                omega=omega_full,
                a_pt=a_pt,
                c_pt=float(c_pt),
                lambda_pt=0.0,
                gamma_reinit=0.0,
            )
        )
 
    # A4) rho_ft ∈ {0.1, 0.9}, lambda sweep (gamma=0, c_pt=0.001)
    for rho_ft in [0.1, 0.9]:
        for lambda_pt in [-0.00099, -0.0005, 0.0, 0.0005, 0.00099]:
            cfgs.append(
                dict(
                    rho_pt=rho_pt_dense,
                    rho_ft=float(rho_ft),
                    omega=omega_full,
                    a_pt=a_pt,
                    c_pt=0.001,
                    lambda_pt=float(lambda_pt),
                    gamma_reinit=0.0,
                )
            )
 
    # ---- Family B: overlap endpoints only, sparse PT rho_pt=0.1, rho_ft=0.1 ----
    rho_pt_sparse = 0.1
    rho_ft_overlap = 0.1
    omegas = [0.0, 1.0]
 
    # B2) omega endpoints × lambda sweep (gamma=0, c_pt=0.001)
    for omega in omegas:
        for lambda_pt in [-0.00099, -0.0005, 0.0, 0.0005, 0.00099]:
            cfgs.append(
                dict(
                    rho_pt=rho_pt_sparse,
                    rho_ft=rho_ft_overlap,
                    omega=float(omega),
                    a_pt=a_pt,
                    c_pt=0.001,
                    lambda_pt=float(lambda_pt),
                    gamma_reinit=0.0,
                )
            )
 
    # B3) omega endpoints × gamma sweep (lambda=0, c_pt=0.001)
    for omega in omegas:
        for gamma_reinit in [0.0, 0.1, 1.0]:
            cfgs.append(
                dict(
                    rho_pt=rho_pt_sparse,
                    rho_ft=rho_ft_overlap,
                    omega=float(omega),
                    a_pt=a_pt,
                    c_pt=0.001,
                    lambda_pt=0.0,
                    gamma_reinit=float(gamma_reinit),
                )
            )
 
    cfgs = _dedup_configs(cfgs)
 
    # Stable ordering: match oracle driver sort key for ptft:
    # (rho_pt, rho_ft, omega, c_pt, lambda_pt, gamma_reinit)
    cfgs.sort(
        key=lambda d: (
            float(d["rho_pt"]),
            float(d["rho_ft"]),
            float(d["omega"]),
            float(d["c_pt"]),
            float(d["lambda_pt"]),
            float(d["gamma_reinit"]),
        )
    )
    return cfgs
 
 
def main() -> None:
    p = argparse.ArgumentParser(
        description="Replica PT+FT curves with imperfect pretraining over oracle config grid (omega endpoints)"
    )
    p.add_argument("--task-id", type=int, required=False, default=None, help="Slurm array task ID (0..N-1)")
    p.add_argument(
        "--n-alpha-chunks",
        type=int,
        default=1,
        help="Split the FT alpha grid into this many chunks to increase parallelism.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print selected combo/chunk and exit.")
    p.add_argument("--list", action="store_true", help="Print config/task counts and exit.")
    p.add_argument(
        "--output-dir",
        type=str,
        default="results/replica_imperfect_pt_oraclegrid",
        help="Directory to save results",
    )
    p.add_argument(
        "--alpha-pt-list",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 0.5],
        help="Imperfect PT sample ratios alpha_pt to run (exclude 1.0).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip computation if the output CSV already exists (default).",
    )
    p.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Disable skip-existing behavior (force recompute).",
    )
    p.set_defaults(skip_existing=True)
    p.add_argument(
        "--write-master",
        action="store_true",
        help="Append an index row to output-dir/master.csv under a file lock (default).",
    )
    p.add_argument(
        "--no-write-master",
        dest="write_master",
        action="store_false",
        help="Disable master.csv index writes.",
    )
    p.set_defaults(write_master=True)
    args = p.parse_args()
 
    alpha_pt_list = [float(x) for x in args.alpha_pt_list]
    # Enforce the user-requested intent: skip alpha_pt >= 1.0 (oracle already exists).
    alpha_pt_list = [x for x in alpha_pt_list if x < 1.0]
    if not alpha_pt_list:
        raise SystemExit("alpha_pt_list empty after filtering (<1.0).")
 
    n_alpha_chunks = int(args.n_alpha_chunks)
    if n_alpha_chunks <= 0:
        raise ValueError("--n-alpha-chunks must be >= 1")
 
    # Match other replica workers: dense alpha sweep in [0.01, 0.5]
    alphas_full = np.linspace(0.01, 0.5, 80)
 
    # Solver/MC settings (match existing replica workers + imperfect script defaults)
    sigma0_pt = 0.0
    gamma_ext = 1e-6
    sigma0_2 = 0.0
    mc = 80_000
    seed = 0
    tol = 1e-6
    max_iters = 900
    damp = 0.25
 
    cfgs = _oracle_ptft_configs_omega01()
 
    # Task mapping: (config_idx, alpha_pt, alpha_chunk_idx)
    combos = list(itertools.product(range(len(cfgs)), alpha_pt_list))
    total_tasks = len(combos) * n_alpha_chunks
 
    if args.list:
        print("configs=" + str(len(cfgs)))
        print("alpha_pt_list=" + ",".join(str(x) for x in alpha_pt_list))
        print("n_alpha_chunks=" + str(n_alpha_chunks))
        print("total_tasks=" + str(total_tasks))
        print("array_range=0-" + str(total_tasks - 1))
        return
 
    if args.task_id is None:
        raise SystemExit("ERROR: provide --task-id or use --list")
 
    task_id = int(args.task_id)
    if task_id < 0 or task_id >= total_tasks:
        raise SystemExit("ERROR: task-id %d out of range 0..%d" % (task_id, total_tasks - 1))
 
    combo_idx = task_id // n_alpha_chunks
    alpha_chunk_idx = task_id % n_alpha_chunks
 
    cfg_idx, alpha_pt = combos[combo_idx]
    cfg = cfgs[int(cfg_idx)]
 
    alpha_chunks = np.array_split(alphas_full, n_alpha_chunks)
    alphas = alpha_chunks[int(alpha_chunk_idx)]
 
    rho_pt = float(cfg["rho_pt"])
    rho_ft = float(cfg["rho_ft"])
    omega = float(cfg["omega"])
    a_pt = float(cfg["a_pt"])
    c_pt = float(cfg["c_pt"])
    lambda_pt = float(cfg["lambda_pt"])
    gamma_reinit = float(cfg["gamma_reinit"])
 
    print(
        "task=%d/%d | cfg=%d/%d | alpha_pt=%s | rho_pt=%s rho_ft=%s omega=%s | c_pt=%s lambda_pt=%s gamma_reinit=%s | alpha_chunk=%d/%d n_alpha=%d"
        % (
            task_id,
            total_tasks - 1,
            int(cfg_idx),
            len(cfgs) - 1,
            str(alpha_pt),
            str(rho_pt),
            str(rho_ft),
            str(omega),
            str(c_pt),
            str(lambda_pt),
            str(gamma_reinit),
            int(alpha_chunk_idx) + 1,
            int(n_alpha_chunks),
            int(len(alphas)),
        )
    )
 
    if args.dry_run:
        return

    # Directory layout: group by alpha_pt and cfg for easy browsing
    out_dir = Path(args.output_dir) / ("alpha_pt=" + _fmt_float(float(alpha_pt))) / ("cfg=%03d" % int(cfg_idx))
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = Path(args.output_dir) / "master.csv"

    alpha_min = float(alphas[0])
    alpha_max = float(alphas[-1])

    row_id = (
        "cfg=%03d" % int(cfg_idx)
        + "__alpha_pt=%.6g" % float(alpha_pt)
        + "__alphachunk=%02dof%02d" % (int(alpha_chunk_idx), int(n_alpha_chunks))
        + "__alpharange=%.6g-%.6g" % (float(alpha_min), float(alpha_max))
    )
 
    fname = (
        "IMPERFECT_REPLICA_oraclegrid"
        + "_cfg%03d" % int(cfg_idx)
        + "_alphaPT" + _fmt_float(float(alpha_pt))
        + "_omega" + _fmt_float(float(omega))
        + "_rhop" + _fmt_float(float(rho_pt))
        + "_rhof" + _fmt_float(float(rho_ft))
        + "_c" + _fmt_float(float(c_pt))
        + "_lam" + _fmt_float(float(lambda_pt))
        + "_greinit" + _fmt_float(float(gamma_reinit))
        + "_seed" + str(int(seed))
        + "_alphachunk%02dof%02d" % (int(alpha_chunk_idx), int(n_alpha_chunks))
        + "_alpharange%.4f-%.4f" % (alpha_min, alpha_max)
        + ".csv"
    )
 
    out_path = out_dir / fname

    # Idempotency: if it exists, skip recompute (default)
    if bool(args.skip_existing) and out_path.exists() and out_path.stat().st_size > 0:
        print("skip-existing: %s" % str(out_path))
        if bool(args.write_master):
            _safe_master_append(
                master_csv,
                row={
                    "row_id": row_id,
                    "status": "skipped_exists",
                    "task_id": int(task_id),
                    "cfg_idx": int(cfg_idx),
                    "alpha_pt": float(alpha_pt),
                    "alpha_chunk_idx": int(alpha_chunk_idx),
                    "n_alpha_chunks": int(n_alpha_chunks),
                    "alpha_min": float(alpha_min),
                    "alpha_max": float(alpha_max),
                    "out_csv": str(out_path),
                    "rho_pt": float(rho_pt),
                    "rho_ft": float(rho_ft),
                    "omega": float(omega),
                    "a_pt": float(a_pt),
                    "c_pt": float(c_pt),
                    "lambda_pt": float(lambda_pt),
                    "gamma_reinit": float(gamma_reinit),
                },
                key_cols=["row_id"],
            )
        return

    curve, reliability, info = rip.ptft_qk_curve_imperfect_pt(
        rho_pt=rho_pt,
        rho_ft=rho_ft,
        omega=omega,
        alpha_pt=float(alpha_pt),
        sigma0_pt=float(sigma0_pt),
        gamma_ext=float(gamma_ext),
        sigma0_2=float(sigma0_2),
        alphas=np.asarray(alphas, float),
        mc=int(mc),
        seed=int(seed),
        a_pt=float(a_pt),
        c_pt=float(c_pt),
        lambda_pt=float(lambda_pt),
        gamma_reinit=float(gamma_reinit),
        tol=float(tol),
        max_iters=int(max_iters),
        damp=float(damp),
    )

    n = len(curve["alpha"])
    df = pd.DataFrame(
        {
            "row_id": np.full(n, row_id),
            "cfg_idx": np.full(n, int(cfg_idx)),
            "task_id": np.full(n, int(task_id)),
            "alpha_pt": np.full(n, float(alpha_pt)),
            "alpha_chunk_idx": np.full(n, int(alpha_chunk_idx)),
            "n_alpha_chunks": np.full(n, int(n_alpha_chunks)),
            "alpha": curve["alpha"],
            "mse_best": curve["mse_best"],
            "mse_fwd": curve["mse_fwd"],
            "mse_bwd": curve["mse_bwd"],
            "diff_db": curve["diff_db"],
            "fp_residual": curve["fp_residual"],
            "mse_se": curve["mse_se"],
            "mse_rel_se": curve["mse_rel_se"],
            "mse_se_db": curve["mse_se_db"],
            # config metadata
            "rho_pt": rho_pt,
            "rho_ft": rho_ft,
            "omega": omega,
            "a_pt": a_pt,
            "c_pt": c_pt,
            "lambda_pt": lambda_pt,
            "gamma_reinit": gamma_reinit,
            "reliability_score_db": float(reliability.get("score_db", float("nan"))),
            # imperfect-PT diagnostics (where available)
            "s2_pt": float(info.get("s2_pt", float("nan"))),
            "gp_pt": float(info.get("gp_pt", float("nan"))),
            "res_pt": float(info.get("res_pt", float("nan"))),
            "pt_oracle": bool(info.get("oracle", False)),
            # run metadata
            "seed": int(seed),
            "mc": int(mc),
            "tol": float(tol),
            "max_iters": int(max_iters),
            "damp": float(damp),
            "sigma0_pt": float(sigma0_pt),
            "gamma_ext": float(gamma_ext),
            "sigma0_2": float(sigma0_2),
        }
    )

    _atomic_write_csv(df, out_path)
    print("saved: %s" % str(out_path))

    if bool(args.write_master):
        _safe_master_append(
            master_csv,
            row={
                "row_id": row_id,
                "status": "ok",
                "task_id": int(task_id),
                "cfg_idx": int(cfg_idx),
                "alpha_pt": float(alpha_pt),
                "alpha_chunk_idx": int(alpha_chunk_idx),
                "n_alpha_chunks": int(n_alpha_chunks),
                "alpha_min": float(alpha_min),
                "alpha_max": float(alpha_max),
                "out_csv": str(out_path),
                "reliability_score_db": float(reliability.get("score_db", float("nan"))),
                "rho_pt": float(rho_pt),
                "rho_ft": float(rho_ft),
                "omega": float(omega),
                "a_pt": float(a_pt),
                "c_pt": float(c_pt),
                "lambda_pt": float(lambda_pt),
                "gamma_reinit": float(gamma_reinit),
                "seed": int(seed),
                "mc": int(mc),
                "tol": float(tol),
                "max_iters": int(max_iters),
                "damp": float(damp),
                "sigma0_pt": float(sigma0_pt),
                "gamma_ext": float(gamma_ext),
                "sigma0_2": float(sigma0_2),
            },
            key_cols=["row_id"],
        )
 
 
if __name__ == "__main__":
    main()
