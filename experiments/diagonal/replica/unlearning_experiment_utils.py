from __future__ import annotations

import fcntl
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


INP_DIM = 5000
RHO_PT = 0.1
RHO_FT = 0.1
A_PT = 1.0
N_TEST = 10_000

FORGETTING_ALPHAS = list(np.linspace(0.01, 0.5, 11))
SANITY_ALPHAS = list(np.linspace(0.01, 0.8, 11))
REPLICA_SANITY_ALPHAS = list(np.linspace(0.01, 0.8, 80))
CORRELATED_QS = [0.25, 0.50, 0.75]
SANITY_TEACHERS = ["aligned_overlap", "zero_overlap", "opposite_overlap"]
REPLICA_SANITY_TEACHERS = ["aligned_overlap", "opposite_overlap"]
POST_SELECTIONS = ["first_valid", "matched_alpha"]
RECOVERY_TARGETS = ["full_pt", "ptonly"]
RECOVERY_ALPHAS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
GD_VARIANTS = ["readout_only_keep_w", "readout_only_reinit_w", "full_keep_w", "full_reinit_w"]

REGIMES_099 = [
    ("regime_II", 1e-3, 0.0, 0.0),
    ("regime_III", 1e-3, 0.0, 10.0),
    ("regime_IV", 1e-3, -0.99e-3, 0.0),
]


def regimes_with_iv_lambda(lambda_mult: float = -0.95) -> List[Tuple[str, float, float, float]]:
    c_pt = 1e-3
    return [
        ("regime_II", c_pt, 0.0, 0.0),
        ("regime_III", c_pt, 0.0, 10.0),
        ("regime_IV", c_pt, float(lambda_mult) * c_pt, 0.0),
    ]


def q_name(q: float) -> str:
    return f"correlated_overlap_q{q:.2f}".replace(".", "p")


def parse_q(teacher_norm: str) -> Optional[float]:
    m = re.fullmatch(r"correlated_overlap_q([0-9]+(?:p[0-9]+)?)", str(teacher_norm))
    if m is None:
        return None
    return float(m.group(1).replace("p", "."))


def alpha_dir(seed: int, alpha: float) -> str:
    return f"seed{int(seed)}_alpha{float(alpha):.4f}"


def append_csv_locked(csv_path: Path, row_or_rows: Any) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = row_or_rows if isinstance(row_or_rows, list) else [row_or_rows]
    df = pd.DataFrame(rows)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        header = not csv_path.exists()
        df.to_csv(csv_path, mode="a", header=header, index=False)
        fcntl.flock(lf, fcntl.LOCK_UN)


def safe_torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_config(run_dir: Path) -> Dict[str, Any]:
    path = Path(run_dir) / "config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_model_and_beta(run_dir: Path, inp_dim: int = INP_DIM) -> Tuple[Any, torch.Tensor]:
    import torch
    from diagonal_network_pretrain_bg import DiagonalNet

    run_dir = Path(run_dir)
    state = safe_torch_load(run_dir / "model.pt")
    net = DiagonalNet(inp_dim, scaling=1.0, lmda=0.0, c=1e-3, c_vec=None, init_method="complex")
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        beta_hat = net.beta().detach().cpu().to(torch.float64)
    return net, beta_hat


def load_run_artifacts(run_dir: Path, inp_dim: int = INP_DIM) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    net, beta_hat = load_model_and_beta(run_dir, inp_dim=inp_dim)
    return {
        "run_dir": run_dir,
        "config": load_config(run_dir),
        "net": net,
        "beta_hat": beta_hat,
        "beta_pt": safe_torch_load(run_dir / "beta_pt.pt").detach().cpu().to(torch.float64),
        "beta_ft": safe_torch_load(run_dir / "beta_ft.pt").detach().cpu().to(torch.float64),
        "support_pt": safe_torch_load(run_dir / "support_pt.pt").detach().cpu().bool(),
        "support_ft": safe_torch_load(run_dir / "support_ft.pt").detach().cpu().bool(),
    }


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return float("nan")
    return float(x[mask].double().mean().item())


def group_metrics(beta_hat: torch.Tensor, beta_pt: torch.Tensor, beta_ft: torch.Tensor, support_pt: torch.Tensor, support_ft: torch.Tensor) -> Dict[str, float]:
    beta_hat = beta_hat.to(torch.float64)
    beta_pt = beta_pt.to(torch.float64)
    beta_ft = beta_ft.to(torch.float64)
    g0 = support_pt & support_ft
    g1 = (~support_pt) & support_ft
    g2 = support_pt & (~support_ft)
    g3 = (~support_pt) & (~support_ft)
    forget = (beta_hat - beta_pt) ** 2
    ft_err = (beta_hat - beta_ft) ** 2
    fsq = (beta_hat ** 2 - beta_pt ** 2) ** 2
    return {
        "p_FT": float(ft_err.mean().item()),
        "F_total": float(forget.mean().item()),
        "F_overlap": _masked_mean(forget, g0),
        "F_new": _masked_mean(forget, g1),
        "F_ptonly": _masked_mean(forget, g2),
        "F_none": _masked_mean(forget, g3),
        "Fsq_total": float(fsq.mean().item()),
        "Fsq_overlap": _masked_mean(fsq, g0),
        "Fsq_new": _masked_mean(fsq, g1),
        "Fsq_ptonly": _masked_mean(fsq, g2),
        "Fsq_none": _masked_mean(fsq, g3),
        "n_overlap": int(g0.sum().item()),
        "n_new": int(g1.sum().item()),
        "n_ptonly": int(g2.sum().item()),
        "n_none": int(g3.sum().item()),
    }


def compute_run_metrics(run_dir: Path, inp_dim: int = INP_DIM) -> Dict[str, Any]:
    art = load_run_artifacts(run_dir, inp_dim=inp_dim)
    return group_metrics(art["beta_hat"], art["beta_pt"], art["beta_ft"], art["support_pt"], art["support_ft"])


@dataclass(frozen=True)
class PostRunSelection:
    method: str
    teacher_norm: str
    seed: int
    post_selection: str
    run_dir: Path
    post_alpha: float
    post_p_FT: float


def run_dir_for_method(base_q_sweep: Path, base_scratch: Path, method: str, teacher_norm: str, seed: int, alpha: float) -> Path:
    if method == "scratch":
        return Path(base_scratch) / "scratch" / teacher_norm / alpha_dir(seed, alpha)
    return Path(base_q_sweep) / method / teacher_norm / alpha_dir(seed, alpha)


def select_post_run(
    *,
    method: str,
    teacher_norm: str,
    seed: int,
    post_selection: str,
    base_q_sweep: Path,
    base_scratch: Path,
    alphas: Sequence[float] = SANITY_ALPHAS,
    first_valid_threshold: float = 0.003,
    matched_alpha: float = 0.563,
) -> PostRunSelection:
    candidates: List[Tuple[float, Path, Dict[str, Any]]] = []
    for alpha in alphas:
        run_dir = run_dir_for_method(base_q_sweep, base_scratch, method, teacher_norm, seed, alpha)
        if not (run_dir / "model.pt").exists():
            continue
        metrics = compute_run_metrics(run_dir)
        candidates.append((float(alpha), run_dir, metrics))

    if not candidates:
        raise FileNotFoundError(f"No completed runs found for method={method}, teacher={teacher_norm}, seed={seed}")

    if post_selection == "first_valid":
        valid = [c for c in candidates if c[2]["p_FT"] <= float(first_valid_threshold)]
        if valid:
            alpha, run_dir, metrics = sorted(valid, key=lambda x: x[0])[0]
        else:
            alpha, run_dir, metrics = sorted(candidates, key=lambda x: x[2]["p_FT"])[0]
    elif post_selection == "matched_alpha":
        alpha, run_dir, metrics = min(candidates, key=lambda x: abs(x[0] - float(matched_alpha)))
    else:
        raise ValueError(f"Unknown post_selection={post_selection!r}")

    return PostRunSelection(
        method=method,
        teacher_norm=teacher_norm,
        seed=int(seed),
        post_selection=post_selection,
        run_dir=run_dir,
        post_alpha=float(alpha),
        post_p_FT=float(metrics["p_FT"]),
    )


def make_recovery_data(beta_target: torch.Tensor, n_train: int, n_test: int, seed: int, inp_dim: int = INP_DIM) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch

    gen_train = torch.Generator(device="cpu").manual_seed(int(seed))
    gen_test = torch.Generator(device="cpu").manual_seed(int(seed) + 1_000_003)
    x_train = torch.randn(int(n_train), int(inp_dim), generator=gen_train, dtype=torch.float64) / math.sqrt(inp_dim)
    x_test = torch.randn(int(n_test), int(inp_dim), generator=gen_test, dtype=torch.float64) / math.sqrt(inp_dim)
    beta_target = beta_target.to(torch.float64)
    return x_train, x_train @ beta_target, x_test, x_test @ beta_target


def target_beta(beta_pt: torch.Tensor, support_pt: torch.Tensor, support_ft: torch.Tensor, target: str) -> torch.Tensor:
    import torch

    if target == "full_pt":
        return beta_pt.clone().to(torch.float64)
    if target == "ptonly":
        out = torch.zeros_like(beta_pt, dtype=torch.float64)
        out[support_pt & (~support_ft)] = beta_pt[support_pt & (~support_ft)].to(torch.float64)
        return out
    raise ValueError(f"Unknown recovery target={target!r}")


def beta_error_rows(beta_rec: torch.Tensor, beta_target: torch.Tensor, beta_pt: torch.Tensor, support_pt: torch.Tensor, support_ft: torch.Tensor) -> Dict[str, float]:
    metrics = group_metrics(beta_rec, beta_pt, beta_target, support_pt, support_ft)
    err = (beta_rec.to(torch.float64) - beta_target.to(torch.float64)) ** 2
    metrics["target_mse"] = float(err.mean().item())
    return metrics
