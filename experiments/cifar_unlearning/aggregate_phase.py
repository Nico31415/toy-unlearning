"""Collect the phase experiments (family, phase, phase_relearn) into CSV tables.

    python experiments/cifar_unlearning/aggregate_phase.py --out_root data/cifar_unlearning/<root> --dest <dir>

Writes family.csv, phase.csv (one row per stage-2 run; every distance, accuracy and layer read-out), phase_track.csv
(read-outs during stage 2, with --ph_track_every), phase_relearn.csv and noise.csv: the distance between family members
of different seeds, the seed-noise floor that sets eta (normalised like the phase distances).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402


def _load(path: Path) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def _flat(prefix: str, d: Optional[Dict]) -> Dict:
    return {f"{prefix}{k}": v for k, v in (d or {}).items() if not isinstance(v, (dict, list))}


def collect(out_root: Path) -> Dict[str, pd.DataFrame]:
    rows: Dict[str, List[Dict]] = {k: [] for k in ("family", "phase", "phase_track", "phase_relearn", "noise")}
    for p in sorted(out_root.glob("family/exp=*/seed=*/L=*/result.json")):
        r = _load(p)
        rows["family"].append({"exp": r["exp"], "seed": r["seed"], "L": r["L"], "loss": r["loss"],
                               "final_train_loss": r["train"]["final_train_loss"], "warning": r.get("warning", ""),
                               **_flat("", r["metrics"]), **_flat("", r.get("probes"))})
    for p in sorted(out_root.glob("phase/exp=*/seed=*/L=*/nS=*/m=*/t=*/arm=*/rescale=*/result.json")):
        r = _load(p)
        key = {"exp": r["exp"], "seed": r["seed"], "L_full": r["L_full"], "n_S": r["n_S"], "L_left": r["L_left"],
               "method": r["method"], "target": r["target"], "arm": r["arm"], "rs_mode": r["rescale"]["mode"],
               "S": r["rescale"]["S"], "loss": r["loss"]}
        rows["phase"].append({**key, "status": r["fit"]["status"], "steps": r["fit"]["steps"],
                              **{f"final_loss_{i}": v for i, v in enumerate(r["fit"]["final_losses"])},
                              **_flat("", r["metrics"]), "dir": str(p.parent.relative_to(out_root))})
        for point in r["fit"].get("tracked", []):
            rows["phase_track"].append({**key, **_flat("", point)})
        for q in sorted(p.parent.glob("relearn/n=*/result.json")):
            rows["phase_relearn"].append(_relearn_row(_load(q), key, str(p.parent.relative_to(out_root))))
    for q in sorted(out_root.glob("family/exp=*/seed=*/*/relearn/n=*/result.json")):
        r = _load(q)
        member = q.parents[2].name  # L=<L> or scratch
        rows["phase_relearn"].append(_relearn_row(r, {"exp": r["exp"], "seed": r["seed"], "loss": r["loss"]},
                                                  f"family/{member}"))
    rows["noise"] = noise_rows(out_root)
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def _relearn_row(r: Dict, key: Dict, source: str) -> Dict:
    curve = r["relearn"]["curve"]
    return {**key, "source": source, "n_RL": r["n_RL"], "status": r["relearn"]["status"],
            "steps": r["relearn"]["steps"], "A_f10_after": curve[-1].get("A_f10") if curve else None,
            "A_r_after": curve[-1].get("A_r") if curve else None}


def noise_rows(out_root: Path) -> List[Dict]:
    """For every (exp, L) with at least two seeds: the mean over seed pairs of the squared output distance on the
    forget test images, divided by the same seed's ||A(L) - A(0)||^2 (as the phase distances are)."""
    import torch
    import torch.nn.functional as F

    out = []
    for exp_dir in sorted(out_root.glob("family/exp=*")):
        exp = exp_dir.name[len("exp="):]
        logits = {}
        for p in sorted(exp_dir.glob("seed=*/L=*/logits.pt")):
            seed, L = int(p.parents[1].name[5:]), int(p.parent.name[2:])
            loss = _load(p.parent / "result.json")["loss"]
            z = torch.load(p)["f"].float()
            logits[(seed, L)] = z if loss == "mse" else F.log_softmax(z, dim=1)
        seeds = sorted({s for s, _ in logits})
        for L in sorted({L for _, L in logits}):
            have = [s for s in seeds if (s, L) in logits and (s, 0) in logits]
            for a, b in itertools.combinations(have, 2):
                norm = 0.5 * sum((logits[(s, L)] - logits[(s, 0)]).pow(2).sum(1).mean().item() for s in (a, b))
                d = (logits[(a, L)] - logits[(b, L)]).pow(2).sum(1).mean().item()
                out.append({"exp": exp, "L": L, "seed_a": a, "seed_b": b, "d2": d / norm if norm > 0 else float("nan")})
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, df in collect(Path(args.out_root)).items():
        if len(df):
            df.to_csv(dest / f"{name}.csv", index=False)
            print(f"{name}.csv: {len(df)} rows")


if __name__ == "__main__":
    main()
