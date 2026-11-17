"""Collect the modular-arithmetic runs into CSV tables (same layout as the CIFAR phase tables).

    python experiments/modular_unlearning/aggregate.py --out_root data/modular_unlearning/<root> --dest <dir>

family.csv (one row per A(L), with its key frequencies and their realised overlap with the partner task), phase.csv,
phase_relearn.csv and noise.csv (distance between seeds of the same family member, the floor that sets eta).
Realised overlap of a member: the share of its key frequencies whose removal from the embedding raises the partner
task's loss by more than twice the median rise over all frequencies.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _load(p):
    with open(p) as fh:
        return json.load(fh)


def _flat(d, prefix=""):
    return {f"{prefix}{k}": v for k, v in (d or {}).items() if not isinstance(v, (dict, list))}


def collect(root: Path):
    rows = {k: [] for k in ("family", "phase", "phase_relearn", "noise")}
    for p in sorted(root.glob("family/exp=*/seed=*/L=*/result.json")):
        r = _load(p)
        keys, prise = r["key_freqs"], np.asarray(r["partner_rises"])
        thr = 2 * np.median(np.abs(prise))
        overlap = float(np.mean([prise[k - 1] > thr for k in keys])) if len(keys) else np.nan
        last = r["fit"]["history"][-1] if r["fit"]["history"] else {}
        rows["family"].append({"exp": r["exp"], "seed": r["seed"], "L": r["L"], "status": r["fit"]["status"],
                               "steps": r["fit"]["steps"], "train_loss": last.get("train_loss"),
                               "key_freqs": " ".join(map(str, keys)), "overlap": overlap,
                               **_flat(r["metrics"]), **_flat(r["probes"])})
    for p in sorted(root.glob("phase/exp=*/seed=*/L=*/nS=*/m=*/t=*/arm=*/rescale=*/result.json")):
        r = _load(p)
        key = {k: r[k] for k in ("exp", "seed", "L_full", "n_S", "L_left", "method", "target", "arm", "S", "rs_mode")}
        rows["phase"].append({**key, "status": r["fit"]["status"], "steps": r["fit"]["steps"],
                              **_flat(r["metrics"]), "dir": str(p.parent.relative_to(root))})
        for q in sorted(p.parent.glob("relearn/n=*/result.json")):
            rows["phase_relearn"].append(_rl(_load(q), str(p.parent.relative_to(root))))
    for q in sorted(root.glob("family/exp=*/seed=*/*/relearn/n=*/result.json")):
        rows["phase_relearn"].append(_rl(_load(q), f"family/{q.parents[2].name}"))
    rows["noise"] = noise(root)
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def _rl(r, source):
    h = r["relearn"]["history"]
    return {"exp": r["exp"], "seed": r["seed"], "source": source, "n_RL": r["n_RL"],
            "status": r["relearn"]["status"], "steps": r["relearn"]["steps"],
            "steps_to_acc": r["relearn"]["steps_to_acc"], "acc_forget_after": h[-1]["acc_forget"] if h else None}


def noise(root: Path):
    import torch

    out = []
    for exp_dir in sorted(root.glob("family/exp=*")):
        z = {}
        for p in exp_dir.glob("seed=*/L=*/logits.pt"):
            r = _load(p.parent / "result.json")
            lo = torch.load(p)["f"].float()
            p_f = int(r["exp"].split("-")[0][1:])
            lo = lo[:, :p_f]
            z[(r["seed"], r["L"])] = lo if r["settings"]["loss"] == "mse" else torch.log_softmax(lo, 1)
        for L in sorted({L for _, L in z}):
            seeds = sorted(s for s, l in z if l == L and (s, 0) in z)
            for a, b in itertools.combinations(seeds, 2):
                norm = 0.5 * sum((z[(s, L)] - z[(s, 0)]).pow(2).sum(1).mean().item() for s in (a, b))
                d = (z[(a, L)] - z[(b, L)]).pow(2).sum(1).mean().item()
                out.append({"exp": exp_dir.name[4:], "L": L, "seed_a": a, "seed_b": b,
                            "d2": d / norm if norm > 0 else float("nan")})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, df in collect(Path(a.out_root)).items():
        if len(df):
            df.to_csv(dest / f"{name}.csv", index=False)
            print(f"{name}.csv: {len(df)} rows")


if __name__ == "__main__":
    main()
