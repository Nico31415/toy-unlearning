"""Collect every result.json into flat CSV tables.

    python experiments/cifar_unlearning/aggregate.py [--loss ce] [--out_root ...] [--dest ...]

Writes pretrain.csv, retrain.csv, unlearn.csv, references.csv, curves.csv and
representation.csv (one row per model and stage: CKA and k-NN) to
data/processed/cifar_unlearning/<loss>/ by default.
"""
from __future__ import annotations

import argparse
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


def _prefixed(metrics: Optional[Dict], prefix: str) -> Dict:
    return {f"{prefix}{k}": v for k, v in (metrics or {}).items()}


def _curve_rows(curve: List[Dict], keys: Dict) -> List[Dict]:
    return [{**keys, **point} for point in curve]


def collect(out_root: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, List[Dict]] = {name: [] for name in
                                     ("pretrain", "retrain", "unlearn", "references", "curves", "representation")}

    for path in sorted(out_root.glob("pretrain/seed=*/result.json")):
        r = _load(path)
        tables["pretrain"].append({"seed": r["seed"], "test_acc": r["test_acc"],
                                   "final_train_loss": r["train"]["final_train_loss"],
                                   "seconds": r["train"]["seconds"]})

    for path in sorted(out_root.glob("retrain/exp=*/seed=*/result.json")):
        r = _load(path)
        tables["retrain"].append({"exp": r["exp"], "seed": r["seed"],
                                  "final_train_loss": r["train"]["final_train_loss"],
                                  "seconds": r["train"]["seconds"], **r["metrics"]})

    for path in sorted(out_root.glob("unlearn/exp=*/seed=*/mu_f=*/rescale=*/f=*/result.json")):
        r = _load(path)
        keys = {"exp": r["exp"], "seed": r["seed"], "kind": r["kind"], "mu_f": r["mu_f"],
                "rescale": r["rescale"]["tag"], "fraction": r["fraction"], "n_r": r["n_r"], "n_f": r["n_f"]}
        ul, rl = r["unlearn"], r["relearn"]
        tables["unlearn"].append({
            **keys, "ul_status": ul["status"], "ul_steps": ul["steps"],
            "ul_forget_reached_step": ul.get("forget_reached_step"), "ul_rollback_step": ul.get("rollback_step"),
            "ul_train_retain_loss": ul["final_retain_loss"], "ul_train_forget_acc": ul["final_forget_acc"],
            **_prefixed(r["after_unlearn"], "ul_"),
            "rl_status": rl["status"], "rl_steps": rl["steps"], **_prefixed(r["after_relearn"], "rl_")})
        tables["curves"] += _curve_rows(rl["curve"], {**keys, "source": "unlearned"})

    befores = {}
    for path in sorted(out_root.glob("references/exp=*/seed=*/before.json")):
        r = _load(path)
        befores[(r["exp"], r["seed"])] = r["metrics"]
    for path in sorted(out_root.glob("references/exp=*/seed=*/model=*/f=*/result.json")):
        r = _load(path)
        before = befores.get((r["exp"], r["seed"]), {}).get(r["model"])
        keys = {"exp": r["exp"], "seed": r["seed"], "kind": r["kind"], "model": r["model"],
                "fraction": r["fraction"], "n_f": r["n_f"]}
        rl = r["relearn"]
        tables["references"].append({**keys, **_prefixed(before, "before_"), "rl_status": rl["status"],
                                     "rl_steps": rl["steps"], **_prefixed(r["after_relearn"], "rl_")})
        tables["curves"] += _curve_rows(rl["curve"], {**keys, "source": r["model"]})

    for path in sorted(out_root.glob("references/exp=*/seed=*/representation_before.json")):
        r = _load(path)
        for model, metrics in r["metrics"].items():
            tables["representation"].append({"exp": r["exp"], "seed": r["seed"], "source": model,
                                             "when": "before_relearn", **metrics})
    for path in sorted(out_root.glob("unlearn/exp=*/seed=*/mu_f=*/rescale=*/f=*/representation.json")):
        r = _load(path)
        keys = {"exp": r["exp"], "seed": r["seed"], "source": "unlearned", "mu_f": r["mu_f"],
                "rescale": r["rescale"], "fraction": r["fraction"]}
        for when in ("after_unlearn", "after_relearn"):
            if when in r:
                tables["representation"].append({**keys, "when": when, **r[when]})
    for path in sorted(out_root.glob("references/exp=*/seed=*/model=*/f=*/representation.json")):
        r = _load(path)
        tables["representation"].append({"exp": r["exp"], "seed": r["seed"], "source": r["source"],
                                         "fraction": r["fraction"], "when": "after_relearn", **r["after_relearn"]})

    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss", default="ce")
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--dest", default=None)
    args = parser.parse_args(argv)
    out_root = Path(args.out_root or f"data/cifar_unlearning/{args.loss}")
    dest = Path(args.dest or f"data/processed/cifar_unlearning/{args.loss}")
    dest.mkdir(parents=True, exist_ok=True)
    for name, frame in collect(out_root).items():
        frame.to_csv(dest / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(frame)} rows")


if __name__ == "__main__":
    main()
