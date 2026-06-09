from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from unlearning_experiment_utils import REGIMES_099, SANITY_ALPHAS, compute_run_metrics, q_name


QS = [0.25, 0.50, 0.75]
SEEDS = list(range(5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-sweep-dir", type=str, default="results/sanity_check_correlated_overlap_q_sweep")
    parser.add_argument("--scratch-dir", type=str, default="results/sanity_check_correlated_overlap_scratch")
    parser.add_argument("--output-csv", type=str, default="results/forgetting/correlated_overlap_with_scratch_metrics.csv")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected artifact is missing.")
    args = parser.parse_args()

    rows = []
    missing = []
    for regime_name, c_pt, lambda_pt, gamma_reinit in REGIMES_099:
        for q in QS:
            teacher_norm = q_name(q)
            for seed in SEEDS:
                for alpha in SANITY_ALPHAS:
                    run_dir = Path(args.q_sweep_dir) / regime_name / teacher_norm / f"seed{seed}_alpha{alpha:.4f}"
                    if not (run_dir / "model.pt").exists():
                        missing.append(str(run_dir))
                        continue
                    metrics = compute_run_metrics(run_dir)
                    rows.append(
                        {
                            "method": regime_name,
                            "regime": regime_name,
                            "teacher_norm": teacher_norm,
                            "overlap_q": float(q),
                            "seed": int(seed),
                            "alpha": float(alpha),
                            "c_pt": float(c_pt),
                            "lambda_pt": float(lambda_pt),
                            "gamma_reinit": float(gamma_reinit),
                            "run_dir": str(run_dir),
                            **metrics,
                        }
                    )

    for q in QS:
        teacher_norm = q_name(q)
        for seed in SEEDS:
            for alpha in SANITY_ALPHAS:
                run_dir = Path(args.scratch_dir) / "scratch" / teacher_norm / f"seed{seed}_alpha{alpha:.4f}"
                if not (run_dir / "model.pt").exists():
                    missing.append(str(run_dir))
                    continue
                metrics = compute_run_metrics(run_dir)
                rows.append(
                    {
                        "method": "scratch",
                        "regime": "scratch",
                        "teacher_norm": teacher_norm,
                        "overlap_q": float(q),
                        "seed": int(seed),
                        "alpha": float(alpha),
                        "c_pt": 1e-3,
                        "lambda_pt": 0.0,
                        "gamma_reinit": 0.0,
                        "run_dir": str(run_dir),
                        **metrics,
                    }
                )

    if args.strict and missing:
        raise SystemExit("Missing expected artifacts:\n" + "\n".join(missing[:50]))

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    if missing:
        print(f"Skipped {len(missing)} missing run directories")


if __name__ == "__main__":
    main()
