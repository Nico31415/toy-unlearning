import pandas as pd
from pathlib import Path
import numpy as np

def _merge_glob_to_csv(input_dir: Path, pattern: str, output_file: Path) -> None:
    print(f"Searching for empirical CSV files in {input_dir} with pattern {pattern!r}...")
    files = sorted(list(input_dir.glob(pattern)))

    if not files:
        print(f"No empirical CSV files found for pattern {pattern!r}.")
        return

    print(f"Found {len(files)} files. Merging...")

    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not df_list:
        print("No valid dataframes to merge.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    print(f"Successfully merged {len(files)} files into {output_file}")
    print(f"Total rows: {len(merged_df)}")


def merge_exp_master(exp_num: int, input_dir: Path) -> None:
    output_file = Path(f"experiments/diagonal/replica/EXPERIMENT{exp_num}emp_ptft_curves.csv")
    _merge_glob_to_csv(input_dir, f"EXPERIMENT{exp_num}*.csv", output_file)


def merge_remaining_experiments_exp1_omega_ext(input_dir: Path) -> None:
    """
    Merge ONLY the Exp1 omega-extension runs (the "remaining" ones), into:
      experiments/diagonal/replica/REMAININGEXPERIMENTS_emp_ptft_curves.csv
    """
    output_file = Path("experiments/diagonal/replica/REMAININGEXPERIMENTS_emp_ptft_curves.csv")

    # Candidate files: Exp1 sweeps (includes omega=0.5 sweeps too, which we'll filter out)
    files = sorted(list(input_dir.glob("EXPERIMENT1_sweep_*.csv")))
    print(f"Searching for Exp1 sweep CSV files in {input_dir} for omega-extension merge...")
    print(f"Found {len(files)} candidate sweep files.")

    if not files:
        print("No Exp1 sweep CSV files found to merge for omega-extension.")
        return

    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not df_list:
        print("No valid dataframes to merge for omega-extension.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)

    # Ensure numeric + required columns exist
    need = ["omega", "c_pt", "lambda_pt", "gamma_reinit"]
    missing_cols = [c for c in need if c not in merged_df.columns]
    if missing_cols:
        raise KeyError(f"Omega-extension merge requires columns {missing_cols} but they are missing.")
    for c in need:
        merged_df[c] = pd.to_numeric(merged_df[c], errors="coerce")

    omega01 = merged_df["omega"].isin([0.0, 1.0])

    # Exp1 omega-extension definitions (match the worker)
    c_sweep = (
        omega01
        & np.isclose(merged_df["lambda_pt"], 0.0, atol=1e-12)
        & np.isclose(merged_df["gamma_reinit"], 0.0, atol=1e-12)
        & (np.isclose(merged_df["c_pt"], 1e-6, atol=1e-15) | np.isclose(merged_df["c_pt"], 1.0, atol=1e-12))
    )
    lambda_sweep = (
        omega01
        & np.isclose(merged_df["c_pt"], 1e-3, atol=1e-15)
        & np.isclose(merged_df["gamma_reinit"], 0.0, atol=1e-12)
        & (
            np.isclose(merged_df["lambda_pt"], -1e-3, atol=1e-15)
            | np.isclose(merged_df["lambda_pt"], -0.99e-3, atol=1e-15)
            | np.isclose(merged_df["lambda_pt"], 0.99e-3, atol=1e-15)
        )
    )
    gamma_sweep = (
        omega01
        & np.isclose(merged_df["c_pt"], 1e-3, atol=1e-15)
        & np.isclose(merged_df["lambda_pt"], 0.0, atol=1e-12)
        & (np.isclose(merged_df["gamma_reinit"], 1.0, atol=1e-12) | np.isclose(merged_df["gamma_reinit"], 10.0, atol=1e-12))
    )

    out_df = merged_df[c_sweep | lambda_sweep | gamma_sweep].copy()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False)
    print(f"Saved omega-extension merged subset to {output_file}")
    print(f"Rows in subset: {len(out_df)}")


def main():
    input_dir = Path("results/emp_ptft_parallel")

    # Keep your standard merged CSVs up to date
    for exp_num in [1, 2, 3]:
        merge_exp_master(exp_num, input_dir)
        print("-" * 30)

    # Write a separate CSV for the Exp1 omega-extension "remaining experiments"
    merge_remaining_experiments_exp1_omega_ext(input_dir)

if __name__ == "__main__":
    main()
