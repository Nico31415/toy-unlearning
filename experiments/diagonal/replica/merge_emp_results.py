import pandas as pd
from pathlib import Path

def main():
    input_dir = Path("results/emp_ptft_parallel")
    output_file = Path("experiments/diagonal/replica/TESTHIGHTHRESH1e-4emp_ptft_curves.csv")
    
    print(f"Searching for empirical CSV files in {input_dir}...")
    files = sorted(list(input_dir.glob("TESTHIGHTHRESH1e-4emp_ptft_*.csv")))
    
    if not files:
        print("No empirical CSV files found to merge.")
        return
        
    print(f"Found {len(files)} files. Merging...")
    
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
        
    if not df_list:
        print("No valid dataframes to merge.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Save to the target location
    merged_df.to_csv(output_file, index=False)
    
    print(f"Successfully merged {len(files)} files into {output_file}")
    print(f"Total rows: {len(merged_df)}")

if __name__ == "__main__":
    main()
