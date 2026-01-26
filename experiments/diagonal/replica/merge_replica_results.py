import pandas as pd
from pathlib import Path

def main():
    input_dir = Path("results/replica_ptft_parallel")
    output_file = Path("experiments/diagonal/replica/replica_ptft_curves.csv")
    
    print(f"Searching for CSV files in {input_dir}...")
    files = sorted(list(input_dir.glob("replica_ptft_*.csv")))
    
    if not files:
        print("No CSV files found to merge.")
        return
        
    print(f"Found {len(files)} files. Merging...")
    
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        df_list.append(df)
        
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Save to the original target location
    merged_df.to_csv(output_file, index=False)
    
    print(f"Successfully merged {len(files)} files into {output_file}")
    print(f"Total rows: {len(merged_df)}")

if __name__ == "__main__":
    main()
