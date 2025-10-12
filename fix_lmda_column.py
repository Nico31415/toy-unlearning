#!/usr/bin/env python3
"""
Script to fix the lmda column in experiment_results.csv by extracting 
the correct lambda values from the model path names.
"""

import pandas as pd
import re
import sys

def extract_lmda_from_path(model_path):
    """
    Extract lambda value from model path string.
    Expected format: ...lmda=VALUE... where VALUE can be 0, -1e-05, -8.5e-06, etc.
    """
    # Look for lmda=VALUE pattern in the path, stopping at the next -- separator
    match = re.search(r'lmda=(.*?)--', model_path)
    if match:
        lmda_str = match.group(1)
        try:
            # Convert string to float
            return float(lmda_str)
        except ValueError:
            print(f"Warning: Could not convert '{lmda_str}' to float in path: {model_path}")
            return 0.0
    else:
        print(f"Warning: No lmda value found in path: {model_path}")
        return 0.0

def fix_lmda_column(csv_file_path):
    """
    Read CSV, extract correct lambda values from model paths, and update the lmda column.
    """
    print(f"Reading CSV file: {csv_file_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Original lmda column values (first 10): {df['lmda'].head(10).tolist()}")
    
    # Extract lambda values from model paths
    print("Extracting lambda values from model paths...")
    df['lmda'] = df['model_path'].apply(extract_lmda_from_path)
    
    print(f"Updated lmda column values (first 10): {df['lmda'].head(10).tolist()}")
    print(f"Unique lambda values found: {sorted(df['lmda'].unique())}")
    
    # Save the corrected CSV
    output_path = csv_file_path.replace('.csv', '_fixed.csv')
    df.to_csv(output_path, index=False)
    print(f"Corrected CSV saved to: {output_path}")
    
    # Also update the original file
    df.to_csv(csv_file_path, index=False)
    print(f"Original file updated: {csv_file_path}")
    
    return df

if __name__ == "__main__":
    csv_file = "/home/na658/multi-task2/experiment_results.csv"
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    try:
        df = fix_lmda_column(csv_file)
        print("\nScript completed successfully!")
        
        # Show some statistics
        print(f"\nSummary:")
        print(f"Total rows processed: {len(df)}")
        print(f"Unique lambda values: {sorted(df['lmda'].unique())}")
        print(f"Lambda value counts:")
        print(df['lmda'].value_counts().sort_index())
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
