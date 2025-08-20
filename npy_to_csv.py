import numpy as np
import pandas as pd
import argparse
import os
import sys

# Renamed this function to match the import in app.py
def convert_npy_to_csv(input_file, output_file, demo_file, records_file, labels=None):
    """
    Loads a .npy prediction file, merges it with a demographics CSV, and saves as a single CSV.
    """
    for f in [input_file, demo_file, records_file]:
        if not os.path.exists(f):
            print(f"Error: Input file not found at {f}", file=sys.stderr)
            sys.exit(1)

    try:
        print(f"Loading predictions from {input_file}...")
        predictions = np.load(input_file)
        df_preds = pd.DataFrame(predictions)
        
        print(f"Loading demographics from {demo_file}...")
        df_demo = pd.read_csv(demo_file)
        df_demo = df_demo.drop(columns=['diagnosis_statement'], errors='ignore')

        print(f"Loading record order from {records_file}...")
        with open(records_file, 'r') as f:
            record_names = [os.path.splitext(line.strip())[0] for line in f if line.strip()]

        if len(record_names) != len(df_preds):
            print(f"FATAL ERROR: Row count mismatch between predictions ({len(df_preds)}) and records file ({len(record_names)}).", file=sys.stderr)
            sys.exit(1)

        column_names = labels
        if not column_names:
            column_names = [f'prediction_{i}' for i in range(df_preds.shape[1])]
        df_preds.columns = column_names
        
        df_preds['filename'] = record_names
        
        print("Merging demographics and predictions by filename...")
        df_final = pd.merge(df_demo, df_preds, on='filename', how='inner')

        print("Reordering columns to [filename, predictions, demographics]...")
        demographic_cols = [col for col in df_demo.columns if col != 'filename']
        new_column_order = ['filename'] + column_names + demographic_cols
        df_final = df_final[new_column_order]

        if len(df_final) != len(df_preds):
            print(f"Warning: {len(df_preds) - len(df_final)} predictions were dropped because their filenames were not found in the demographics CSV.")

        df_final.to_csv(output_file, index=False)
        print(f"Successfully created final report at: {output_file}")

    except Exception as e:
        print(f"An error occurred during the process: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a .npy prediction file to a labeled .csv file and merge with demographics.')
    parser.add_argument('input_file', type=str, help='Path to the input .npy file.')
    parser.add_argument('output_file', type=str, help='Path for the final merged output .csv file.')
    parser.add_argument('--demo-file', type=str, required=True, help='Path to the demographics CSV file.')
    parser.add_argument('--records-file', type=str, required=True, help='Path to the records/list file that was used to generate the H5 file.')
    parser.add_argument('--labels', nargs='+', help='A list of space-separated column labels for the CSV file.')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Call the renamed function
    convert_npy_to_csv(
        input_file=args.input_file,
        output_file=args.output_file,
        demo_file=args.demo_file,
        records_file=args.records_file,
        labels=args.labels
    )