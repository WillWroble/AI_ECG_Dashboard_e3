import pandas as pd
import argparse
import os
import sys

def combine_csv_reports(input_dir, output_file, demo_file_path):
    """
    Combines three prediction CSVs, joins them with a primary demographics file,
    and saves a single master CSV report.
    The final column order is [filename][all predictions][demographics].
    """
    # --- 1. Load the three prediction CSV files ---
    files_to_combine = [
        'ECGdiagnosis_dashboard_final_predictions.csv',
        'LV_dysfunction_CHD_predictions.csv',
        'ACHD_mortality_predictions.csv'
    ]
    loaded_dfs = {}
    for filename in files_to_combine:
        path = os.path.join(input_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: Input file not found at {path}. It will be skipped.", file=sys.stderr)
            continue
        print(f"Loading {filename}...")
        loaded_dfs[filename] = pd.read_csv(path)

    if len(loaded_dfs) < 2:
        print("Error: Fewer than two prediction CSV files were found. Nothing to combine.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Load the primary demographics file ---
    print(f"Loading primary demographics from {demo_file_path}...")
    if not os.path.exists(demo_file_path):
        print(f"Error: Demographics file not found at {demo_file_path}", file=sys.stderr)
        sys.exit(1)
    new_demo_df = pd.read_csv(demo_file_path)


    # --- 3. Identify all unique prediction columns from the prediction files ---
    dataframes = list(loaded_dfs.values())
    common_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_columns.intersection_update(df.columns)

    all_prediction_cols = []
    for df in dataframes:
        # A prediction column is any column that isn't common to all files or is not 'filename'
        pred_cols = [col for col in df.columns if col not in common_columns or col == 'filename']
        # Remove filename to get a pure prediction list for this df
        pred_cols.remove('filename')
        all_prediction_cols.extend(pred_cols)
    
    print(f"Found a total of {len(all_prediction_cols)} unique prediction columns.")


    # --- 4. Merge the prediction files together ---
    base_df = dataframes[0]
    for i in range(1, len(dataframes)):
        other_df = dataframes[i]
        # Identify the unique prediction columns for this specific file
        other_pred_cols = ['filename'] + [col for col in other_df.columns if col not in common_columns]
        base_df = pd.merge(base_df, other_df[other_pred_cols], on='filename', how='inner')

    # Now, isolate just the filename and all the prediction columns
    predictions_df = base_df[['filename'] + all_prediction_cols]


    # --- 5. Join the combined predictions with the new demographics ---
    print("Merging predictions with new demographics file...")
    final_df = pd.merge(predictions_df, new_demo_df, on='filename', how='inner')
    
    # An 'inner' join ensures we only keep records that exist in both datasets.


    # --- 6. Ensure final column order is correct ---
    demographic_cols = [col for col in new_demo_df.columns if col != 'filename']
    final_column_order = ['filename'] + all_prediction_cols + demographic_cols
    # Reorder the dataframe to match the desired structure
    final_df = final_df[final_column_order]


    # --- 7. Save the final result ---
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    final_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully created final report: {output_file}")
    print(f"Final report has {len(final_df)} rows and {len(final_df.columns)} columns.")
    print("Columns are ordered as [filename][predictions][demographics].")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine model prediction CSVs and merge with a primary demographics file.',
        epilog='Example: python combine_reports.py --input-dir ./csv_output --demo-file ./demos/main_demo.csv --output-file ./final_report.csv'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing the individual model prediction CSV files.'
    )
    parser.add_argument(
        '--demo-file',
        type=str,
        required=True,
        help='Path to the primary demographics CSV file to merge.'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Full path for the final, combined CSV output file.'
    )
    
    args = parser.parse_args()
    combine_csv_reports(args.input_dir, args.output_file, args.demo_file)