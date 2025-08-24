"""
ECG Processing and Prediction Pipeline

This script orchestrates the conversion of a directory of XML ECG files into 
prediction CSVs and a set of PNG images. It is designed to be run from the 
command line and is configured to run a predefined set of models.

The pipeline will generate one final CSV per model in the output directory.

Prerequisites:
- All required custom modules must be in the Python path or the same directory:
  - XML_to_dat.xml_to_dat
  - ecg_preprocess.generate_h5
  - Run_Predictions.predict
  - npy_to_csv
  - dat_to_ECGPhoto.ecg_preprocessing.create_images_all_12lead

Example Usage:
python pipeline.py \
    --input-dir ./xml_input \
    --output-csv-dir ./csv_output \
    --output-png-dir ./png_output
"""

import os
import sys
import argparse
import tempfile
import shutil
import pandas as pd
import traceback

# --- Import custom modules ---
try:
    from XML_to_dat.xml_to_dat import convert_xml_to_dat_and_demo
    from ecg_preprocess.generate_h5 import main as generate_h5_main
    from Run_Predictions.predict import predict as run_predictions
    from npy_to_csv import convert_npy_to_csv
    # Updated import path for the image generation function
    from dat_to_ECGPhoto.ecg_preprocessing.create_images_all_12lead import create_images_from_dat
except ImportError as e:
    print(f"Error: A required module could not be imported: {e}")
    print("Please ensure all custom scripts are in the same directory or your PYTHONPATH.")
    sys.exit(1)

# --- Helper function for file paths ---
def resource_path(relative_path):
    """ Get absolute path to a resource, works for dev and for PyInstaller. """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Model Configuration ---
# The specific models to be run are now hardcoded here.
MODELS_TO_RUN = [
    'ECGdiagnosis_dashboard_final.hdf5',
    'LV_dysfunction_CHD.hdf5',
    'ACHD_mortality.hdf5'
]

MODEL_LABELS = {
    'ACHD_mortality.hdf5': ['risk_lte_30_prob', 'risk_lte_35_prob', 'risk_lte_40_prob', 'risk_lte_45_prob', 'risk_lte_50_prob'],
    'LV_dysfunction_CHD.hdf5': ['LVEF_lte_30_prob', 'LVEF_lte_35_prob', 'LVEF_lte_40_prob', 'LVEF_lte_45_prob', 'LVEF_lte_50_prob'],
    'ECGdiagnosis_dashboard_final.hdf5': ['Incomplete RBBB','Nonspecific ST/T Wave Changes','Atrial Fibrillation','Atrial Flutter','Ectopic Atrial Tachycardia','Complete Heart Block','Concern for Ischemia','Pericarditis','Prolonged QTc','Sinus Bradycardia','Sinus Tachycardia','Supraventricular Tachycardia','WPW','Inadequate Study','Right Ventricular Hypertrophy','Left Ventricular Hypertrophy','Lateral T Wave Inversion','Complete RBBB','Accelerated Junctional Rhythm','AIVR','Atrial Pacing','Superior Axis Deviation','Left Axis Deviation','Right Axis Deviation','Atrial Enlargement','Dextrocardia','Diminished LV Forces','Diminished RV Forces','Dual Chamber Pacing','Ectopic Atrial Rhythm','First Degree Block','IVCD','Junctional Escape','Left Hemiblock','Low Voltages','Wenckebach','Mobitz Type 2','Atrial Premature Contraction','Ventricular Premature Contraction','Short PR Interval','Ventricular Pacing','Ventricular Tachycardia','Normal ECG','Abnormal ECG','High Grade Abnormality']

}

# --- Main Pipeline Logic ---
def execute_pipeline(all_xml_files, output_csv_dir, output_png_dir):
    """
    Runs the full conversion and prediction pipeline for a hardcoded list of models.
    """
    session_dir = tempfile.mkdtemp()
    print(f"INFO: Created temporary working directory: {session_dir}")

    # Dictionary to store lists of dataframes for each model's results across batches
    model_results = {os.path.splitext(m)[0]: [] for m in MODELS_TO_RUN}

    try:
        batch_size = 1000
        print(f"INFO: Found {len(all_xml_files)} XML files. Processing in batches of {batch_size}.")

        for i in range(0, len(all_xml_files), batch_size):
            batch_files = all_xml_files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            print(f"\n--- Processing Batch {batch_num} ({len(batch_files)} files) ---")

            # Define and create intermediate directories for the batch
            batch_xml_input_dir = os.path.join(session_dir, 'XML_input')
            batch_dat_output_dir = os.path.join(session_dir, 'dat_output')
            batch_h5_output_dir = os.path.join(session_dir, 'h5_output')
            batch_prediction_output_dir = os.path.join(session_dir, 'predictions_output')
            batch_demo_output_dir = os.path.join(session_dir, 'demographics')
            
            for d in [batch_xml_input_dir, batch_dat_output_dir, batch_h5_output_dir, batch_prediction_output_dir, batch_demo_output_dir]:
                if os.path.exists(d): shutil.rmtree(d)
                os.makedirs(d)

            for f_path in batch_files:
                shutil.copy(f_path, batch_xml_input_dir)
            
            demographics_csv = os.path.join(batch_demo_output_dir, 'demo.csv')
            master_h5_file = os.path.join(batch_h5_output_dir, 'ECGs.h5')
            records_file = os.path.join(batch_dat_output_dir, 'RECORDS.txt')

            # --- Model-Independent Steps (Run Once Per Batch) ---
            print("STEP 1: Converting XML to DAT format...")
            convert_xml_to_dat_and_demo(input_dir=batch_xml_input_dir, dat_output_dir=batch_dat_output_dir, demo_output_dir=batch_demo_output_dir)
            
            print("STEP 2: Generating PNG images from DAT files...")
            create_images_from_dat(dat_input_dir=batch_dat_output_dir, png_output_dir=output_png_dir)

            print("STEP 3: Generating H5 file from DAT files...")
            generate_h5_main([records_file, master_h5_file, '--root_dir', batch_dat_output_dir, '--use_all_leads', '--remove_baseline', '--new_len', '2048', '--new_freq', '250'])

            # --- Model-Dependent Steps (Loop Through Each Model) ---
            for model_basename in MODELS_TO_RUN:
                model_name = os.path.splitext(model_basename)[0]
                print(f"\n-- Running predictions for model: {model_name} --")
                model_path = resource_path(os.path.join('models', model_basename))
                
                prediction_npy_file = os.path.join(batch_prediction_output_dir, f"predictions__{model_name}.npy")
                batch_final_csv = os.path.join(batch_prediction_output_dir, f"predictions__{model_name}.csv")

                print(f"STEP 4: Running predictions...")
                run_predictions(input_file=master_h5_file, model_file=model_path, output_file=prediction_npy_file)

                print(f"STEP 5: Converting NPY predictions to CSV...")
                labels = MODEL_LABELS.get(model_basename, [])
                convert_npy_to_csv(input_file=prediction_npy_file, output_file=batch_final_csv, demo_file=demographics_csv, records_file=records_file, labels=labels)
                
                if os.path.exists(batch_final_csv):
                    model_results[model_name].append(pd.read_csv(batch_final_csv))
                    print(f"Batch predictions for {model_name} completed successfully.")
                else:
                    print(f"WARNING: No CSV output was generated for {model_name} in batch {batch_num}.")

        # --- Combine and Save Final Results ---
        print("\n--- Combining results from all batches ---")
        for model_name, df_parts in model_results.items():
            if df_parts:
                final_df = pd.concat(df_parts, ignore_index=True)
                final_csv_path = os.path.join(output_csv_dir, f"{model_name}_predictions.csv")
                final_df.to_csv(final_csv_path, index=False)
                print(f"Final predictions for {model_name} saved to: {final_csv_path}")
            else:
                print(f"No results were generated for model {model_name}.")

    except Exception:
        print(f"\n--- PIPELINE FAILED ---")
        traceback.print_exc()
    finally:
        print(f"INFO: Cleaning up temporary directory: {session_dir}")
        shutil.rmtree(session_dir)

# --- Main execution block ---
def main():
    parser = argparse.ArgumentParser(
        description="ECG XML to Prediction CSV and PNG Pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the input XML files.")
    parser.add_argument("--output-csv-dir", type=str, required=True, help="Directory where the final prediction CSVs will be saved.")
    parser.add_argument("--output-png-dir", type=str, required=True, help="Directory where the generated ECG PNG images will be saved.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        sys.exit(1)

    os.makedirs(args.output_csv_dir, exist_ok=True)
    os.makedirs(args.output_png_dir, exist_ok=True)

    all_xml_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith('.xml')]
    if not all_xml_files:
        print(f"Error: No XML files found in '{args.input_dir}'")
        sys.exit(1)

    execute_pipeline(
        all_xml_files=all_xml_files,
        output_csv_dir=args.output_csv_dir,
        output_png_dir=args.output_png_dir
    )
    
    print("\nPipeline execution finished.")

if __name__ == '__main__':
    main()
