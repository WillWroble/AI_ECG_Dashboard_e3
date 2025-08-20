import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import sys
import traceback
import socket
import webbrowser
from flask import Flask, render_template, request, jsonify
from waitress import serve
try:
    from flask import Request as FlaskRequest  # Flask >= 1.x often exposes this
except Exception:
    from flask.wrappers import Request as FlaskRequest  # fallback
# --- Direct Function Imports ---
from XML_to_dat.xml_to_dat import convert_xml_to_dat_and_demo
from ecg_preprocess.generate_h5 import main as generate_h5_main
from Run_Predictions.predict import predict as run_predictions
from npy_to_csv import convert_npy_to_csv

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- App Setup ---
app = Flask(__name__)
# Set a 1 GB upload limit for both Flask and Waitress
# Size & multipart limits
MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024
MAX_FORM_MEM    = 2  * 1024 * 1024 * 1024
MAX_FORM_PARTS  = 200_000

app.config["MAX_CONTENT_LENGTH"]   = MAX_UPLOAD_SIZE
app.config["MAX_FORM_MEMORY_SIZE"] = MAX_FORM_MEM
app.config["MAX_FORM_PARTS"]       = MAX_FORM_PARTS

class PatchedRequest(FlaskRequest):
    max_form_memory_size = MAX_FORM_MEM
    max_form_parts       = MAX_FORM_PARTS

app.request_class = PatchedRequest



# --- Model Configuration ---
MODELS = [
    'ACHD_mortality.hdf5',
    'ECG_Diagnosis.hdf5',
    'LV_dysfunction_CHD.hdf5',
    'final_diastolic.hdf5',
    'chd_diagnosis.hdf5',
]
MODEL_LABELS = {
    'ACHD_mortality.hdf5': ['risk_lte_30_prob', 'risk_lte_35_prob', 'risk_lte_40_prob', 'risk_lte_45_prob', 'risk_lte_50_prob'],
    'ECG_Diagnosis.hdf5': ['LVEF_prob', 'age_echo_prob', 'tof_prob', 'cardiomyopathy_prob', 'asd_prob', 'cavc_prob', 'coa_prob', 'dorv_prob', 'dtga_prob', 'ebstein_prob', 'hlhs_prob', 'ltga_prob', 'pa_prob', 'tapvr_prob', 'triatresia_prob', 'truncus_prob', 'vsd_prob', 'dextrocardia_prob', 'other_prob_1', 'other_prob_2', 'other_prob_3', 'other_prob_4', 'other_prob_5'],
    'LV_dysfunction_CHD.hdf5': ['LVEF_lte_30_prob', 'LVEF_lte_35_prob', 'LVEF_lte_40_prob', 'LVEF_lte_45_prob', 'LVEF_lte_50_prob'],
    'final_diastolic.hdf5': ['pcwp15', 'pcwp18', 'pcwp21', 'pcwp24', 'pcwp27'],
    'chd_diagnosis.hdf5': ['asd', 'dextrocardia', 'ltga', 'pda', 'cavc', 'coa', 'dilv', 'dirv', 'dolv', 'dorv', 'dtga', 'ebstein', 'hlhs', 'iaa', 'pulmatresia', 'tapvr', 'tof', 'triatresia', 'truncus', 'criticalps', 'criticalas', 'cardiomyopathy', 'hcm', 'dcm', 'critical', 'any']
}

# --- Main Pipeline Logic ---
def execute_pipeline(session_dir, all_xml_files, selected_model_basename):
    """
    Runs the full pipeline by processing uploaded XML files in batches to conserve memory.
    """
    batch_size = 1000  # Process 50 files at a time
    batch_results_dfs = []
    model_path = resource_path(os.path.join('models', selected_model_basename))
    model_name = os.path.splitext(selected_model_basename)[0]

    print(f"Found {len(all_xml_files)} files. Processing in batches of {batch_size}.")

    for i in range(0, len(all_xml_files), batch_size):
        batch_files = all_xml_files[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"\n--- Processing Batch {batch_num} ({len(batch_files)} files) ---")

        # Create a temporary directory for this specific batch
        batch_session_dir = tempfile.mkdtemp(dir=session_dir)
        
        # Define paths for this batch
        batch_xml_input_dir = os.path.join(batch_session_dir, 'XML_input')
        batch_dat_output_dir = os.path.join(batch_session_dir, 'dat_output')
        batch_h5_output_dir = os.path.join(batch_session_dir, 'h5_output')
        batch_prediction_output_dir = os.path.join(batch_session_dir, 'predictions_output')
        batch_demo_output_dir = os.path.join(batch_session_dir, 'demographics')
        
        for d in [batch_xml_input_dir, batch_dat_output_dir, batch_h5_output_dir, batch_prediction_output_dir, batch_demo_output_dir]:
            os.makedirs(d, exist_ok=True)
            
        # Copy the batch of files to the temporary batch input directory
        for f_path in batch_files:
            shutil.copy(f_path, batch_xml_input_dir)
            
        # Define file paths for the pipeline stages
        demographics_csv = os.path.join(batch_demo_output_dir, 'demo.csv')
        master_h5_file = os.path.join(batch_h5_output_dir, 'ECGs.h5')
        records_file = os.path.join(batch_dat_output_dir, 'RECORDS.txt')
        prediction_npy_file = os.path.join(batch_prediction_output_dir, f"predictions__{model_name}.npy")
        batch_final_csv = os.path.join(batch_prediction_output_dir, f"predictions__{model_name}.csv")

        # --- Execute Pipeline Stages for the Batch ---
        convert_xml_to_dat_and_demo(input_dir=batch_xml_input_dir, dat_output_dir=batch_dat_output_dir, demo_output_dir=batch_demo_output_dir)
        generate_h5_main([records_file, master_h5_file, '--root_dir', batch_dat_output_dir, '--use_all_leads', '--remove_baseline', '--new_len', '2048', '--new_freq', '250'])
        run_predictions(input_file=master_h5_file, model_file=model_path, output_file=prediction_npy_file)
        labels = MODEL_LABELS.get(selected_model_basename, [])
        convert_npy_to_csv(input_file=prediction_npy_file, output_file=batch_final_csv, demo_file=demographics_csv, records_file=records_file, labels=labels)
        
        # Store the result of the batch and clean up the batch directory
        if os.path.exists(batch_final_csv):
            batch_results_dfs.append(pd.read_csv(batch_final_csv))
        shutil.rmtree(batch_session_dir)

    # Combine results from all batches
    if not batch_results_dfs:
        raise Exception("Pipeline did not generate any results.")
    
    final_df = pd.concat(batch_results_dfs, ignore_index=True)
    final_csv_path = os.path.join(session_dir, "final_predictions.csv")
    final_df.to_csv(final_csv_path, index=False)
    
    return final_csv_path

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', models=MODELS)

@app.route('/run_pipeline', methods=['POST'])
def handle_request():
    session_dir = tempfile.mkdtemp()
    try:
        xml_files = request.files.getlist('xml_files')
        selected_model = request.form.get('model')
        if not xml_files or not xml_files[0].filename or not selected_model:
            return jsonify({'error': 'Missing XML files or model selection.'}), 400
            
        xml_input_dir = os.path.join(session_dir, 'XML_input_master')
        os.makedirs(xml_input_dir)
        xml_file_paths = []
        for f in xml_files:
            path = os.path.join(xml_input_dir, os.path.basename(f.filename))
            f.save(path)
            xml_file_paths.append(path)

        output_csv_path = execute_pipeline(session_dir, xml_file_paths, selected_model)
        
        df = pd.read_csv(output_csv_path)
        df = df.replace({np.nan: None}) 
        column_order = df.columns.tolist()
        table_data = df.to_dict(orient='records')
        with open(output_csv_path, 'r') as f:
            csv_string = f.read()

        return jsonify({'table': table_data, 'csv': csv_string, 'columns': column_order})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred in the pipeline: {str(e)}'}), 500
    finally:
        shutil.rmtree(session_dir)

if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        print(f"Starting server on {url}")
        webbrowser.open(url)
        serve(
            app,
            sockets=[s],
            max_request_body_size=MAX_UPLOAD_SIZE
        )
