import numpy as np
import pandas as pd
from .ECGXMLReader import ECGXMLReader
import os
import wfdb
import argparse
import glob

def convert_xml_to_dat_and_demo(input_dir, dat_output_dir, demo_output_dir):
    """
    Converts all XML files in a directory to WFDB .dat/.hea format,
    and also creates a single demographics CSV and a RECORDS.txt file.
    """
    # Ensure output directories exist
    os.makedirs(dat_output_dir, exist_ok=True)
    os.makedirs(demo_output_dir, exist_ok=True)

    # Define full output paths
    records_file_path = os.path.join(dat_output_dir, "RECORDS.txt")
    demo_csv_path = os.path.join(demo_output_dir, "demo.csv")

    # Get a list of all XML files in the input directory
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    
    all_demo_data = []
    processed_record_names = []

    for file_path in xml_files:
        try:
            print(f"Processing {os.path.basename(file_path)}...")
            ecg = ECGXMLReader(file_path, augmentLeads=True)
            
            # --- Signal Processing ---
            waveform = ecg.Waveforms[1]
            lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            new_lead_order = ['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            numpy_leads = np.stack([ecg.getLeadVoltages(lead) for lead in lead_order], axis=1)
            numpy_leads = numpy_leads.astype(np.int16)
            gain = 256.4102564102564

            # --- Data Quality Check ---
            if np.any(np.isnan(numpy_leads)) or np.any(np.sum(np.abs(numpy_leads), axis=1) == 0) or np.any(np.sum(np.abs(np.diff(numpy_leads, axis=1))) == 0):
                print(f"Skipping {os.path.basename(file_path)} due to quality issues.")
                continue

            # --- Demographics Extraction ---
            patientid = ecg.PatientDemographics.get('PatientID')
            ecg_date = ecg.TestDemographics.get('AcquisitionDate')
            ecg_time = ecg.TestDemographics.get('AcquisitionTime')
            dob = ecg.PatientDemographics.get('DateofBirth')
            gender = ecg.PatientDemographics.get('Gender')
            race = ecg.PatientDemographics.get('Race')
            sitename = ecg.TestDemographics.get('SiteName')
            location = ecg.TestDemographics.get('LocationName')
            measurements = ecg.RestingECGMeasurements
            hr = measurements.get('VentricularRate')
            pr_interval = measurements.get('PRInterval')
            qrs_duration = measurements.get('QRSDuration')
            qtc_duration = measurements.get('QTCorrected')
            paxis = measurements.get('PAxis')
            qrsaxis = measurements.get('RAxis')
            taxis = measurements.get('TAxis')
            age = float('nan')
            if dob and ecg_date:
                age = (pd.to_datetime(ecg_date) - pd.to_datetime(dob)) / pd.Timedelta(1, unit='d') / 365.25

            record_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Append data to lists for later writing
            demo_row = {
                'filename': record_name, 'patientid': patientid, 'ecg_date': ecg_date, 'ecg_time': ecg_time,
                'dob': dob, 'age': age, 'gender': gender, 'race': race, 'sitename': sitename,
                'location': location, 'hr': hr, 'pr_interval': pr_interval, 'qrs_duration': qrs_duration,
                'qtc_duration': qtc_duration, 'paxis': paxis, 'qrsaxis': qrsaxis, 'taxis': taxis
            }
            all_demo_data.append(demo_row)
            processed_record_names.append(record_name)

            # --- Write WFDB .dat and .hea files ---
            wfdb.wrsamp(
                record_name=record_name, 
                write_dir=dat_output_dir, 
                fs=int(waveform['LeadData'][0]['LeadSampleCountTotal']) / 10, 
                units=['mV'] * 12,
                sig_name=new_lead_order,
                d_signal=numpy_leads, 
                adc_gain=[gain] * 12,
                baseline=[0] * 12,
                fmt=['16'] * 12 # Use 16-bit format for int16
            )
        except Exception as e:
            print(f"Could not process file {os.path.basename(file_path)}. Error: {e}")

    # --- Write collected data to files once at the end ---
    if all_demo_data:
        df_demo = pd.DataFrame(all_demo_data)
        df_demo.to_csv(demo_csv_path, index=False)
        print(f"\nDemographics saved to {demo_csv_path}")

    if processed_record_names:
        with open(records_file_path, 'w') as f:
            for name in processed_record_names:
                f.write(name + '\n')
        print(f"Record list saved to {records_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert XML files to DAT format and extract demographics.')
    parser.add_argument('--input-dir', required=True, help='Directory containing input XML files.')
    parser.add_argument('--dat-output-dir', required=True, help='Directory to save output .dat and .hea files.')
    parser.add_argument('--demo-output-dir', required=True, help='Directory to save the output demographics CSV.')

    args = parser.parse_args()
    
    convert_xml_to_dat_and_demo(args.input_dir, args.dat_output_dir, args.demo_output_dir)