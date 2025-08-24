import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Your required imports
from dat_to_ECGPhoto.ecg_preprocessing import ecg_plot_JM_v12
from dat_to_ECGPhoto.ecg_preprocessing import preprocess
from dat_to_ECGPhoto.ecg_preprocessing import read_ecg

def create_images_from_dat(dat_input_dir, png_output_dir):
    """
    This function contains your original, unchanged image generation logic.
    It takes directory paths as arguments instead of using hardcoded ones.
    """
    # ### YOUR ORIGINAL LOGIC STARTS HERE - UNCHANGED ###

    records_file_path = os.path.join(dat_input_dir, 'RECORDS.txt')
    new_length = 2500
    new_scale = 1.25

    # Check if the RECORDS file exists before proceeding
    if not os.path.exists(records_file_path):
        print(f"Warning: RECORDS.txt not found in {dat_input_dir}. Skipping image generation.")
        return

    with open(records_file_path, 'r') as file:
        content = file.readlines()
    
    print(f"--- Generating {len(content)} PNG images ---")
    for line in content:
        # Construct paths dynamically
        record_name = line.strip()
        if not record_name:
            continue
        
        path = os.path.join(dat_input_dir, record_name)
        out_path = os.path.join(png_output_dir, os.path.splitext(record_name)[0])

        # This entire processing block is directly from your script
        rhythm_lead = 2
        leads_all = np.arange(0,12)
        ecg, sample_rate, leads = read_ecg.read_ecg(path)
        
        ecg, sample_rate, leads = preprocess.preprocess_ecg(
            ecg, sample_rate, leads,
            new_len=new_length,
            scale=new_scale,
            use_all_leads=True,
            remove_baseline=True
        )

        leads_init = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        leads_final = np.array(leads_init)[leads_all.astype(int)]
        
        ecg_rhythm1 = ecg[rhythm_lead:(rhythm_lead+1),0:int(new_length/4)]
        ecg_rhythm2 = ecg[rhythm_lead:(rhythm_lead+1),int(new_length/4):int(2*new_length/4)]
        ecg_rhythm3 = ecg[rhythm_lead:(rhythm_lead+1),int(2*new_length/4):int(3*new_length/4)]
        ecg_rhythm4 = ecg[rhythm_lead:(rhythm_lead+1),int(3*new_length/4):int(4*new_length/4)]
        ecg_col1 = ecg[leads_all[0:3],0:int(new_length/4)]
        ecg_col2 = ecg[leads_all[3:6],int(new_length/4):int(2*new_length/4)]
        ecg_col3 = ecg[leads_all[6:9],int(2*new_length/4):int(3*new_length/4)]
        ecg_col4 = ecg[leads_all[9:12],int(3*new_length/4):int(4*new_length/4)]
        ecg_final = np.concatenate((ecg_col1, ecg_rhythm1, ecg_col2, ecg_rhythm2, ecg_col3, ecg_rhythm3, ecg_col4, ecg_rhythm4))
        
        lead_index = [leads_final[0], leads_final[1], leads_final[2], leads_init[rhythm_lead], leads_final[3], leads_final[4], leads_final[5], '', leads_final[6], leads_final[7], leads_final[8], '', leads_final[9], leads_final[10], leads_final[11], '']
        
        plt.figure()
        ecg_plot_JM_v12.plot(ecg_final, sample_rate=sample_rate, lead_index=lead_index)

        plt.tick_params(
            axis='both', which='both', bottom=False, top=False, left=False,
            right=False, labelleft=False, labelbottom=False
        )
        
        ecg_plot_JM_v12.save_as_png(out_path)
        plt.close()

    # ### YOUR ORIGINAL LOGIC ENDS HERE ###