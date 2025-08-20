import ecg_plot_JM_v12
import argparse
import matplotlib.pyplot as plt
import preprocess
import os
import read_ecg
import numpy as np
from PIL import Image
import cv2 
import sys
import random
import sys

directory = './dat_output/'
out_directory = './images_output/'
name = 'RECORDS.txt'
filename = directory + name
new_length = 2500
new_scale = 1.25

with open(filename, 'r') as file:
    # Read each line in the file
    content = file.readlines()
    for line in content:
        path = os.path.join(directory,line.replace('\n', ''))
        out_path = os.path.join(out_directory,line.replace('\n', ''))
        my_list = ['standard'] * 100
        which_order = random.choice(my_list)

        rhythm_lead = 2
        leads_all = np.arange(0,12)
        ecg, sample_rate, leads = read_ecg.read_ecg(path)
        
        ecg, sample_rate, leads = preprocess.preprocess_ecg(ecg, sample_rate, leads,
                                                                new_len=new_length,
                                                                scale=new_scale,
                                                                use_all_leads=True,
                                                                remove_baseline=True)
        # --- NEW DEBUGGING BLOCK ---
        print("\n--- Individual Lead Amplitudes After Preprocessing ---")
        # Loop through each of the 12 leads and print its max amplitude
        for i, lead_name in enumerate(leads):
            # Get the data for the i-th lead
            lead_data = ecg[i, :]
            # Calculate its maximum absolute value
            max_amp = np.max(np.abs(lead_data))
            print(f"Lead {lead_name.ljust(4)}: Max Amplitude = {max_amp}")
        print("-----------------------------------------------------\n")
        # --- END OF DEBUGGING BLOCK ---

        leads_init = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        leads_final = np.array(leads_init)[leads_all.astype(int)]
        if which_order == 'standard':
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
            ecg_plot_JM_v12.plot(ecg_final, sample_rate=sample_rate, lead_index=lead_index)
                # rm ticks
            plt.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    left=False,
                    right=False,
                    labelleft=False,
                    labelbottom=False)  # labels along the bottom edge are off
            print(f"Final array max value BEFORE plotting: {np.max(np.abs(ecg_final))}")
            ecg_plot_JM_v12.save_as_png(out_path)
