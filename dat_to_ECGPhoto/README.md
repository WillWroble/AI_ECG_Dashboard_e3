## Folder Structure
ecg-preprocessing holds the scripts to convert .dat/.dea ECG files to a photo of an ECG
sample_data holds sample data of .dat/.dea ECG files, and the script outputs when running python create_images_all_12lead.py

## Script Inputs
Each ECG will have to be converted from an XML file to a .dat/.dea file. The event ID of the ECG corresponds to the name of each .dat/.dea file. These .dat/.dea files will then be used as inputs into the create_images_all_12lead.py script to convert ECG digital waveforms into a printout. For now, the create_images_all_12lead.py is given the directory (in this case '../sample_data/'), and the ECGs to convert (from 'RECORDS.txt'). This will have to be adjusted for real-time implementation.

## Script Outputs
Pictures of ECGs for each event ID are printed in the '../sample_data/' folder. These are labeled by the event ID followed by .png
