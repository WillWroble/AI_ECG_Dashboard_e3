import numpy as np
import pandas as pd
import xmltodict
from ECGXMLReader import ECGXMLReader
import matplotlib.pyplot as plt
import base64
import array
import os
import wfdb
import h5py
import xml.etree.ElementTree as ET
from datetime import datetime
import glob

input_path = "./XML_input/"
output_path = "./demographics/"
output_csv_file = os.path.join(output_path, "demo.csv")


# Clear the file if it exists before starting the loop
try:
    os.remove(output_csv_file)
    print(f"Cleared existing file: {output_csv_file}")
except FileNotFoundError:
    pass
fileList = glob.glob(input_path + "*.xml")

file_num = -1
for file_name in fileList:
    print(file_name)
    ecg = ECGXMLReader(file_name)
    file_num += 1
    try:
        patientid, ecg_date, ecg_time = ecg.PatientDemographics['PatientID'], ecg.TestDemographics['AcquisitionDate'], ecg.TestDemographics['AcquisitionTime']
    except:
        patientid, ecg_date, ecg_time = float('nan'), float('nan'), float('nan')
    try:
        dob = ecg.PatientDemographics['DateofBirth']
        age = (pd.to_datetime(ecg_date)-pd.to_datetime(dob))/pd.Timedelta(1, unit='d')/365.25
    except:
        dob, age = float('nan'), float('nan')
    try:
        gender, race = ecg.PatientDemographics['Gender'], ecg.PatientDemographics['Race']
    except:
        gender, race = float('nan'), float('nan')
    try:
        diagnosis_statement = ecg.Diagnosis['DiagnosisStatement']
    except:
        diagnosis_statement = float('nan')
    try:
        sitename, location = ecg.TestDemographics['SiteName'], ecg.TestDemographics['LocationName']
    except:
        sitename, location = float('nan'), float('nan')
    try:
        hr, pr_interval, qrs_duration, qtc_duration, paxis, qrsaxis, taxis = ecg.RestingECGMeasurements['VentricularRate'], ecg.RestingECGMeasurements['PRInterval'], ecg.RestingECGMeasurements['QRSDuration'], ecg.RestingECGMeasurements['QTCorrected'], ecg.RestingECGMeasurements['PAxis'], ecg.RestingECGMeasurements['RAxis'], ecg.RestingECGMeasurements['TAxis']
    except:
        hr, pr_interval, qrs_duration, qtc_duration, paxis, qrsaxis, taxis = float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    demo = pd.DataFrame((file_name[:-4], patientid, ecg_date, ecg_time, dob, age, gender, race, diagnosis_statement, sitename, location, hr, pr_interval, qrs_duration, qtc_duration, paxis, qrsaxis, taxis)).T
    if file_num == 0:
        demo.to_csv(output_path + "demo.csv", mode='a', header=['filename', 'patientid', 'ecg_date', 'ecg_time', 'dob', 'age', 'gender', 'race', 'diagnosis_statement', 'sitename', 'location', 'hr', 'pr_interval', 'qrs_duration', 'qtc_duration', 'paxis', 'qrsaxis', 'taxis'], index=False)
    else:
        demo.to_csv(output_path + "demo.csv", mode='a', header=False, index=False)       