#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:33:26 2025

@author: Max
"""

import numpy as np
import os
import re
import pandas as pd
from scipy.stats import skewnorm, bootstrap
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import zscore, iqr, skewtest, gaussian_kde

plt.close('all')
homeDir = os.getcwd()

def get_folder_to_import():
    folderName =  input('Drag a folder with d_data_all files to process ').replace('\\ ', ' ').strip("'").strip('"').strip()
    folderName=folderName.strip().strip("'").strip('"')
    folderName = os.path.abspath(folderName)
    # acq_name_list = acq_name.split(' ')
    # acq_name_list = [l.strip(',').strip("'") for l in acq_name_list]
    return(folderName)


folderName = get_folder_to_import()
dirList = []
ds = []
for root, dirs, files in os.walk(folderName):
    if 'd_data_all.xlsx' in files:
        dirName = root
        os.chdir(dirName)
        fileName = 'd_data_all.xlsx'
        print('Processing file in {0}'.format(dirName))
        d = pd.read_excel(fileName)
        # result = fit_skew(fileName, num_resamples=300)
        ds.append(d)
        dirList.append(dirName)
        
df = pd.concat(ds)

df['dirList'] = ''
df.loc[df['meas_line']==0, 'dirList'] = dirList
df.loc[df['dirList']=='', 'dirList'] = np.nan
df['dirList'] = df['dirList'].ffill()

df['meas_type'] = df['dirList'].apply(lambda x: os.path.split(x)[1])
df['folder_name'] = df['dirList'].apply(lambda x: os.path.split(os.path.split(x)[0])[1])
df['meas_date'] = df['folder_name'].str.extract('([0-9]{1,2}-[0-9]{1,2})_')
df['sample_ID'] = df['folder_name'].str.extract('[0-9]{1,2}-[0-9]{1,2}_(.+)')
os.chdir(homeDir)
df.to_feather('d_data_all_runs_2025.feather')

        
 







