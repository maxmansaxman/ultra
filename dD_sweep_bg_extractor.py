# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:02:49 2026

@author: Thermo
"""


import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as sm
import shutil

import matplotlib.pyplot as plt

import Ultra_lab_book_parser_CH4 as ups
  
def get_largest_file(directory):
    # Get all entries in the directory that are files
    files = [f for f in os.scandir(directory) if f.is_file()]
    
    if not files:
        return None

    # Use max() with a key to find the entry with the largest st_size
    largest = max(files, key=lambda f: f.stat().st_size)
    
    return largest.path, largest.stat().st_size

sweepbgs = []
normalbgs = []
counter = 0
parentFolder = os.path.abspath('C:/Users/Thermo/Documents/Ultra/Data/CH4')


i = 0
for dirPath, dirNames, fileNames in os.walk(parentFolder):
    if dirPath.endswith('dD_MRAI'):
        if 'sweepScans.xlsx' in fileNames:
            # read sweepbg
            thisSweepPath = os.path.join(dirPath, 'sweepScans.xlsx')
            thisSweep = pd.read_excel(thisSweepPath, index_col=0)
            print('Reading {0} from {1}'.format(counter, dirPath))
            thisSweep['count'] = counter
            thisSweep['is_sample'] = 1
            thisSweep.loc[0:int(len(thisSweep)/2), 'is_sample']=0
            thisSweep['massPosition'] = thisSweep.index
            thisSweep.loc[thisSweep['is_sample']==1, 'massPosition'] -=int(len(thisSweep)/2)
            thisSweep['filePath'] = dirPath
            sweepbgs.append(thisSweep)
            
            # read mainbgs
            if 'backgrounds_mean.xlsx' in fileNames:
                thisNormalBg = pd.read_excel( os.path.join(dirPath, 'backgrounds_mean.xlsx'))
                thisNormalBg['count'] = counter
                thisNormalBg['filePath'] = dirPath
                normalbgs.append(thisNormalBg)
            
            counter +=1
            continue

dsweep = pd.concat(sweepbgs)
dnbg = pd.concat(normalbgs).ffill()
to_mean = ['i16', 'i17']
to_first = ['block', 'cycle_number', 'filePath']
dns = dnbg.groupby(['count','is_sample'])[to_first].first()
dns = dns.merge(dnbg.groupby(['count','is_sample'])[to_mean].mean(), how='left', left_index=True, right_index=True).reset_index()
dsweep = dsweep.merge(dns, how='left', on=['count', 'is_sample'], suffixes=['', '_nbg'])
dsweep.to_excel('dD_sweep_scans_all_9-2026.xlsx')            

        # if newDirName not in dirPath:
        #     destPath = os.path.join(dirPath, newDirName)
        #     os.makedirs(destPath)
        #     # find largest file (assume this is the data file)
        #     largestFile, largestFileSize = get_largest_file(dirPath)
        #     largestFileName = os.path.split(largestFile)[1]
        #     toCopy = [largestFileName, 'logbook.xlsx']
        #     # copy these over
        #     for fileName in toCopy:
        #         try:
        #             shutil.copy2(os.path.join(dirPath, fileName), os.path.join(destPath, fileName))
        #         except(FileNotFoundError):
        #             print('No logbook in {0}'.format(dirPath))
        #             continue
                
        #     # step into new dir
        #     os.chdir(destPath)
        #     # run script with this big file
        #     targetFile = os.path.abspath(largestFileName)
        #     ups.run_main_script(filePath=targetFile)
        #     # step back out
        #     os.chdir('..')
        #     plt.close('all')
            
            
            

