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
fragbgs = []

parentFolder = os.path.abspath('C:/Users/Thermo/Documents/Ultra/Data/CH4/2026')


i = 0
for dirPath, dirNames, fileNames in os.walk(parentFolder):
    if 'dD_MRAI' in dirPath:
        # make new dir
        newDirName = 'dynamic_bg_13C'
        if newDirName not in dirPath:
            destPath = os.path.join(dirPath, newDirName)
            os.makedirs(destPath)
            # find largest file (assume this is the data file)
            largestFile, largestFileSize = get_largest_file(dirPath)
            largestFileName = os.path.split(largestFile)[1]
            toCopy = [largestFileName, 'logbook.xlsx']
            # copy these over
            for fileName in toCopy:
                try:
                    shutil.copy2(os.path.join(dirPath, fileName), os.path.join(destPath, fileName))
                except(FileNotFoundError):
                    print('No logbook in {0}'.format(dirPath))
                    continue
                
            # step into new dir
            os.chdir(destPath)
            # run script with this big file
            targetFile = os.path.abspath(largestFileName)
            ups.run_main_script(filePath=targetFile)
            # step back out
            os.chdir('..')
            plt.close('all')
            
            
            

