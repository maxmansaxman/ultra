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
  

h2rdats = []


parentFolder = os.path.abspath('C:/Users/Thermo/Documents/Ultra/Data/CH4/2026')


i = 0
for dirPath, dirNames, fileNames in os.walk(parentFolder):
    for fileName in fileNames:
        if fileName.endswith('_adduct.csv'):
            adductFile = os.path.join(dirPath, fileName)
            h2r = ups.check_D2_resolution(adductFile)
            h2r['fileName'] = fileName
            h2rdats.append(h2r)
            plt.close('all')
        
h2rall = pd.concat(h2rdats).reset_index()
           
h2 = h2rall.groupby('fileName')
hg = h2[['resolution_erf', 'resolution_interp', 'tailing_13CH5']].mean()
hg = hg.merge(h2[['resolution_erf', 'resolution_interp', 'tailing_13CH5']].std(), left_index=True, right_index=True, how='left', suffixes=['', '_std'])
hg.to_excel('dD2_adduct_summary.xlsx')   

