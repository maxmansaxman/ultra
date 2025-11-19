# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:22:49 2025

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

berkPath = 'C:/Users/Max/Dropbox/postdoc_projects/Processed_ultra_data/methane_data/2019'

psuPath = 'C:/Users/Max/Dropbox/PSU/lab/Ultra/Data/CH4'

ds = []
paths = [berkPath, psuPath]

counter=0
for i, uni in enumerate(['Berkeley', 'PSU']):
    folderName = paths[i]   
    for root, dirs, files in os.walk(folderName):
        fileNames = [i for i in files if 'peak_center_data.xlsx' in i]

        # if 'logbook.xlsx' in files:
            # dirName = root
            # os.chdir(dirName)
            # fileNames = [i for i in files if 'peak_center_data.xlsx' in i]
        if len(fileNames) > 0:
            fileName=fileNames[0]
            print('Processing file in {0}'.format(root))
            thisPath = os.path.join(os.path.abspath(root), fileName)
            d = pd.read_excel(thisPath
                              , index_col=0)
            ds.append(d)
            ds[-1]['Ultra'] = uni
            ds[-1]['id'] = counter
            counter+=1
    
df = pd.concat(ds)

df['abs_delta_offset'] = df['delta_offset'].abs()
# lose ones greater than
df.loc[df['abs_delta_offset']>1.3e-4, 'abs_delta_offset'] = np.nan

# fig, ax = plt.subplots(nrows=2, sharex=True)
fig, ax = plt.subplots()

bins =np.linspace(0, 1.3e-4, num=30)
h1 = ax.hist('abs_delta_offset', bins=bins, alpha=0.4,
                data=df.loc[df['Ultra']=='Berkeley', :], color='purple',
                label='UC Berkeley, n={0}'.format(len(df.loc[df['Ultra']=='Berkeley', :])), density=True)
h2 = ax.hist('abs_delta_offset', bins=bins, alpha=0.4,
                data=df.loc[df['Ultra']=='PSU', :], color='C0',
                label='Penn State, n={0}'.format(len(df.loc[df['Ultra']=='PSU', :])), density=True)
# ax[0].legend()
ax.legend()
plt.title('Difference between sucessive peak centers (in Da)')
ax.set_xlabel(r'$\Delta$ peak center offset (Da) ')
ax.set_ylabel('Frequency')
fig.savefig('Peak_center_precision_Berkeley_vs_PSU.png')
