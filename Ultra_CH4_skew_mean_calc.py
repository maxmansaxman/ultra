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
from scipy.stats import zscore

plt.close('all')
homeDir = os.getcwd()

def get_list_of_files_to_import():
    acq_name =  input('Drag all d_data_all files to process ').replace('\\ ', ' ').strip("'").strip('"').strip()
    acq_name=acq_name.strip().strip("'").strip('"')
    acq_name_list = acq_name.split(' ')
    acq_name_list = [l.strip(',').strip("'") for l in acq_name_list]
    return(acq_name_list)

def statistic(data):
    tFit = skewnorm.fit(data)
    return(tFit[1])

def bootstrap_skew_mean(data, num_resamples=1000):
    """
    Estimates the sampling distribution of the mean using bootstrap resampling.

    Args:
        data (array-like): The original dataset.
        num_resamples (int): The number of bootstrap samples to generate.

    Returns:
        numpy.ndarray: An array of means from the bootstrap samples.
    """
    n = len(data)
    bootstrap_means = []
    for _ in range(num_resamples):
        # Create a bootstrap sample by sampling with replacement
        resample_indices = np.random.choice(n, n, replace=True)
        bootstrap_sample = data[resample_indices]
        bootstrap_means.append(statistic(bootstrap_sample))
    return np.array(bootstrap_means)

def fit_skew(fileName, percentCutoff=60, num_resamples=100, zcut=6):
    
    # read in file
    d = pd.read_excel(fileName, index_col=0)
    # 
    d['is_sample'] = d['is_sample'].astype(bool)
    #assume the last one is the H4 peak
    RtoUse= re.findall('R1[0-9]', d.columns[-2])[0] + '_stable'
    
    dg = d.groupby(['measure_line', 'is_sample'])
    dp = (dg['i17_raw'].count() - dg['is_off_peak'].sum())/dg['i17_raw'].count()
    d = d.merge((dp*100).reset_index(), how='left', on=['measure_line', 'is_sample'])
    d['percent_on_peak'] = d[0]
    d = d.drop(columns = [0])

    # fit sample and std, compute delta
    means = []
    fits = []
    boots = []
    fig, ax = plt.subplots(nrows=2)
    for b in [0, 1]:
        theseData = d.loc[(d['is_sample']==bool(b)) & (d['percent_on_peak']>percentCutoff) & (d['signal_is_stable']), RtoUse].values
        
        zScore = np.abs(zscore(theseData))
        theseData = theseData[zScore<zcut]
        zScore = np.abs(zscore(theseData))
        theseData = theseData[zScore<zcut]
       
        # theseData = d.loc[(d['is_sample']==bool(b)) & (d['percent_on_peak']>percentCutoff) & (d['signal_is_stable']), RtoUse].values
        thisH = ax[b].hist(theseData, bins=50, density=True)
        xlim = ax[b].get_xlim()
        
        xRange = np.linspace(*xlim, num=1000)
        thisFit = skewnorm.fit(theseData)
        fits.append(thisFit)
        means.append(thisFit[1])
        ax[b].plot(xRange, skewnorm.pdf(xRange, *thisFit), '-')
        ax[b].set_xlim(*xlim)
        
        boots.append(bootstrap_skew_mean(theseData, num_resamples=num_resamples))
        
        
    
    booty = np.stack(boots)
    meany = np.asarray(means)
    delta = (meany[1]/meany[0]-1)*1000
    deltaSE = np.sqrt(((booty.std(axis=1)/meany)**2).sum())*(meany[1]/meany[0])*1000
    result = "delta: {0:.3f} ± {1:.3f}‰".format(delta,deltaSE)
    
    ax[b].text(0.05, 0.95, result, transform=ax[b].transAxes, va='top')
    fig.savefig('skewFits.pdf')
    
    print(result)
    
    with open("skewDelta.txt", "w") as file:
        file.write(result)
    return(result)


acq_name_list = get_list_of_files_to_import()

for i in acq_name_list:
    dirName = os.path.dirname(i)
    fileName = 'd_data_all.xlsx'
    joinName = os.path.join(dirName, fileName)
    if os.path.exists(joinName):
        os.chdir(dirName)
        result = fit_skew(fileName, num_resamples=300)

w = input('press Enter to exit... ')








