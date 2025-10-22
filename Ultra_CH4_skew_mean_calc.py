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
from scipy.stats import zscore, iqr, skewtest

plt.close('all')
homeDir = os.getcwd()

def get_folder_to_import():
    folderName =  input('Drag a folder with d_data_all files to process ').replace('\\ ', ' ').strip("'").strip('"').strip()
    folderName=folderName.strip().strip("'").strip('"')
    folderName = os.path.abspath(folderName)
    # acq_name_list = acq_name.split(' ')
    # acq_name_list = [l.strip(',').strip("'") for l in acq_name_list]
    return(folderName)

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

def fit_skew(fileName, percentCutoff=60, num_resamples=100, iqm=3, medianCut=0.3):
    
    # read in file
    d = pd.read_excel(fileName, index_col=0)
    # 
    d['is_sample'] = d['is_sample'].astype(bool)
    #assume the last one is the H4 peak
    RtoUse= re.findall('R1[0-9]', d.columns[-2])[0] + '_stable'
    massToUse = RtoUse.split('_')[0][1:]
    
    dg = d.groupby(['measure_line', 'is_sample'])
    dp = (dg['i{0}_raw'.format(massToUse)].count() - dg['is_off_peak'].sum())/dg['i{0}_raw'.format(massToUse)].count()
    d = d.merge((dp*100).reset_index(), how='left', on=['measure_line', 'is_sample'])
    d['percent_on_peak'] = d[0]
    d = d.drop(columns = [0])

    # fit sample and std, compute delta
    locs = []
    fits = []
    medsReal = []
    medsFit = []
    clr=0
    fig, ax = plt.subplots(nrows=2)
    for acq in d['acq_number'].unique():
        locs.append([])
        fits.append([])
        medsReal.append([])
        medsFit.append([])        
        for b in [0, 1]:
            theseData = d.loc[(d['is_sample']==bool(b)) & (
                d['percent_on_peak']>percentCutoff) & (
                    d['signal_is_stable']) & (
                        d['acq_number']==acq), RtoUse].values
            # filter for outliers based on median
            dMed = np.median(theseData)
            medsReal[-1].append(dMed)
            theseData = theseData[(theseData > dMed*medianCut) & (theseData < dMed*(1+medianCut))]
            # filter for outliers beyond IQR (conservative estimate)
            Q1 = np.percentile(theseData, 25)
            Q3 = np.percentile(theseData, 75)
            IQR = iqr(theseData)
            lowBound = Q1 - iqm*IQR
            highBound = Q3 + iqm*IQR
            dataCln = theseData[(theseData >= lowBound) & (theseData <= highBound)]
            
        
        # zScore = np.abs(zscore(theseData))
        # theseData = theseData[zScore<zcut]
        # zScore = np.abs(zscore(theseData))
        # theseData = theseData[zScore<zcut]
       
        # theseData = d.loc[(d['is_sample']==bool(b)) & (d['percent_on_peak']>percentCutoff) & (d['signal_is_stable']), RtoUse].values
            thisH = ax[b].hist(dataCln, bins=50, density=True, alpha=0.3, color='C{0}'.format(clr))
            xlim = ax[b].get_xlim()
        
            xRange = np.linspace(*xlim, num=1000)
            thisFit = skewnorm.fit(dataCln, loc=np.median(dataCln), method='MLE')
            fits[-1].append(thisFit)
            locs[-1].append(thisFit[1])
            medsFit[-1].append(skewnorm.median(*thisFit))
            ax[b].plot(xRange, skewnorm.pdf(xRange, *thisFit), '-', color='C{0}'.format(clr))
            ax[b].set_xlim(*xlim)
            ylim = ax[b].get_ylim()
            ax[b].plot([medsReal[-1][-1], medsReal[-1][-1]], ylim, '--', color='C{0}'.format(clr))
            ax[b].set_ylim(*ylim)
            
            clr +=1
           
        
        # boots.append(bootstrap_skew_mean(theseData, num_resamples=num_resamples))
        
        
    
    # booty = np.stack(boots)
    mf = np.asarray(medsFit).T
    deltas = (mf[1]/mf[0]-1)*1000
    delta = np.mean(deltas)
    deltaSD = np.std(deltas)
    deltaSE = np.std(deltas)/np.sqrt(len(deltas))
    
    mr = np.asarray(medsReal).T
    deltasM = (mr[1]/mr[0]-1)*1000
    deltaM = np.mean(deltasM)
    deltaMSD =  np.std(deltasM)
    deltaMSE = np.std(deltasM)/np.sqrt(len(deltasM))
    result = ("data median delta: {0:.3f} ± {1:.3f}‰ \n".format(deltaM,deltaMSE),
              "fit median delta: {0:.3f} ± {1:.3f}‰".format(delta,deltaSE))
    
    rs = ''.join(result)
    ax[b].text(0.05, 0.95, rs, transform=ax[b].transAxes, va='top')
    fig.savefig('skewFits.pdf')
    plt.close('all')
    print(rs)
    resDict = {'d_data': deltaM,
               'd_data_SD': deltaMSD,
               'd_data_SE': deltaMSE,
               'd_fit': delta,
               'd_fit_SD': deltaSD,
               'd_fit_SE': deltaSE}
    
    with open("skewDelta.txt", "w") as file:
        file.write(rs)
    return(resDict)


folderName = get_folder_to_import()
dirList = []
resList = []
for root, dirs, files in os.walk(folderName):
    if 'd_data_all.xlsx' in files:
        dirName = root
        os.chdir(dirName)
        fileName = 'd_data_all.xlsx'
        print('Processing file in {0}'.format(dirName))
        result = fit_skew(fileName, num_resamples=300)
        resList.append(result)
        dirList.append(dirName)
        


        
        
# for i in acq_name_list:
#     dirName = os.path.dirname(i)
#     fileName = 'd_data_all.xlsx'
#     joinName = os.path.join(dirName, fileName)
#     if os.path.exists(joinName):
#         os.chdir(dirName)
#         result = fit_skew(fileName, num_resamples=300)

w = input('press Enter to exit... ')








