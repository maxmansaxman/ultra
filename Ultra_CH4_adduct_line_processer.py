# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:45:55 2025

@author: Thermo
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams.update({'mathtext.default': 'regular'})
import pandas as pd
import os
import re
from datetime import datetime
from scipy.special import erf, erfinv
from scipy.optimize import minimize, least_squares
from scipy import sparse
from scipy.sparse.linalg import spsolve

from lmfit import Model
from scipy.signal import find_peaks

# plt.style.use('ggplot')
plt.close('all')

plt.ion()

homeDir = os.getcwd()




def get_list_of_files_to_import():
    acq_name =  input('Drag all Qtegra files to process ').replace('\\ ', ' ').strip("'").strip('"').strip()
    acq_name=acq_name.strip().strip("'").strip('"')
    acq_name_list = acq_name.split(' ')
    acq_name_list = [l.strip(',').strip("'") for l in acq_name_list]
    return(acq_name_list)

def import_scans_stich_masses(fileName):
    # attempt to treat as a newer style export that is readable with pyarrow
    dn = pd.read_csv(fileName+'.csv', sep=';', engine='pyarrow')
    print('Valid file, now calculating...')
    dn.columns = range(len(dn.columns))
    # split and append blocks, meas, and peak IDs
    dn_extras = dn[2].str.split(':', expand=True)
    dn_extras.columns = ['block', 'meas', 'peak']
    dn = pd.concat([dn, dn_extras], axis=1)
    peakIDs_obs = list(dn['peak'].unique())
    peakIDs_obs.remove(None)
    dc = dn.loc[(dn[3]=='Y [cps]'), [1, 'block', 'meas', 'peak', 4]].copy()
    # make simpler version, merge master and reference blocks
    dfs =  []
    for peak in dc['peak'].unique():
        dfs.append(dc.loc[(dc['peak']==peak), [1, 'block', 'meas', 4]].copy())
        dfs[-1].rename(columns={1:'i', 4: peak}, inplace=True)
        dfs[-1] = dfs[-1].astype({'i':int, 'block':int, 'meas':int, peak:float})
    ds = pd.merge(*dfs, how='outer', on=['i', 'block', 'meas'])
    while True:
        
        if 'CDD' in fileName:
            CO2TemplatePath = homeDir + '/forCO2template/CO2_purity_template_CDD_v2.xlsx'
            scanType='CDD'
            break
        elif 'amp' in fileName:
            CO2TemplatePath = homeDir + '/forCO2template/CO2_purity_template_amp_v2.xlsx'
            scanType='amp'
            break
        else:
            print('Cannot identify file type by name')
            print('Ensure filename contains either "CDD" or "amp"')
            input("Press ENTER to continue... ")
            
        measDate = re.findall('(1?\d-[1-3]?\d-20[2,3]\d)', fileName)[0]
        measDateTime = datetime.strptime(measDate, '%m-%d-%Y')
        if measDateTime < datetime(2025, 3, 13):
            CO2TemplatePath = CO2TemplatePath.replace('v1', 'v2')
        
        
    
    cTemplate = pd.read_excel(CO2TemplatePath)
    # merge everything except the intensities
    colsToAdd = cTemplate.columns[~cTemplate.columns.str.contains('(cps)', regex=False)]
    dsf = ds.merge(cTemplate.loc[:, colsToAdd], how='inner', on=['i', 'block'])
    # save
    dsf.to_excel('{0}_processed_all.xlsx'.format(fileName))
        
        
    # work on purifications and fits
    # group by block, sample, measure line, mass to average mutiple integrations
    dsg = dsf.groupby(['Measure', 'block', 'Measure Line', 'Mass ReferenceCollector (u)'])
    
    toMean = ['MasterCollector', 'ReferenceCollector', 'Time (s)', 'Mass MasterCollector (u)']
    toFirst = ['Variable Volume']
    # average successive integrations
    dsm = pd.merge(dsg[toMean].mean(), dsg[toMean].std(), how='inner', left_index=True, right_index=True, suffixes=['', '_sd'])
    # reset index so that easier to work with
    dsm = dsm.merge(dsg[toFirst].first(), how='left', left_index=True, right_index=True)
    dsm = dsm.reset_index()
    # export scans and return file
    dsm.to_excel('{0}_processed_mean.xlsx'.format(fileName))
    return(dsm, scanType)



def loop_and_get_peak_heights(dsm):
    peaksPath = homeDir + '/forCO2template/CO2_isobars_for_fits.xlsx'
    peaksToFit = pd.read_excel(peaksPath,
                               index_col=0).dropna(subset=['toUse'])
    peaksToFit['Mass_round'] = peaksToFit['Mass'].round(3)
    
    scanPath = homeDir + '/forCO2template/CO2_purity_blocks.xlsx'
    scanIDs = pd.read_excel(scanPath)
    
    doFit = scanIDs.loc[(scanIDs['doFit']==True), :]
    # loop through scans, plot values
    results = []
    # peaks = []
    # peaksMeta = []
    
    # add placeholders for corrected mass, intensity
    toAdd = ['Mass (Da)', 'Intensity (cps)', 'Mass_round']
    for key in toAdd:
        dsm[key] = np.nan
        
    for i, row in doFit.iterrows():
        rowResults = []
        # block, measureLine = scan
        ids = dsm.loc[(dsm['Scan type']==row['Labbook Type']) & 
                            (dsm['block']==row['block']) &
                            (dsm['Measure Line']==row['Measure Line']), :].index
        ids =  dsm.index.isin(ids)

        knownPeaks = peaksToFit.loc[peaksToFit['toUse']==row['Cardinal mass'], :]



        dsm.loc[ids, 'Mass (Da)'] = dsm.loc[ids, 'Mass {0} (u)'.format(row['Cup'])]
        dsm.loc[ids, 'Intensity (cps)'] = dsm.loc[ids, row['Cup']]


        
        # correct mass scale based on reference peak, assuming ref peak is highest in specified gas
        biggestPeak = dsm.loc[ids & (dsm['Variable Volume']=='Variable Volume {0}'.format(
            row['gasForFit'])), :]
        biggestPeakMass = biggestPeak.nlargest(n=5, columns='Intensity (cps)')['Mass (Da)'].mean()
        expectedMass = knownPeaks.loc[knownPeaks['Formula'] == row['referencePeak'], 'Mass'].iloc[0]
        massOffsetInitial = biggestPeakMass - expectedMass
        dsm.loc[ids, 'Mass (Da)'] = dsm.loc[ids, 'Mass (Da)'] - massOffsetInitial

        dsm.loc[ids, 'Mass_round'] = dsm.loc[ids, 'Mass (Da)'].round(3)
        
        gases = dsm.loc[ids, 'Variable Volume'].sort_values().unique()

        fig, ax = plt.subplots(figsize=(8,8), nrows=2, sharex=True)
        for j, gas in enumerate(gases):
            igs = dsm.loc[ids & (dsm['Variable Volume']==gas), :].index
            thisScan = dsm.loc[igs, :]

            xData = thisScan['Mass (Da)'].values
            yData = thisScan['Intensity (cps)'].values
            if row['CupType'] == 'amp':
                peaks, peakDict = find_peaks(yData, prominence=1e4, width=3, rel_height=0.5, distance=8)
            else:
                peaks, peakDict = find_peaks(yData, prominence=4, width=3, rel_height=0.5, distance=8)
            
            # thesePeaks = knownPeaks.copy()
            # thesePeaks['Abundance'] = amplitudes
            # thesePeaks['gas'] = gas
            # thesePeaks['block'] = block
            # thesePeaks['Measure line'] = measureLine
            # thesePeaks['Scan type'] = scanType
            
            # peakFits.append(res)
            thispdf = pd.DataFrame(peakDict)
            thispdf['peakID'] = peaks
            thispdf['peak_intensity'] = yData[peaks]
            thispdf['gas'] = gas
            
            rowResults.append(thispdf)

            ax[0].plot(xData, yData, '-o', alpha=0.3, markersize=1, color='C{0}'.format(j), label=gas)
            ax[1].plot(xData, yData, '-o', alpha=0.3, markersize=1, color='C{0}'.format(j), label=gas)
            if len(peaks)>0:
                ax[0].plot(xData[peaks], yData[peaks], 'x', color='C{0}'.format(j), label='__nolegend__')
                ax[1].plot(xData[peaks], yData[peaks], 'x', color='C{0}'.format(j), label='__nolegedn__')
                
        # compute midpoints of all found peaks
        rowResults = pd.concat(rowResults)
        rowResults['peak_midpoint'] = (rowResults['right_ips'] + rowResults['left_ips'])/2
        rowResults['Mass'] = np.interp(rowResults['peak_midpoint'], range(len(xData)), xData)
        rowResults['Mass_fwhm_left'] = np.interp(rowResults['left_ips'], range(len(xData)), xData)
        rowResults['Mass_fwhm_right'] = np.interp(rowResults['right_ips'], range(len(xData)), xData)
        
        rowResults['baseline'] = rowResults['peak_intensity'] - rowResults['prominences']
        rowResults['peak_hm'] = rowResults['peak_intensity'] - rowResults['prominences']/2



        # sort for merge
        rowResults = rowResults.sort_values(by='Mass')
        # merge in known peaks
        rowResults = pd.merge_asof(rowResults, knownPeaks, on='Mass', direction='nearest', tolerance=0.003)
        # append to results tab for combine
        results.append(rowResults)
        
        # loop through once more and plot peak widths, centers
        for j, gas in enumerate(gases):
            thesePeaks = rowResults.loc[rowResults['gas']==gas, :]
            for k, tp in thesePeaks.iterrows():
                for thisAx in ax:
                    thisAx.plot([tp['Mass_fwhm_left'], tp['Mass_fwhm_right']],
                                np.array([tp['peak_hm'], tp['peak_hm']]), '-',
                                color='C{0}'.format(j), label='__nolegend__')
                    thisAx.plot(tp['Mass'], tp['peak_hm'], '|',
                                color='C{0}'.format(j), label='__nolegend__')

                # if sample gas, label peak
                if gas =='Variable Volume B':
                    ax[1].text(tp['Mass'] + 0.001, tp['peak_intensity'], tp['Formula'], fontsize='x-small')

                    
        
        

        
        yMax = dsm.loc[ids, 'Intensity (cps)'].max()
        yMin = dsm.loc[ids, 'Intensity (cps)'].min()


                
        ax[1].set_xlabel('Mass (u)')
        ax[0].set_ylabel('Intensity (cps)')
        ax[1].set_ylabel('Intensity (cps)')
        # zoom into low mass peaks for second one

        if yMin > -1:
            ax[1].set_ylim(-3, 60)
        else:
            ax[1].set_ylim(yMin*1.1, yMax*0.011)
        ax[0].legend()

        fig.savefig('CO2_purity_{0}_scan_block_{1}_line_{2}_mass_{3}_with_peaks.pdf'.format(
            row['Labbook Type'], row['block'], row['Measure Line'], int(row['Cardinal mass'])))
        # plt.close(fig)
    # concatenate all peaks, save them
    results = pd.concat(results)
    return(results)

def loop_and_plot_peak_scans(dsm, scanType):
    # import list of peaks to look for
    scans = list(dsm.groupby(['block', 'Measure Line']).groups.keys())
    # loop through scans, plot values
    scanPath =homeDir + '/forCO2template/CO2_purity_blocks.xlsx'
    scanIDs = pd.read_excel(scanPath)
    # peaks

    # gasDict = {'Variable Volume A': 'AWG-CO2', 'Variable Volume B': 'PC-2', 'Variable Volume C': 'Argon blank'}
    for scan in scans:
        if scan not in [(1,1), (4,1)]:
            theseData = dsm.loc[(dsm['block']==scan[0]) & (dsm['Measure Line']==scan[1]), :]
            cardinalMass = theseData['Mass ReferenceCollector (u)'].mean().round(0)
            gases = theseData['Variable Volume'].sort_values().unique()
            
            fig, ax = plt.subplots(figsize=(8,8), nrows=2, sharex=True)
            for i, gas in enumerate(gases):
                # peaks, peakDict = 
                ax[0].plot('Mass ReferenceCollector (u)', 'ReferenceCollector', '-o', alpha=0.3, markersize=1,
                        color='C{0}'.format(i), data=theseData.loc[theseData['Variable Volume']==gas, :], label=gas)
                ax[1].plot('Mass ReferenceCollector (u)', 'ReferenceCollector', '-o', alpha=0.3, markersize=1,
                        color='C{0}'.format(i), data=theseData.loc[theseData['Variable Volume']==gas, :], label=gas)
            ax[1].set_xlabel('Mass (u)')
            ax[0].set_ylabel('Intensity (cps)')
            ax[1].set_ylabel('Intensity (cps)')
            # zoom into low mass peaks for second one
            ylims = ax[0].get_ylim()
            yMax = theseData['ReferenceCollector'].max()
            yMin = theseData['ReferenceCollector'].min()
            if yMin > -1:
                ax[1].set_ylim(-3, 60)
            else:
                ax[1].set_ylim(yMin*1.1, yMax*0.011)
            ax[0].legend()
            fig.savefig('CO2_purity_{0}_scan_block_{1}_line_{2}_mass_{3}.pdf'.format(scanType,scan[0], scan[1], int(cardinalMass)))
            plt.close(fig)
    return



def peak_shape_model_vectorized(massArray, massOffset, sigma, cupWidth, amplitudes, cardinalMass):
    ''' model of a narrow peak using the difference between two erfs'''
    intensity = np.asarray(amplitudes)[:, np.newaxis]/2*(erf((massArray + massOffset + cupWidth)/cardinalMass/sigma)
                             - erf((massArray + massOffset - cupWidth)/sigma/cardinalMass))
    return(intensity.sum(axis=0))

def peak_minimizer_vectorized(p, *extraArgs):
    massArray, signal = extraArgs
    model = peak_shape_model_vectorized(massArray, p[0], p[1], p[2], p[3:])
    # model = modelArray.sum(axis=0)
    misfit = np.abs(model-signal)
    return(np.sum(misfit))


def peak_model_least_sq(p, massArray, intensities, cardinalMass):
    
    model = peak_shape_model_vectorized(massArray, p[0], p[1], p[2], p[3:], cardinalMass)

    return(model - intensities)

def peak_model_least_sq_fixed_width_sigma(p, massArray, intensities, cupWidth, sigma, cardinalMass):
    
    model = peak_shape_model_vectorized(massArray, p[0], sigma, cupWidth, p[1:], cardinalMass)

    return(model - intensities)

def peak_minimizer_vectorized(p, *extraArgs):
    massArray, signal = extraArgs
    model = peak_shape_model_vectorized(massArray, p[0], p[1], p[2], p[3:])
    # model = modelArray.sum(axis=0)
    misfit = np.abs(model-signal)
    return(np.sum(misfit))

# def correct_baseline(mass, intensity, peakLocations):


def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
      W = sparse.spdiags(w, 0, L, L)
      Z = W + lam * D.dot(D.transpose())
      z = spsolve(Z, w*y)
      w = p * (y > z) + (1-p) * (y < z)
    return(z)
    
    # return

def integrate_peaks(dsm, scanType, fileName, massCorrection='add', labelPeaks=True):
    peaksPath = homeDir + '/forCO2template/CO2_isobars_for_fits.xlsx'
    peaksToFit = pd.read_excel(peaksPath,
                               index_col=0).dropna(subset=['toUse'])
    peaksToFit['Mass_round'] = peaksToFit['Mass'].round(3)
    scanPath = homeDir + '/forCO2template/CO2_purity_blocks.xlsx'
    scanIDs = pd.read_excel(scanPath)
    scans = list(dsm.groupby(['block', 'Measure Line']).groups.keys())
    doFit = scanIDs.loc[(scanIDs['doFit']==True) & (scanIDs['Type']==scanType), :]
    # loop through scans, plot values
    results = []
     
    for i, row in doFit.iterrows():
        block = row['block']
        measureLine = row['Measure Line']
        # block, measureLine = scan
        theseData = dsm.loc[(dsm['block']==block) & (dsm['Measure Line']==measureLine), :].copy()
        cardinalMass = theseData['Mass ReferenceCollector (u)'].mean().round(0)
        gases = theseData['Variable Volume'].sort_values().unique()
        scanID = row
        knownPeaks = peaksToFit.loc[peaksToFit['toUse']==scanID['Cardinal mass'], :]
        
        # new general columns for intensity, corrected mass scale
        theseData['Intensity (cps)'] = theseData[scanID['Cup']]
        theseData['Intensity_bg_corr'] =  theseData['Intensity (cps)']
        theseData['Model intensity'] = np.nan
        theseData['Residual'] = np.nan


        theseData['Mass (Da)'] = theseData['Mass {0} (u)'.format(scanID['Cup'])]
        # correct mass scale based on reference peak, assuming ref peak is highest in specified gas
        biggestPeak = theseData.loc[theseData['Variable Volume']=='Variable Volume {0}'.format(
            scanID['gasForFit']), :].nlargest(n=5, columns='Intensity (cps)')['Mass (Da)'].mean()
        expectedMass = knownPeaks.loc[knownPeaks['Formula'] == scanID['referencePeak'], 'Mass'].iloc[0]
        if massCorrection =='add':
            massOffsetInitial = biggestPeak - expectedMass
            theseData['Mass (Da)'] -= massOffsetInitial
        else:
            massScaleCompression = expectedMass/biggestPeak
            theseData['Mass (Da)'] *= massScaleCompression
            
        theseData['Mass_round'] = theseData['Mass (Da)'].round(3)
        
        fig, ax = plt.subplots(figsize=(8,8), nrows=2, sharex=True)
        for j, gas in enumerate(gases):
            ids = theseData.loc[theseData['Variable Volume']==gas, :].index
            thisScan = theseData.loc[ids, :]
            massesToFit = knownPeaks['Mass'].values[:, np.newaxis]
            massRange = thisScan['Mass (Da)'].values

    
    return

def fit_peak_heights(dsm, scanType, fileName, massCorrection='add', labelPeaks=True, bounded=False):
    # import list of peaks to look for
    peaksPath = homeDir + '/forCO2template/CO2_isobars_for_fits.xlsx'
    peaksToFit = pd.read_excel(peaksPath,
                               index_col=0).dropna(subset=['toUse'])
    peaksToFit['Mass_round'] = peaksToFit['Mass'].round(3)
    scanPath = homeDir + '/forCO2template/CO2_purity_blocks.xlsx'
    scanIDs = pd.read_excel(scanPath)
    scans = list(dsm.groupby(['block', 'Measure Line']).groups.keys())
    doFit = scanIDs.loc[(scanIDs['doFit']==True) & (scanIDs['Type']==scanType), :]
    # loop through scans, plot values
    results = []
    fits = []
    # gasDict = {'Variable Volume A': 'AWG-CO2', 'Variable Volume B': 'PC-2', 'Variable Volume C': 'Argon blank'}
    
    
    # first, fit a simple one to get cup widths 
    firstFit = doFit.loc[doFit['fitFirst']==True, :].iloc[0]
    intialResults, initialFits = perform_least_sq_fit(dsm, firstFit, peaksToFit)
    sigma = initialFits[0].x[1]
    cupWidth = initialFits[0].x[2]
    # redo fits, but using these widths and cupwidths
    for i, row in doFit.iterrows():
        newResults, newFits = perform_least_sq_fit(dsm, row, peaksToFit,
                                                   sigma=sigma, cupWidth=cupWidth, makePlot=True)
        
        results += newResults
        fits += newFits

    
        
            
    try:    
        fitSummary = pd.concat(results)
        return(fitSummary)
    except(ValueError):
        return([])

        
def perform_least_sq_fit(dsm, row, peaksToFit, cupWidth=None, sigma=None, makePlot=False, bounded=True):
    results = []
    peakFits = []

    block = row['block']
    measureLine = row['Measure Line']
    # block, measureLine = scan
    theseData = dsm.loc[(dsm['block']==block) & (dsm['Measure Line']==measureLine), :].copy()
    cardinalMass = theseData['Mass ReferenceCollector (u)'].mean().round(0)
    gases = theseData['Variable Volume'].sort_values().unique()
    scanID = row
    knownPeaks = peaksToFit.loc[peaksToFit['toUse']==scanID['Cardinal mass'], :]
    
    # new general columns for intensity, corrected mass scale
    theseData['Intensity (cps)'] = theseData[scanID['Cup']]
    theseData['Intensity_bg_corr'] =  theseData['Intensity (cps)']
    theseData['Model intensity'] = np.nan
    theseData['Residual'] = np.nan


    theseData['Mass (Da)'] = theseData['Mass {0} (u)'.format(scanID['Cup'])]
    # correct mass scale based on reference peak, assuming ref peak is highest in specified gas
    biggestPeak = theseData.loc[theseData['Variable Volume']=='Variable Volume {0}'.format(
        scanID['gasForFit']), :].nlargest(n=5, columns='Intensity (cps)')['Mass (Da)'].mean()
    expectedMass = knownPeaks.loc[knownPeaks['Formula'] == scanID['referencePeak'], 'Mass'].iloc[0]

    massOffsetInitial = biggestPeak - expectedMass
    theseData['Mass (Da)'] -= massOffsetInitial

        
    theseData['Mass_round'] = theseData['Mass (Da)'].round(3)
    
    # 
    fig, ax = plt.subplots(figsize=(8,8), nrows=2, sharex=True)
    for j, gas in enumerate(gases):
        ids = theseData.loc[theseData['Variable Volume']==gas, :].index
        thisScan = theseData.loc[ids, :]
        massesToFit = knownPeaks['Mass'].values[:, np.newaxis]
        massRange = thisScan['Mass (Da)'].values
        
        # estimate backgrounds using the highest 5% mass of every scan
        forBg = thisScan['Mass (Da)'].nlargest(int(0.05*len(thisScan))).index
        bgMean = thisScan.loc[forBg, 'Intensity (cps)'].mean()
        # apply bg corr
        theseData.loc[ids, 'Intensity_bg_corr'] -= bgMean
        
        
        # set up mass array for vectorized peak creation
        massArray = massesToFit- massRange
        # set up intensities
        signal = thisScan['Intensity_bg_corr'].values
        # initial guesses
        massOffset = 0
 
        amplitudeGuess = thisScan.loc[thisScan['Mass_round'].isin(peaksToFit['Mass_round']),
                                      :].groupby('Mass_round')['Intensity_bg_corr'].mean().clip(lower=0).values
        
        # catch cases where incorrect number of peaks
        if len(amplitudeGuess) < len(knownPeaks):
            amplitudeGuess = np.ones(len(knownPeaks))*np.mean(amplitudeGuess)
        if sigma is None:
            sigma = 0.0004
    
            if cupWidth is None:    
                if scanID['CupWidth']=='Narrow':
                    cupWidth = 0.0005
                else:
                    cupWidth=0.005
        
            extraArgs = (massArray, signal, cardinalMass)
            paramsGuess = np.asarray([massOffset, sigma, cupWidth] + list(amplitudeGuess))
            if bounded:
                lowerBounds = [-0.001, 1e-7, 1e-7] + list(np.zeros(len(amplitudeGuess)))
                upperBounds = [0.001, 1, 1] + list(np.ones(len(amplitudeGuess))*1e12)
                bounds = (lowerBounds, upperBounds)
                res = least_squares(peak_model_least_sq, paramsGuess, args=extraArgs, bounds=bounds)
            else:
                res = least_squares(peak_model_least_sq, paramsGuess, args=extraArgs)

            massOffset, sigma, cupWidth = res.x[:3]
            amplitudes = res.x[3:]

        
        else:
            extraArgs = (massArray, signal, cupWidth, sigma, cardinalMass)
            paramsGuess = np.asarray([massOffset] + list(amplitudeGuess))
            if bounded:
                lowerBounds = [-0.001] + list(np.zeros(len(amplitudeGuess)))
                upperBounds = [0.001, ] + list(np.ones(len(amplitudeGuess))*1e12)
                bounds = (lowerBounds, upperBounds)
                res = least_squares(peak_model_least_sq_fixed_width_sigma, paramsGuess, args=extraArgs, bounds=bounds)
            else:
                res = least_squares(peak_model_least_sq_fixed_width_sigma, paramsGuess, args=extraArgs)

            # res = least_squares(peak_model_least_sq_fixed_width_sigma, paramsGuess, args=extraArgs)
            massOffset = res.x[:1]
            amplitudes = res.x[1:]

           

        
        modelArray = peak_shape_model_vectorized(massArray, massOffset, sigma, cupWidth, amplitudes, cardinalMass)
        theseData.loc[ids, 'Model intensity'] = modelArray
        
        # apply mass offset to OG data
        theseData.loc[ids, 'Mass (Da)'] -= massOffset
        
        theseData.loc[ids, 'Residual'] = theseData.loc[ids, 'Intensity_bg_corr'] - modelArray


        
        thesePeaks = knownPeaks.copy()
        thesePeaks['Abundance'] = amplitudes
        thesePeaks['gas'] = gas
        thesePeaks['block'] = block
        thesePeaks['Measure line'] = measureLine
        thesePeaks['Scan type'] = scanType
        
        peakFits.append(res)
        results.append(thesePeaks)
        if makePlot:
            
            ax[0].plot('Mass (Da)', 'Intensity_bg_corr', '.',
                       alpha=0.3, color='C{0}'.format(j), data=theseData.loc[ids, :],
                       label=gas)
            ax[0].plot('Mass (Da)', 'Model intensity', '-',
                       alpha=0.3, color='C{0}'.format(j), data=theseData.loc[ids, :],
                       label='__nolegend__')
            
            ax[1].plot('Mass (Da)', 'Residual', '.', alpha=0.3, color='C{0}'.format(j), data=theseData.loc[ids, :])
    
    # if labelPeaks:
    #     for i, thisPeak in knownPeaks.iterrows():
    if makePlot:
        ax[0].set_ylabel('Intensity (cps)')
        ax[1].set_ylabel('Residual (cps)')
        ax[1].set_xlabel('Mass (Da)')
        ax[0].legend(loc='best')
        fig.savefig('CO2_purity_{0}_scan_block_{1}_line_{2}_mass_{3}_with_fit.pdf'.format(
            scanType, block, measureLine, int(cardinalMass)))
    plt.close(fig)
    return(results, peakFits)



    
###########################################################################
#
# Main script
#
###########################################################################

dsms = []
fitSums = []
fileNames = []
acqNameList = get_list_of_files_to_import()
for dDataFile in acqNameList:  
    dDataFile = os.path.abspath(dDataFile)
    if os.path.exists(dDataFile) and dDataFile.endswith('.csv'):
        fileName = os.path.basename(dDataFile).split('.csv')[0]
        os.chdir(os.path.dirname(dDataFile))
    else:
        print('Not a .csv file ')
        raise(TypeError)
    dsm, scanType = import_scans_stich_masses(fileName)
    
    dsm['Scan type'] = scanType
    dsm['Filename'] = fileName


    dsms.append(dsm)


dsm = pd.concat(dsms).reset_index(drop=True)

peaks = loop_and_get_peak_heights(dsm)
# save spreadsheet
peaks.to_excel('peak_height_summary.xlsx')

# dsms.to_excel('CO2_purity_scans_processed.xlsx')

# fitSums = pd.concat(fitSums)

# fitSums.to_excel('CO2_contaminants_summary.xlsx')





