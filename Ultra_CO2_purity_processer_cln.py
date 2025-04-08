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
from scipy.signal import find_peaks

plt.close('all')

plt.ion()

homeDir = os.getcwd()
dataDir = '/Users/Thermo/Documents/Ultra/Data/CO2/' 


def get_list_of_files_to_import():
    acq_name =  input('Drag all Qtegra files to process, separated by a space ').replace('\\ ', ' ').strip("'").strip('"').strip()
    acq_name=acq_name.strip().strip("'").strip('"')
    acq_name_list = acq_name.split(' ')
    acq_name_list = [l.strip(',').strip("'") for l in acq_name_list]
    return(acq_name_list)

def add_summary_lines(fileName, dsm):
    runLogPath = dataDir + 'Ultra_CO2_run_log.xlsx'
    runLog = pd.read_excel(runLogPath)
    sampleID, measDate = re.findall('_([A-Za-z0-9\\- .]+)_(1?[0-9]-[1-3]?[0-9]-2?0?[2,3][0-9])', fileName)[0]
    try:
        measDateTime = datetime.strptime(measDate, '%m-%d-%Y')
    except(ValueError):
        measDateTime = datetime.strptime(measDate, '%m-%d-%y')
    # runLog['SampleID'] = runLog['Volume B gas'].str.split(' ', expand=True)[0]
    runLog['SampleID_simple'] = runLog['SampleID'].str.lower().str.replace('[-. ]+','', regex=True)
    sampleID_simple = re.sub('[-. ]+','', sampleID.lower())
    runRow = runLog.loc[(runLog['Date']==measDateTime) & (runLog['SampleID_simple']==sampleID_simple)].iloc[0,:]
    toCopy = ['Sample number', 'Date', 'Source pressure', 'SampleID', 'SampleDate',
              'Volume A gas', 'Volume B gas', 'Volume C gas']
    for key in toCopy:
        dsm[key] = runRow[key]
    
    dsm['Volume B gas_full'] = dsm['Volume B gas'].copy()
    dsm['Volume B gas'] = dsm['SampleID']
    return(dsm)

def add_to_summary_sheet(peaks, ask=True):
    if ask:
        addToSheet = input('Add these peaks to summary sheet? (y/n) ').strip().lower()
        if addToSheet =='n':
            return
        else:
            peakLogPath = dataDir + 'Ultra_CO2_peak_heights.xlsx'
            peakLog = pd.read_excel(peakLogPath, index_col=None)
            thisSampleNumber = peaks['Sample number'].unique()[0]
            if thisSampleNumber in peakLog['Sample number'].unique():
                while True:
                    choice = input('Sample already in summary sheet. (o)verwrite or (q)uit... ').strip().lower()
                    if choice =='q':
                        return
                    elif choice == 'o':
                        toDrop = peakLog.loc[peakLog['Sample number']==thisSampleNumber, :].index
                        peakLog = peakLog.drop(index=toDrop)
                        break
                    else:
                        print('Not a valid selection')

            peakLogNew = pd.concat([peakLog, peaks])
            while True:
                try:
                    peakLogNew.to_excel(peakLogPath, index=None)
                    break
                except(PermissionError):
                    print('Unable to save peaks because the following file is open:\n {0}'.format(peakLogPath))
                    input('Close the file and press Enter to try again... ')
    return
                
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
            CO2TemplatePath = homeDir + '/forCO2template/CO2_purity_template_CDD_v3.xlsx'
            scanType='CDD'
            break
        elif 'amp' in fileName:
            CO2TemplatePath = homeDir + '/forCO2template/CO2_purity_template_amp_v3.xlsx'
            scanType='amp'
            break
        else:
            print('Cannot identify file type by name')
            print('Ensure filename contains either "CDD" or "amp"')
            input("Press ENTER to continue... ")
            
    measDate = re.findall('(1?[0-9]-[1-3]?[0-9]-2?0?[2,3][0-9])', fileName)[0]
    #
    try:
        measDateTime = datetime.strptime(measDate, '%m-%d-%Y')
    except(ValueError):
        measDateTime = datetime.strptime(measDate, '%m-%d-%y')
    # measDateTime = datetime.strptime(measDate, '%m-%d-%Y')
    if measDateTime < datetime(2025,3,19):
        CO2TemplatePath = CO2TemplatePath.replace('v3', 'v2')
        if measDateTime < datetime(2025, 3, 13):
            CO2TemplatePath = CO2TemplatePath.replace('v2', 'v1')
        
    
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



def loop_and_get_peak_heights(dsm, closeFig=False):
    peaksPath = homeDir + '/forCO2template/CO2_isobars_for_fits.xlsx'
    peaksToFit = pd.read_excel(peaksPath,
                               index_col=0).dropna(subset=['toUse'])
    peaksToFit['Mass_round'] = peaksToFit['Mass'].round(3)
    
    scanPath = homeDir + '/forCO2template/CO2_purity_blocks.xlsx'
    scanIDs = pd.read_excel(scanPath)
    
    doFit = scanIDs.loc[(scanIDs['doFit']==True) & (scanIDs['Labbook Type'].isin(dsm['Scan type'].unique())), :]
    # loop through scans, plot values
    results = []

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
        biggestPeakMass = biggestPeak.nlargest(n=3, columns='Intensity (cps)')['Mass (Da)'].mean()
        expectedMass = knownPeaks.loc[knownPeaks['Formula'] == row['referencePeak'], 'Mass'].iloc[0]
        massOffsetInitial = biggestPeakMass - expectedMass
        dsm.loc[ids, 'Mass (Da)'] = dsm.loc[ids, 'Mass (Da)'] - massOffsetInitial

        dsm.loc[ids, 'Mass_round'] = dsm.loc[ids, 'Mass (Da)'].round(3)
        
        gases = dsm.loc[ids, 'Variable Volume'].sort_values().unique()

        fig, ax = plt.subplots(figsize=(8,8), nrows=2, sharex=True)
        for j, gas in enumerate(gases):
            igs = dsm.loc[ids & (dsm['Variable Volume']==gas), :].index
            thisScan = dsm.loc[igs, :]
            gasIDKey = 'Volume {0} gas'.format(gas[-1])
            gasID = thisScan[gasIDKey].unique()[0]
            

            xData = thisScan['Mass (Da)'].values
            yData = thisScan['Intensity (cps)'].values
            if row['CupType'] == 'amp':
                if row['CupSize'] == 'Wide':
                    peaks, peakDict = find_peaks(yData, prominence=1e7, width=5, rel_height=0.5, distance=8)
                else:
                    peaks, peakDict = find_peaks(yData, prominence=1e4, width=3, rel_height=0.5, distance=8)
            else:
                peaks, peakDict = find_peaks(yData, prominence=4, width=3, rel_height=0.5, distance=8)

            thispdf = pd.DataFrame(peakDict)
            thispdf['peakID'] = peaks
            thispdf['peak_intensity'] = yData[peaks]
            thispdf['gas'] = gas
            thispdf['gasID'] = gasID

            

            
            toMean = ['Time (s)', 'Measure', 'block', 'Measure Line']
            toUnique = ['Filename', 'Scan type']
            for key in toMean:
                thispdf[key] = thisScan[key].mean()
            for key in toUnique:
                thispdf[key] = thisScan[key].unique()[0]

            
            rowResults.append(thispdf)

            ax[0].plot(xData, yData, '-o', alpha=0.3, markersize=1, color='C{0}'.format(j), label=gasID)
            ax[1].plot(xData, yData, '-o', alpha=0.3, markersize=1, color='C{0}'.format(j), label=gasID)
            # if len(peaks)>0:
            #     ax[0].plot(xData[peaks], yData[peaks], 'x', color='C{0}'.format(j), label='__nolegend__')
            #     ax[1].plot(xData[peaks], yData[peaks], 'x', color='C{0}'.format(j), label='__nolegedn__')
                
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
                # if gas =='Variable Volume B':
                #     ax[1].text(tp['Mass'] + 0.001, tp['peak_intensity'], tp['Formula'], fontsize='x-small')
    
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
        if closeFig:
            plt.close(fig)
        # plt.close(fig)
    # concatenate all peaks, save them
    results = pd.concat(results)
    toLoopAdd = ['Sample number', 'Date', 'Source pressure', 'SampleID', 'SampleDate', 'Volume B gas_full']
    for key in toLoopAdd:
        results[key] = dsm[key].unique()[0]
    results = results.reset_index(drop=True)
    return(results)

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
dsm = add_summary_lines(fileName, dsm)

peaks = loop_and_get_peak_heights(dsm, closeFig=True)
# save spreadsheet
peaks.to_excel('peak_height_summary.xlsx', index=None)
# add to summary sheet

add_to_summary_sheet(peaks, ask=True)


