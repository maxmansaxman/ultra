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
# import mass18_D2_peak_processer
from scipy.special import erf, erfinv
from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d, UnivariateSpline, LSQUnivariateSpline

# from scipy import sparse
# from scipy.sparse.linalg import spsolve

# from lmfit import Model
# from scipy.signal import find_peaks

plt.close('all')

plt.ion()

homeDir = os.getcwd()


def peak_shape_model(mass, peak_center, amplitude, sigma, cup_width=0.00048):
    ''' model of a narrow peak suing the difference between two erfs'''
    intensity = amplitude/2*(erf((mass - peak_center + cup_width)/sigma)
                             - erf((mass - peak_center - cup_width)/sigma))
    return(intensity)

def four_peak_model(mass, center_13CD, amplitude_13CD, amplitude_13C_adduct,
                     amplitude_D2, amplitude_D_adduct,
                     sigma, cup_width):
    intensity = peak_shape_model(mass, center_13CD, amplitude_13CD, sigma,
                                 cup_width=cup_width) \
                + peak_shape_model(mass, center_13CD + 0.00155,
                                   amplitude_13C_adduct, sigma, cup_width=cup_width) \
                + peak_shape_model(mass, center_13CD + 0.00292,
                                   amplitude_D2, sigma,
                                   cup_width=cup_width) \
                + peak_shape_model(mass, center_13CD + 0.00447,
                                   amplitude_D_adduct, sigma,
                                   cup_width=cup_width)
    return(intensity)


def one_peak_minimizer(p, *extra_args):
    masses, signal = extra_args
    model = peak_shape_model(masses, p[0], p[1],
                             p[2], p[3])
    misfit = np.abs(model - signal)
    # misfit = (model - signal)**2

    return(np.sum(misfit))


def four_peak_minimizer(p, *extra_args):
    masses, signal, sigma, cup_width = extra_args
    model = four_peak_model(masses, p[0], p[1], p[2], p[3], p[4],
                            sigma, cup_width)
    misfit = np.abs(model - signal)
    return(np.sum(misfit))


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
        
        if 'adduct' in fileName:
            fileDateStr = re.findall('[0-9]{1,2}-[0-9]{1,2}-[0-9]{2}', fileName)[0]
            fileDate = datetime.datetime.strptime(fileDateStr, '%m-%d-%y')
            if fileDate < datetime.datetime(2025, 5, 16):
                CO2TemplatePath = homeDir + '/CH4templates/CH4_sweep_template_adducts_v1.xlsx'
            else:
                CO2TemplatePath = homeDir + '/CH4templates/CH4_sweep_template_adducts_v2.xlsx'
            scanType='CDD'
            break
        else:
            print('Cannot identify file type by name')
            print('Ensure filename contains either "CDD" or "amp"')
            input("Press ENTER to continue... ")
            
        # measDate = re.findall('(1?\d-[1-3]?\d-20[2,3]\d)', fileName)[0]
        # measDateTime = datetime.strptime(measDate, '%m-%d-%Y')
        # if measDateTime < datetime(2025, 3, 13):
        #     CO2TemplatePath = CO2TemplatePath.replace('v1', 'v2')
            
    
    cTemplate = pd.read_excel(CO2TemplatePath)
    # merge everything except the intensities
    colsToAdd = cTemplate.columns[~cTemplate.columns.str.contains('(cps)', regex=False)]
    dsf = ds.merge(cTemplate.loc[:, colsToAdd], how='inner', on=['i', 'block'])
    # save
    dsf.to_excel('{0}_processed_all.xlsx'.format(fileName))
          
    # # work on purifications and fits
    # # group by block, sample, measure line, mass to average mutiple integrations
    # dsg = dsf.groupby(['Measure', 'block', 'Measure Line', 'Mass ReferenceCollector (u)'])
    
    # toMean = ['MasterCollector', 'ReferenceCollector', 'Time (s)', 'Mass MasterCollector (u)']
    # toFirst = ['Variable Volume']
    # # average successive integrations
    # dsm = pd.merge(dsg[toMean].mean(), dsg[toMean].std(), how='inner', left_index=True, right_index=True, suffixes=['', '_sd'])
    # # reset index so that easier to work with
    # dsm = dsm.merge(dsg[toFirst].first(), how='left', left_index=True, right_index=True)
    # dsm = dsm.reset_index()
    # # export scans and return file
    # dsm.to_excel('{0}_processed_mean.xlsx'.format(fileName))
    return(dsf, scanType)



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

            thispdf = pd.DataFrame(peakDict)
            thispdf['peakID'] = peaks
            thispdf['peak_intensity'] = yData[peaks]
            thispdf['gas'] = gas
            
            toMean = ['Time (s)', 'Measure', 'block', 'Measure Line']
            toUnique = ['Filename', 'Scan type']
            for key in toMean:
                thispdf[key] = thisScan[key].mean()
            for key in toUnique:
                thispdf[key] = thisScan[key].unique()[0]

            
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

def compute_resolution(hrdf):
    peakMax = hrdf['H4_model'].max()
    resEval = {}
    resEval['pts'] = np.array([0.05, 0.95])
    resEval['signal'] = resEval['pts']*peakMax
    # resEval['hi_mass_model'] = []
    # resEval['lo_mass_model'] = []
    # resEval['hi_mass_interp'] = []
    # resEval['lo_mass_interp'] = []
    for i, pt in enumerate(resEval['pts']):
        signal = pt*peakMax
        for k in ['model', 'interp']:
            theseMasses = hrdf.loc[hrdf['H4_' + k] > signal, 'Mass'].values[[0,-1]]
            key = '{0}_{1}'.format(k, pt)
            resEval[key] = theseMasses
    redf = pd.DataFrame(data=resEval)
    modelRes = (18.01/(redf['model_0.05'] - redf['model_0.95'])).abs().mean()
    interpRes = (18.01/(redf['interp_0.05'] - redf['interp_0.95'])).abs().mean()
    print('ERF model resolution: {0:.0f}'.format(modelRes))
    print('Interpolated resolution: {0:.0f}'.format(interpRes))
    return(modelRes, interpRes)



def fit_H2O_peaks(dh):
    fig, ax = plt.subplots()
    H2OResults = {}
    toAdd = ['integration', 'center', 'height', 'sigma', 'cup_width',
             'tailing_13CH5', 'tailing_12CH4D', 'resolution_erf', 'resolution_interp']
    for key in toAdd:
        H2OResults[key] = []
    cupWidth = 0.000455
    sigmaGuess = 0.0004
    evalPoints = np.array([0.00137, -0.00155])
    cutoff = 0.0001
    massScaleConversion = 18.0106/18.0439
    dh['Mass'] = dh['Mass ReferenceCollector (u)']
    # dh['Mass_H4'] = dh['Mass_center']/massScaleConversion
    dh['Signal'] = dh['ReferenceCollector']
    dh['Model'] = np.nan
    dh['Signal_normed'] = np.nan
    dh['Model_normed'] = np.nan
    dh['Mass_precise'] = np.nan

    for thisInt in dh['Integration'].unique():
        
        massRange = dh.loc[dh['Integration']==thisInt, 'Mass'].values
        dh.loc[dh['Integration']==thisInt, 'Mass_precise'] = np.linspace(massRange[0], massRange[-1],
                                                                                num=len(massRange))
        thisDh = dh.loc[dh['Integration']==thisInt]
        # deal with imprecise mass window
        
        # now, fit and plot the OH peak
        extraArgs = (thisDh['Mass_precise'].values,
                     thisDh['Signal'].values)
        paramsGuess = np.array([thisDh.loc[thisDh['Signal'].idxmax(), 'Mass_precise'],
                                 thisDh['Signal'].max(), sigmaGuess,
                                 cupWidth])
        res = minimize(one_peak_minimizer, paramsGuess, args=extraArgs)
        center, height, sigma, cupWidth = res.x
        dh.loc[dh['Integration']==thisInt, 'Model'] = peak_shape_model(
                dh.loc[dh['Integration']==thisInt, 'Mass_precise'], center, height,
                sigma, cupWidth)
        
        dh.loc[dh['Integration']==thisInt, 'Signal_normed'] = dh.loc[dh['Integration']==thisInt, 'Signal']/height
        dh.loc[dh['Integration']==thisInt, 'Model_normed'] = dh.loc[dh['Integration']==thisInt, 'Model']/height
        
        H2OResults['integration'].append(thisInt)
        H2OResults['center'].append(center)
        H2OResults['height'].append(height)
        H2OResults['sigma'].append(sigma)
        H2OResults['cup_width'].append(cupWidth)
        
        thisInterp = UnivariateSpline(extraArgs[0], extraArgs[1], s=300)
        theseEvalPoints = center + evalPoints*massScaleConversion
        cpsObs =np.array([thisDh.loc[(thisDh['Mass_precise'] - theseEvalPoints[0]).abs() < cutoff, 'Signal'].mean(),
                          thisDh.loc[(thisDh['Mass_precise'] - theseEvalPoints[1]).abs() < cutoff, 'Signal'].mean()])
        tailingFactors = cpsObs/height
        
        H2OResults['tailing_13CH5'].append(tailingFactors[0])
        H2OResults['tailing_12CH4D'].append(tailingFactors[1])
        
        hrMasses = np.linspace(massRange[0], massRange[-1], num=1000)
        hrH2O =  peak_shape_model(hrMasses, res.x[0], res.x[1], res.x[2], res.x[3])
        hrdf = pd.DataFrame(data={'Mass_center': hrMasses,
                                          'H4_model': hrH2O})
        hrdf['Mass'] = hrdf['Mass_center']
        hrdf['H4_interp'] = thisInterp(hrdf['Mass_center'])

        modelRes, interpRes = compute_resolution(hrdf)
        H2OResults['resolution_erf'].append(modelRes)
        H2OResults['resolution_interp'].append(interpRes)

    for thisInt in dh['Integration'].unique():
        thisDh = dh.loc[dh['Integration']==thisInt]
        
        ax.plot('Mass_precise', 'Signal', '.', color='C{0}'.format(thisInt), data=thisDh)
        ax.plot('Mass_precise', 'Model', '-', color='C{0}'.format(thisInt), data=thisDh)
    
    h2r = pd.DataFrame(data=H2OResults)
    ax.text(0,0.9,
            'M/∆M erf: {0:.0f}\n interp: {1:.0f}'.format(h2r['resolution_erf'].mean(), h2r['resolution_interp'].mean()),
            transform=ax.transAxes)
    fig.savefig('H2O_scans.pdf')

    return(h2r)

def loop_and_make_adduct_line(da, h2r):
    
    cupWidth = h2r['cup_width'].mean()
    sigma = h2r['sigma'].mean()
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,7))
    ax = ax.ravel()
    fits = np.zeros((2,2,2))
    adf = {}
    toAdd = ['measure', 'center', 'i_16', 'i_13CH3D', 'i_13CH5', 'i_12CH4D','is_std']
    for key in toAdd:
        adf[key] = []
    for meas in da['Measure'].unique():
        thisDa = da.loc[da['Measure']==meas, :]
        adf['i_16'].append(thisDa['MasterCollector'].median())
        massRange = thisDa['Mass ReferenceCollector (u)'].values
        massPrecise = np.linspace(massRange[0], massRange[-1], num=len(massRange))
        signal = thisDa['ReferenceCollector'].values
        signalMax = np.max(signal)
        # now, fit and plot the OH peak
        extraArgs = (massPrecise,
                     signal, h2r['sigma'].mean(), h2r['cup_width'].mean())
        paramsGuess = np.array([massPrecise[np.argmax(signal)],
                                signalMax, signalMax/2, 100, signalMax/30])
        res = minimize(four_peak_minimizer, paramsGuess, args=extraArgs)
        adf['center'].append(res.x[0])       
        adf['i_13CH3D'].append(res.x[1])       
        adf['i_13CH5'].append(res.x[2])
        adf['i_12CH4D'].append(res.x[4])   
        adf['is_std'].append(bool(meas%2))
        adf['measure'].append(meas)

        ax[0].plot(massPrecise, signal, '.', alpha=0.3, color='C{0}'.format(meas%2))
        modelPred = four_peak_model(massPrecise, res.x[0], res.x[1],
                                    res.x[2], res.x[3], res.x[4], sigma, cupWidth)
        ax[0].plot(massPrecise, modelPred, '-', alpha=0.3,color='C{0}'.format(meas%2))
    
    ax[0].set_xlabel('H4 mass (Da)')
    ax[0].set_ylabel('intensity (cps)')
    adf = pd.DataFrame(data=adf)
    forRatio  = adf.columns[adf.columns.str.contains('i_1[2,3]')]
    for i, thisCol in enumerate(forRatio):
        species = thisCol.split('_')[-1]
        ratio = 'R_{0}'.format(species)
        adf[ratio] = adf[thisCol]/adf['i_16']
    
        for cond in [0, 1]:
            tadf = adf.loc[adf['is_std']==cond, :]

            
            ax[i+1].plot('i_16', ratio, 'o', color='C{0}'.format(cond), data=tadf)
        
            if species in ['13CH5', '12CH4D']:
                thisFit = np.polyfit(tadf['i_16'], tadf[ratio], 1)
                fits[:,i-1, cond] = thisFit
                i16range = np.linspace(adf['i_16'].min(), adf['i_16'].max())
                ax[i+1].plot(i16range, thisFit[0]*i16range + thisFit[1], '-', color='C{0}'.format(cond))
        ax[i+1].set_xlabel('i16 (cps)')
        ax[i+1].set_ylabel(ratio)
    
    fig.savefig('adduct_line.pdf')
    tails = h2r[['tailing_13CH5', 'tailing_12CH4D']].mean()
    return(fits, tails)

  
        
        

        
#         # ax.plot('Mass_center', 'Signal', '.', data=thisDh)
#         # ax.plot('Mass ReferenceCollector (u)', 'H4_model', '-', data=thisDh)

        
# # cps_interp_OH = OH_interp(res.x[0] + mass_scale_points*17.09/18.041)





        
        
    
    

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



