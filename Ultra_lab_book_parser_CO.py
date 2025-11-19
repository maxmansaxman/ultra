
"""
Processing script Qtegra export files of methyl group measurements

Written by Max Lloyd in 2018-2022
max.k.lloyd@gmail.com

"""




import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams.update({'mathtext.default': 'regular'})
mpl.rcParams.update({'lines.markeredgecolor': 'black'})
mpl.rcParams.update({'scatter.edgecolors': 'black'})
mpl.rcParams.update({'lines.markersize': 10})
import pandas as pd
import os
from scipy.special import erf, erfinv
import statsmodels.api as sm
from scipy.interpolate import interp1d, UnivariateSpline, LSQUnivariateSpline
from scipy.optimize import minimize, least_squares
from scipy.stats import zscore, skewnorm
import re
import datetime
plt.close('all')

homeDir = os.getcwd()
idx = pd.IndexSlice


def get_measurement_params(auto_detect=False, file_name=''):
    ''' Retreives three key measurement parameters (number of cylces,
    sub-integration time, number of sub-integrations) by either prompting for 
    input or assuming based on the filename.
    '''
    if auto_detect and len(file_name) > 0:
        # assumes parameters based on file name and standard configuration
        repeatBlocks=False
        if '_dD_' in file_name:
            (cycle_num, integration_time, integration_num) = (10, 0.524, 180)
            peakIDs = ['i16', 'i17']
            blockIDs = ['sweep', 'meas', 'frag', 'bg']

            return(cycle_num, integration_time, integration_num, peakIDs, blockIDs, repeatBlocks)
        elif '_d13CD_' in file_name:
            if '_NH3_' in file_name:
                (cycle_num, integration_time, integration_num) = (6, 1.048, 90)
                peakIDs = ['i16', 'i17', 'i18']
                blockIDs = ['sweep', 'meas','bg_NH3','bg']
                repeatBlocks=True
            else:               
                (cycle_num, integration_time, integration_num) = (10, 1.048, 90)
                peakIDs = ['i16', 'i17', 'i18']
                blockIDs = ['sweep', 'bg_NH3', 'meas','bg_NH3','bg']
            return(cycle_num, integration_time, integration_num, peakIDs, blockIDs, repeatBlocks)
        elif '_dD2_' in file_name:
            (cycle_num, integration_time, integration_num) = (10, 1.048, 90)
            peakIDs = ['i16', 'i17', 'i18']
            blockIDs = ['meas', 'bg']

            return(cycle_num, integration_time, integration_num, peakIDs, blockIDs, repeatBlocks)
        elif 'D17O' in file_name: 
            (cycle_num, integration_time, integration_num) = (10, 1.048, 60)
            peakIDs = ['i_16', 'i_17', 'i_18']
            blockIDs = ['bg', 'meas', 'bg']

        elif 'CO_clump' in file_name: 
            (cycle_num, integration_time, integration_num) = (10, 1.048, 90)
            peakIDs = ['i28', 'i29', 'i30', 'i31']
            blockIDs = ['bg', 'meas', 'bg']
            return(cycle_num, integration_time, integration_num, peakIDs, blockIDs, repeatBlocks)


        else:
            print('Auto detection of measurement params failed... ')      
    while True:
        try:
            integration_num = int(input('number of integrations in each measurement? '))
            break
        except ValueError:
            print('Not a valid number, try again...')
    while True:
        try:
            cycle_num = int(input('number of cycles in each acquire? '))
            break
        except ValueError:
            print('Not a valid number, try again...')
    while True:
        try:
            integration_time = float(input('Individual integration time? '))
            break
        except ValueError:
            print('Not a valid float, try again...')
    while True:
        try:
            basePeak = input('Name of base peak? ')
            peakIDs = [basePeak, 'others']
            blockIDs = ['meas', 'bg']
            break
        except ValueError:
            print('Not a valid peak ID')
    return(cycle_num, integration_time, integration_num, peakIDs, blockIDs)


def compute_N2_contents(fileName):
    dn = pd.read_csv(fileName, sep=';', engine='pyarrow')
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
        dfs[-1] = dfs[-1].astype({'i':int, 'block':int, 'meas':int, peak:float}).set_index(
            ['i', 'block', 'meas'])
    ds = pd.concat(dfs, axis=1, join='outer')
    peaks = ds.columns.values
    dn = ds.loc[idx[:,:,2], :].copy()
    ds.loc[idx[:,:,2], :] =  np.nan
    dsn = ds.merge(dn, how='outer', left_index=True, right_index=True, suffixes=['', '_N2'])
    dsn['cycle_number'] = np.floor((dsn.index.get_level_values(0)/240).values)
    dsn['is_sample']= dsn['cycle_number']%2
    dsn = dsn.reset_index()
    cycles = dsn['cycle_number'].unique()
    # fig, ax = plt.subplots()
    hasN2 = peaks[:2]
    # fig, ax . plt.subplots()
    for peak in hasN2:
        # hasN2.append(peak)
        dsn[peak+'_pred'] = np.nan
        for cycle in cycles:
            thisDr = dsn.loc[(dsn['cycle_number']==cycle), peak]
            X = thisDr.index.values
            X = sm.add_constant(X)
            XtoFill = X[thisDr.isna()]
            # get Y
            yTrue = thisDr[thisDr.notnull()].values

            resRLM = sm.RLM(yTrue, X[thisDr.notnull()],  M=sm.robust.norms.TrimmedMean()).fit()
            dsn.loc[dsn['cycle_number']==cycle, peak+'_pred'] = resRLM.predict(X)
    
    # loop again and plot
    fig, ax = plt.subplots(nrows=len(hasN2))
    for i, peak in enumerate(hasN2):
        dsn[peak + '_justN2'] = dsn[peak + '_N2'] - dsn[peak + '_pred']
        dsn['RN2_'+peak] = dsn[peak + '_justN2']/dsn[peak + '_pred']
    dsn['R15'] = dsn['i29_justN2']/dsn['i28_justN2']/2
    dsn['i30_15N2_pred'] = dsn['i28_justN2']*(dsn['R15']**2)
    dsn['percent_N2'] = dsn['i28_justN2']/dsn['i28_pred']*100
    
    
    dgn = dsn.groupby(['is_sample', 'cycle_number'])
    dgm = dgn.mean()
    
    dgf = dsn.groupby(['is_sample'])
    dgfm = dgf.mean()
    
    
    fig, ax = plt.subplots()
    N2_fits = []
    d15N = (dgfm['R15'][1]/dgfm['R15'][0]-1)*1000
    dsn['i30_N2_corr'] = np.nan
    
    for i, gas in enumerate(['Std','Sample']):
        thisDf = dgm.loc[idx[i], :]
        ax.plot('i28_pred', 'i30_15N2_pred', 'o', color='C{0}'.format(i),
                data=thisDf,label='{0}: {1:.2f}% N2'.format(gas, thisDf['percent_N2'].mean()) )
        thisFit = np.polyfit(thisDf['i28_pred'], thisDf['i30_15N2_pred'], 1)
        xrange = np.linspace(thisDf['i28_pred'].min(), thisDf['i28_pred'].max())
        ax.plot(xrange, xrange*thisFit[0] + thisFit[1], '-',
                zorder=0, color='C{0}'.format(i),
                label='__nolegend__')
        N2_fits.append(thisFit)
        
        dsn.loc[dsn['is_sample']==i, 'i30_N2_corr'] = dsn.loc[dsn['is_sample']==i,'i30'] - dsn.loc[
            dsn['is_sample']==i, 'i28']*N2_fits[i][0] + N2_fits[i][1]
    ax.text(0.99, 0.01,'Sample d15N = {0:.1f}‰'.format(d15N),
            transform=ax.transAxes, va='bottom', ha='right')
    ax.set_xlabel('i28 (cps)')
    ax.set_ylabel(r'Pred. $^{15}N_2$ on i30 (cps)')
    ax.legend()
    fig.savefig('N2_calibration.pdf')
    
    # make summary df
    dsn['cycle_matched'] = (dsn['cycle_number']/2).astype(int)
    dsn['R29'] = dsn['i29']/dsn['i28']
    dsn['R30'] = dsn['i30_N2_corr']/dsn['i28']
    dsn['R31'] = dsn['i31']/dsn['i28']
    
    dgs = dsn.groupby(['is_sample', 'cycle_matched'])
    
    toUse = ['i28', 'percent_N2', 'R29', 'R30', 'R31', 'R15']
    
    dsum = pd.merge(dgs[toUse].mean(), dgs[toUse].std(),
                    left_index=True, right_index=True, suffixes=['', '_sd'])
    dsum = dsum.merge(dgs[toUse].std()/np.sqrt(dgs[toUse].count()),
                      left_index=True, right_index=True, suffixes=['', '_se'])
    
    # pivot
    dpiv = pd.pivot_table(dsum.reset_index(), index='cycle_matched', columns='is_sample')
    dpiv['P_imbalance'] = (dpiv['i28'][1]/dpiv['i28'][0]-1)*100
    deltas = []
    for i in ['29', '30', '31']:
        R = 'R'+i
        delta = 'd'+i
        dpiv[delta] = (dpiv[R][1]/dpiv[R][0]-1)*1000
        dpiv[delta+'_se'] = np.sqrt(((dpiv[R+'_se']/dpiv[R])**2).sum(axis=1))*1000
        deltas.append(delta)
    
    mv = {'mean':[], 'se':[]}
    fig, ax = plt.subplots(nrows=len(deltas), sharex=True)
    for i, delta in enumerate(deltas):
        thisMean = dpiv[delta].mean()
        thisSE = dpiv[delta].std()/np.sqrt(len(dpiv[delta]))
        mv['mean'].append(thisMean)
        mv['se'].append(thisSE)
        ax[i].errorbar(dpiv['i28'].mean(axis=1), dpiv[delta],
                       yerr=dpiv[delta+'_se'].values, fmt='o', color='C{0}'.format(i),
                       ecolor='k')
        ax[i].text( 0.99, 0.01, '{0:.2f}±{1:.2f}‰'.format(thisMean, thisSE),
                   transform=ax[i].transAxes, ha='right', va='bottom')
        ax[i].set_ylabel(r'$\delta${0} (‰ vs. wg)'.format(delta[1:]))
    ax[i].set_xlabel('i28 (cps)')
    fig.savefig('N2_deltas.pdf')
    mv['key'] = deltas
    mv = pd.DataFrame(mv).set_index('key')
    mv.to_excel('N2_deltas_summary.xlsx')
    dpiv.to_excel('N2_deltas_all.xlsx')
    return(N2_fits, dpiv)

def process_Qtegra_csv_file(d_data_file, peakIDs, blockIDs, prompt_for_params=False,
                            integration_time=16.7, integration_num=10,
                            cycle_num=10, sigma_filter=2,
                            prompt_for_backgrounds=True,
                            input_tail_D2_background=False, repeatBlocks=False, zscoreCutoff=6):
    """
    Main function for importing, cleaning and processing a single .csv file
    exported by Qtegra.

    Parameters
    ----------
    d_data_file : string
        path containing absoulte file location and name.
    prompt_for_params : bool, optional
        Boolean for whether to prompt user for measurement params.
        The default is False.
    integration_time : float, optional
        sub-integration time in seconds. The default is 16.7.
    integration_num : int, optional
        number of consecutive sub-integrations per integration.
        The default is 10.
    cycle_num : int, optional
        number of sample-standard cycles in a single acquisition
        (before a new pressure balance occurs).
        The default is 10.
    sigma_filter : float, optional
        z-score used to decide sensitivity of off-peak outlier test.
        The default is 2.
    prompt_for_backgrounds : bool, optional
        Whether or not to prompt user to input backgrounds manually.
        The default is True.
    input_tail_D2_background : bool, optional
        Whether or not to prompt user to input a total background on 12CHD2
        due to peak tailing. The default is False.

    Returns
    -------
    dr : pd.DataFrame
        DataFrame containing all background-corrected intensity and ratio data.
        Each row is its own sub-integration.
    drm : pd.DataFrame
        DataFrame containing average intensity, ratio, and delta values for
        each sample-standard cycle. Each row is its own integration.
    file_name : string
        Name of the file processed, not including the file path.

    """
    # 1. Import data and convert to a dataframe with one sub-int per row
    d_data_file = os.path.abspath(d_data_file)
    if os.path.exists(d_data_file) and d_data_file.endswith('.csv'):
        file_name = os.path.basename(d_data_file).split('.csv')[0]
    else:
        print('Not a .csv file ')
        raise(TypeError)
    # attempt to treat as a newer style export that is readable with pyarrow
    dn = pd.read_csv(d_data_file, sep=';', engine='pyarrow')
    print('Valid file, now calculating...')
    dn.columns = range(len(dn.columns))
    # split and append blocks, meas, and peak IDs
    dn_extras = dn[2].str.split(':', expand=True)
    dn_extras.columns = ['block', 'meas', 'peak']
    dn = pd.concat([dn, dn_extras], axis=1)
    peakIDs_obs = list(dn['peak'].unique())
    peakIDs_obs.remove(None)
    
    
    dc = dn.loc[(dn[3]=='Y [cps]'), [1, 'block', 'meas', 'peak', 4]].copy()
    #rename columns to match legacy processer
    d = dc.rename(columns={1: 'measure_line', 4:'f'})
    d['measure_line'] = d['measure_line'].astype(int)
    d = d.sort_values(by=['measure_line', 'peak'])
    d['block'] = d['block'].astype(int)
    # rename bg blocks to be same as base peaks
    d['peak'] = d['peak'].str.replace('_bg', '')
    # remove these from peak list
    peakIDs_cln = [i for i in peakIDs_obs if '_bg' not in i]
    # commands that could make processing simpler later
    dp = d.pivot(columns='peak', values='f', index='measure_line')
    dp = dp.reset_index()
    dp = dp.merge(d.groupby('measure_line')[['block', 'meas']].first().reset_index(),
                  how='left', on='measure_line')
    dp[peakIDs_cln] = dp[peakIDs_cln].astype(float)
    # apply block IDs
    if repeatBlocks:
        # if repeatBlocks, assume it's the middle ones
        blockIDs_repeats = [blockIDs[0]]
        toRepeat = blockIDs[1:-1]
        for i in range(0, len(dp['block'].unique())-2, len(toRepeat)):
            blockIDs_repeats += toRepeat
        blockIDs_repeats += [blockIDs[-1]]
        # rename so can move forward
        blockIDs_simple = blockIDs.copy()
        blockIDs = blockIDs_repeats
            
        
        
    # test if sweep block is missing
    if len(d['block'].unique()) + 1 == len(blockIDs):
        blockIDs = blockIDs[1:]
    dp['blockID'] = dp['block'].map(dict(zip(range(1, len(blockIDs)+1), blockIDs)))
    # now, split things up
    db = dp.loc[dp['blockID']=='bg', :].copy()
    dr = dp.loc[dp['blockID'] == 'meas',:].copy()

    if 'sweep' in blockIDs:
        # extract sweep block
        dsweep = dp.loc[dp['blockID'] == 'sweep',:].copy()
        # remove sweep peaks from peak list
        peakIDs_obs_full = peakIDs_obs[:]
        peakIDs_obs = [i for i in peakIDs_obs_full if 'Collector' not in i]
        peakIDs_cln = [i for i in peakIDs_cln if 'Collector' not in i]

        # also drop these columns from dr, db, dseep
        dr = dr.dropna(how='all', axis=1)
        db = db.dropna(how='all', axis=1)
        dsweep = dsweep.dropna(how='all', axis=1)
    # now that clean, define basepeak 
    basePeak = peakIDs_obs[0]


    
    dr['measure_line'] = dr.loc[:,'measure_line']- dr.iloc[0,:]['measure_line']
    dr['meas_line'] = dr.loc[:, 'measure_line']
    dr = dr.set_index('meas_line')

    # if prompt_for_params:
    #     cycle_num, integration_time, integration_num = get_measurement_params()    
    dr = sort_by_cycles(dr, integration_num, cycle_num=cycle_num)
    if 'frag' in blockIDs:
        dfrag = dp.loc[dp['blockID'] == 'frag',:].copy()
        dfrag = sort_by_cycles(dfrag, integration_num)
    if 'bg_NH3' in blockIDs:
        dbNH3 = dp.loc[dp['blockID']=='bg_NH3', :].copy()
        dbNH3 = sort_by_cycles(dbNH3, integration_num)
    # 2. Process and apply backgrounds measurements
    if db.size > 0:   
        # prep columns for big background data merge
        db['measure_line'] = db.loc[:,'measure_line']- db.iloc[0,:]['measure_line']
        db['meas_line'] = db.loc[:,'measure_line']
        db = db.set_index('meas_line')

        try:
            dbr = sort_by_cycles(db, integration_num)
            dbr['is_outlier'] = False
            # do a double outlier test
            dbrg1 = dbr.groupby(['block', 'is_sample'], group_keys=True)[peakIDs].apply(zscore)
            dbr['is_outlier'] = (dbrg1 > zscoreCutoff).any(axis=1).droplevel([0, 1])
            dbrg2 = dbr.loc[~dbr['is_outlier']].groupby(['block', 'is_sample'], group_keys=True)[peakIDs].apply(zscore)
            dbr.loc[~dbr['is_outlier'], 'is_outlier'] = (dbrg2 > zscoreCutoff).any(axis=1).droplevel([0, 1])
            
            bgs = compare_sample_and_std_bgs(dbr, peakIDs)
            dbg = bgs.reset_index().groupby('is_sample')

        except(KeyError):
            # Catches case where background data are incomplete or missing
            print("Incomplete background data found...assigning backgrounds to zero")
            db_b = pd.DataFrame(data={'i15': [0.0, 0.0, 0.0, 0.0],
                                      'i16': [0.0, 0.0,0.0, 0.0],
                                      'i17': [0.0, 0.0,0.0, 0.0],
                                      'is_sample': [0, 0, 1, 1]})
            dbg = db_b.groupby('is_sample')
            prompt_for_backgrounds = True

        # 2.1 apply backgrounds
        bgsUsed = {}
        for thisPeak in peakIDs:
            make15N2Correction = False

            dr[thisPeak + '_raw'] = dr[thisPeak].copy()
            # apply bgs
            # first, check if backgrounds are significantly different at 3sigma level:
            tbg = dbg[thisPeak]
            
  
            bgDiff = np.abs(tbg.mean()[1] - tbg.mean()[0])
            bgThresh = (3*tbg.std()/np.sqrt(tbg.count())).mean()
            
            if thisPeak=='i30':
                filesInFolder = os.listdir()
                N2files =  [i for i in filesInFolder if '_N2_corr.csv' in i]
                if len(N2files):
                    N2file = N2files[0]
                    print('N2 file found. Proceeding with 15N2 bg calibration.')
                    N2fits, dN2piv = compute_N2_contents(N2file)
                    make15N2Correction = True
                
            
 
            if bgDiff > bgThresh:
                print('Backgrounds on {0} are significantly different outside of '
                      'uncertainty \n std: '
                      '{1:.3f}, sa: {2:.3f}, pm {3:.3f}'.format(
                          thisPeak,
                          tbg.mean()[0],
                          tbg.mean()[1],
                          (2*tbg.std()/np.sqrt(tbg.count())).mean()))
                bckgrnd_choice = input('Apply (s)ame, (d)ifferent, '
                                        'or (c)ustom background corrections? ').lower()
                # then, apply to sample and standard separately
                if bckgrnd_choice == 'd':
                    bgfs = [tbg.mean()[0], tbg.mean()[1]]
   
                elif bckgrnd_choice == 'c':
                    manual_background_std = input(
                        'Input a background value for peak {0} on wg, '
                        'or press enter to calculate automatically...'.format(
                            thisPeak)).strip()
                    manual_background_sa = input(
                        'Input a background value for peak {0} on sample, '
                        'or press enter to calculate automatically...'.format(
                            thisPeak)).strip()
                    if len(manual_background_std) > 0:
                        bgfs = [manual_background_std, manual_background_sa]


                else:
                    # otherwise, apply the same to both
                    bgfs = [tbg.mean().mean(), tbg.mean().mean()]

                 
            else:
                # otherwise, apply the same to both
                bgfs = [tbg.mean().mean(), tbg.mean().mean()]
            
            # add bgs to list
            bgsUsed[thisPeak] = bgfs
            if make15N2Correction:
                # fits = bgfs['fits']
                # tails = bgfs['tails']
                # bgscat = bgfs['scattered']
                dr[thisPeak+'_15N2bg'] = np.nan
                dr[thisPeak+'_scatbg'] = np.nan
                dr['percent_N2'] = np.nan
                dr['R15'] = np.nan

                for isSample in [0,1]: 
                    dr.loc[dr['is_sample']==isSample, thisPeak+'_scatbg'] = bgfs[isSample]                               
                    dr.loc[dr['is_sample']==isSample, thisPeak+'_15N2bg'] = dr.loc[
                        dr['is_sample']==isSample, 'i28']*N2fits[isSample][0] + N2fits[isSample][1]
                    
                    dr.loc[dr['is_sample']==isSample, 'percent_N2'] = dN2piv['percent_N2'].mean()[isSample]                              
                    dr.loc[dr['is_sample']==isSample, 'R15'] = dN2piv['R15'].mean()[isSample]                              


                dr[thisPeak] = dr[thisPeak + '_raw'] -  dr[thisPeak + '_15N2bg'] -  dr[thisPeak + '_scatbg']
                make15N2Correction=False
            else:  
                for isSample in [0,1]:                                
                    dr.loc[dr['is_sample']==isSample,thisPeak] = dr.loc[
                        dr['is_sample']==isSample,thisPeak + '_raw'] - bgfs[isSample]

            # if there's a sweep block, apply these bgs, too
            if 'sweep' in blockIDs:
                # assume sweep block is on the highest mass collector
                if thisPeak == peakIDs_obs[-1]:
                    dsweep['ReferenceCollector_raw'] = dsweep['ReferenceCollector'].copy()
                    dsweep['ReferenceCollector'] -= bgfs[0]
            # then, compute ratio
            if thisPeak != basePeak:
                thisR = 'R' + thisPeak.strip('i_') + '_unfiltered'
                dr[thisR] = dr[thisPeak]/dr[basePeak]
    else:
        prompt_for_backgrounds = True
    if prompt_for_backgrounds:
        print('No background data detected')
        bgsUsed = {}
        for thisPeak in peakIDs:
            bgfs = []
            dr[thisPeak + '_raw'] = dr[thisPeak].copy()
            for isSample, gasID in enumerate(['std', 'sample']):
                while True:
                    thisBg = input('Input background for {0} on {1}: '.format(gasID, thisPeak))
                    try:
                        if len(thisBg)==0:
                            thisBg = 0
                        else:
                            thisBg = float(thisBg)
                    except(ValueError):
                        print('Not an interpretable bg value.\n Try again or press ENTER to skip')    
                    break
                bgfs.append(thisBg)
                dr.loc[dr['is_sample']==isSample,thisPeak] = dr.loc[
                    dr['is_sample']==isSample,thisPeak + '_raw'] - bgfs[isSample]
            bgsUsed[thisPeak] = bgfs
            if thisPeak != basePeak:
                thisR = 'R' + thisPeak.strip('i_') + '_unfiltered'
                dr[thisR] = dr[thisPeak]/dr[basePeak]

    dr['integration_time'] = integration_time
    # 3. Filter for outlier sub-integrations, calculate delta values
    dr = filter_out_signal_spikes(dr, peakIDs, integration_time, basedOn='_unfiltered')
    # Calculate confidence interval expected given the number of observations. 
    # I.e., sigma window in which all observations should fall
    proportional_confidence_interval = np.sqrt(2)*erfinv(
        1.0 - 1.0/(integration_num*2.0))
    dr = filter_for_outliers(dr, peakIDs, sigmaFilter=6, basedOn='_stable')
  
    dr = filter_for_max_ratios(dr, peakIDs,
                               sigmaFilter=6, basedOn='_cln')
    drm = calculate_deltas(dr, peakIDs)

    
    # export sweep blocks for records
    if 'sweep' in blockIDs:
        export_sweep_block(dsweep)
    return(dr, drm, file_name)

def export_sweep_block(dsweep):
    dsweep.to_excel('sweepScans.xlsx')
    return

def calculate_k_factor(dr, dfrag, plot_results=False):
    """
    Computes the k-factor (-H fragmentation rate) using the data
    from the background scans. I.e., k where i_CH3 = k*i_CH4

    Parameters
    ----------
    dr : pd.DataFrame
        DataFrame containing all raw data, background-corrected.
    dbr : pd.DataFrame
        DataFrame of main backgrounds for i16.
    dbg_15 : pd.DataFrame
        DataFrame of backgrounds containing off-peak 13CH2 fragment.
    plot_results : bool, optional
        Whether or not to display k-factor data. The default is False.

    Returns
    -------
    k_factor_weighted : float
        The mean k-factor for the dataset.

    """
    frags = dfrag.loc[:,['measure_line','is_sample','i15']].copy()
    peaks = dr.loc[(dr['measure_line'].max()-dr['measure_line'] <5), [
        'measure_line','is_sample','i16']].copy()
    # divide up sample vs std  
    frags.loc[frags['is_sample'] == True,'i15_sa'] = frags.loc[
        (frags['is_sample'] == True) ,'i15']
    frags.loc[frags['is_sample'] == False,'i15_std'] = frags.loc[
        (frags['is_sample'] == False) ,'i15']
    # same for i16
    peaks.loc[peaks['is_sample'] == True,'i16_sa'] = peaks.loc[
        (peaks['is_sample'] == True) ,'i16']
    peaks.loc[peaks['is_sample'] == False,'i16_std'] = peaks.loc[
        (peaks['is_sample'] == False) ,'i16']
    # now, append frags to end of peaks. 
    # Timing is such that this works out perfectly because both peaks and
    # frags took the same amount of time
    # this corrects for bleedout
    pf = pd.concat([peaks, frags])
    if plot_results:
        fig, ax = plt.subplots(2, figsize=(8,6))
        for i in ['std', 'sa']:
            this_fit_16 = np.polyfit(pf['i16_'+ i].dropna().index,
                                     pf['i16_'+ i ].dropna(),1)
            pf['i16_fit_'+i] = this_fit_16[0]*pf.index + this_fit_16[1]     
            pf['k_factor_' + i] = pf['i15_'+i]/pf['i16_fit_' + i]
            print('{0} frag rate = {1:.4f} +/- {2:.4f}'.format(
                i, pf['k_factor_' + i].mean(), pf['k_factor_' + i].std()))
            ax[0].plot(pf.index,
                       pf['i16_' + i],
                       '.',
                       label='i16_' + i)
            ax[0].plot(pf.index,
                       pf['i15_' + i],
                       '.',
                       label='i15_' + i)
            ax[0].plot(pf.index,
                       pf['i16_fit_' + i],
                       '--',
                       label='i16_fit_' + i)
            ax[1].plot(pf.index,
                       pf['k_factor_'+i],
                       '.',
                       label='k_factor_'+i)
        ax[1].set_xlabel('index')
        ax[0].set_ylabel('i (cps)')
        ax[1].set_ylabel('frag rate')
        ax[0].legend(loc = 'best')
        ax[1].legend(loc = 'best')
        fig.savefig('frag_rate_calcs.pdf') 
    else:
        for i in ['std', 'sa']:
            this_fit_16 = np.polyfit(pf['i16_'+ i].dropna().index,
                                     pf['i16_'+ i ].dropna(),1)
            pf['i16_fit_'+i] = this_fit_16[0]*pf.index + this_fit_16[1]
            pf['k_factor_' + i] = pf['i15_'+i]/pf['i16_fit_' + i]
            print('{0} frag rate = {1:.4f} +/- {2:.4f}'.format(
                i, pf['k_factor_' + i].mean(), pf['k_factor_' + i].std()))
    k_factor_weighted = (
        pf['k_factor_std'].mean()*pf['k_factor_std'].count() + pf[
            'k_factor_sa'].mean()*pf['k_factor_sa'].count())/(
                pf['k_factor_std'].count() + pf['k_factor_sa'].count())
    print('mean frag rate = {0:.4f}'.format(k_factor_weighted))
    return(k_factor_weighted)
        
    

def sort_by_cycles(dr, integration_num, cycle_num=None):
    """
    Assigns global and local integration and aquisition numbers for each
    sub-integration, based on measurement metadata

    Parameters
    ----------
    dr : pd.DataFrame
        DataFrame to which new values are added.
    integration_num : int
        Number of consecutive sub-integrations per integration.
    cycle_num : int
        Number of sample-std cycles ber acquisition.

    Returns
    -------
    dr : pd.DataFrame
        DataFrame with new values.

    """
    dr[['integration_number', 'cycle_number', 'acq_number']] = np.nan
    for block in dr['block'].unique():
        # correct for increasing measure lines
        dr.loc[dr['block']==block, 'measure_line'] = dr.loc[dr['block']==block, 'measure_line'] - dr.loc[dr['block']==block, 'measure_line'].min()
        dr.loc[dr['block']==block, 'integration_number'] = (dr['measure_line']%integration_num).astype(int)
        dr.loc[dr['block']==block, 'measure_line'] = (dr.loc[dr['block']==block, 'measure_line']/integration_num).apply(np.floor).astype(int)
        if cycle_num is None:
            # assume no repeats, which means we can compute cycle numbers directly for each block
            cycle_num = int((len(dr.loc[dr['block']==block, 'measure_line'].unique())-1)/2)
        dr.loc[dr['block']==block, 'cycle_number'] = (dr.loc[dr['block']==block, 'measure_line']%(cycle_num*2+1)).astype(int)
        dr.loc[dr['block']==block, 'acq_number'] = (dr.loc[dr['block']==block, 'measure_line']/(cycle_num*2+1)).astype(int)
    dr['is_sample'] = (dr['cycle_number']%2).astype(int)
    return(dr)

def filter_out_signal_spikes(dr, peakIDs, integration_time, zscoreCutoff=30, basedOn='_unfiltered'):
    """
    Performs a robust linear regression on base peak to identify signal spikes

    Parameters
    ----------
    dr : pd.DataFrame
        DataFrame of measurement observations.
    peakIDs : list
        List of peaks, used to ID base peak.
    weightCutoff : float, optional
        Cutoff passed to RLM results, points less than weight are outliers 
        and excluded from subsequent tests. The default is 0.5.

    Returns
    -------
    dr : pd.Dataframe
        DataFrame with new columns for outliers added

    """
    # assume base peak is first peak in list
    basePeak = peakIDs[0]
    dr['signal_is_stable'] = True
    # loop through measure lines
    for block in dr['block'].unique():
        for measureLine in dr['measure_line'].unique():
            thisDr = dr.loc[(dr['block']==block) & (dr['measure_line']==measureLine), :]
            # get Xogdr
            X = thisDr['integration_number'].values
            X = sm.add_constant(X)
            # get Y
            yTrue = thisDr[basePeak].values
            # fit with robust linear model
            # use default settings for now
            resRLM = sm.RLM(yTrue, X,  M=sm.robust.norms.TrimmedMean()).fit()
            # find unstable ones
            shotNoise = np.sqrt(resRLM.params[0]*integration_time)
            unstableIntegrations = X[(np.abs(resRLM.resid*integration_time/shotNoise) > zscoreCutoff), 1]
            # apply to dr
            dr.loc[(dr['block']==block) & (dr['measure_line']==measureLine) & (
                dr['integration_number'].isin(unstableIntegrations)),
                'signal_is_stable'] = False
    
    # now, filter based on this condition
    sigs_to_rd = {}
    for peak in peakIDs[1:]:
        mass = peak.strip('i_')
        sigs_to_rd[peak] = ('R'+mass, 'd'+mass)
    # loop through and apply filter
    for i in sigs_to_rd.keys():
        if i in dr.columns:
            filter_ratio,d = sigs_to_rd[i]
            filter_ratio_base = filter_ratio + basedOn
            filter_ratio_applied = filter_ratio + '_stable'
            dr[filter_ratio_applied] = dr[filter_ratio_base].copy()
            dr.loc[~dr['signal_is_stable'], filter_ratio_applied] = np.nan
    return(dr)

def filter_for_outliers(dr, peakIDs, sigmaFilter=6, basedOn = '_stable'):
    """
    Perfoms a simple outlier test on populations of sub-integrations

    Parameters
    ----------
    dr : pd.DataFrame
        DataFrame, background-corrected
    sigma_filter : float, optional
        z-score of outlier filter to apply. The default is 3.

    Returns
    -------
    dr : pd.Dataframe
        DataFrame with new columns for outliers added

    """
    # dg =dr.groupby('measure_line')
    # drm = dr.set_index('measure_line')
    # basePeak = peakIDs[0]
    # sigs_to_rd = {}
    filterRatiosBased = []
    filterRatiosApplied = []
    for peak in peakIDs[1:]:        
        mass = peak.strip('i_')
        # sigs_to_rd[peak] = ('R'+mass, 'd'+mass)
        filterRatiosBased.append('R'+ mass + basedOn)
        filterRatiosApplied.append('R'+ mass + '_cln')
    
    for i in range(len(filterRatiosBased)):
        filtRatBase = [filterRatiosBased[i]]
        filtRatApply = [filterRatiosApplied[i]]
        dg = dr.groupby(['block', 'is_sample'], group_keys=True)
        Rmedians = dg[filtRatBase].median()
        imedians = dg[peakIDs].median()
        # shot noise std dev of ratios
        shotNoiseRs = np.sqrt((1/(imedians*dg[['integration_time']].median().values)).sum(axis=1)).values*Rmedians.values.T
        # assign outliers to values more than 5 sigma from the medians
        zscores = (dr.pivot(columns=['is_sample'], values=filtRatBase) - Rmedians.values.T)/shotNoiseRs
        zscores.sum(axis=1) > sigmaFilter
        dr['is_outlier'] = (zscores.sum(axis=1).abs() > sigmaFilter)
        
        # dr['is_outlier'] = (dr.groupby(['block', 'measure_line'], group_keys=True)[filterRatiosBased].apply(zscore) > sigmaFilter).any(axis=1).droplevel([0,1])
        # dr.loc[~dr['is_outlier'], 'is_outlier'] = (dr.loc[~dr['is_outlier']].groupby(['block', 'measure_line'], group_keys=True)[filterRatiosBased].apply(zscore) > sigmaFilter).any(axis=1).droplevel([0,1])
        dr[filtRatApply] = dr[filtRatBase].copy()
        dr.loc[dr['is_outlier'], filtRatApply] = np.nan

    # sigs_to_rd = {'i16': ('R16','d16'), 'i17': ('R17','d17')}
    # for i in sigs_to_rd.keys():
        # # if i in dr.columns:
        #     filter_ratio,d = sigs_to_rd[i]
        #     filter_ratio_base = filter_ratio + basedOn
        #     filter_ratio_applied = filter_ratio + '_cln'
        #     dr['is_outlier'] = ((
        #         np.abs(drm[filter_ratio_base]-dg[filter_ratio_base].mean())
        #         )/dg[filter_ratio_base].std()> sigma_filter
        #         ).reset_index()[filter_ratio_base].copy()
            # dr[filter_ratio_applied] = dr[filter_ratio_base].copy()
            # dr.loc[dr['is_outlier'], filter_ratio_applied] = np.nan
    return(dr)
    
def filter_for_max_ratios(dr, peakIDs, sigmaFilter=5, dbr=[], nHighest=5, basedOn='_cln', makePlots=False):
    """
    Outlier test for off-peak drifts. Flags sub-integrations where measurement
    drifts off-peak, based on observing ratios below a certain threshold
    from the n-to-highest value.

    Parameters
    ----------
    dr : pd.DataFrame
        DataFrame with all data, background corrected.
    sigma_filter : float, optional
        z-score for ratio test. The default is 6.
    dbr : pd.DataFrame, optional
        background dataframe added to shot noise calculation. The default is [].

    Returns
    -------
    dr : pd.DataFrame
        DataFrame with off-peak sub-integrations flagged

    """
    idx = pd.IndexSlice
    
    basePeak = peakIDs[0]

    for peak in peakIDs[1:]:
        mass = peak.strip('i_')
        filtRatBase = 'R'+ mass + basedOn
        filtRatApply = 'R'+ mass + '_on_peak'
        iPeaks = [basePeak, peak]
    
        
    
        # again to get median R per acq
        dg = dr.groupby(['block', 'acq_number', 'is_sample'], group_keys=True)
        Rmed = dg[filtRatBase].median().rename('R_med')
        # also get mode for each acq, which is more robust to outliers than medians in skewed distribution
        # first, fit each distribution as a skewnorm
        skewFits = dg[filtRatBase].apply(lambda x: skewnorm.fit(x.dropna()))
        # next, get an xrange for each data set
        Rranges = dg[filtRatBase].apply(lambda x: np.linspace(x.min(), x.max(), num=int(1e4)))
        # combine these into a df
        skewModes = pd.DataFrame(data={'params': skewFits, 'x': Rranges})
        # compute pdf
        skewModes['pdf'] = skewModes.apply(lambda y: skewnorm.pdf(y.x, *y.params), axis=1)
        Rmode = skewModes.apply(lambda y: y.x[np.argmax(y.pdf)], axis=1).rename('R_mode')
        # counts medians, just used for shot noise calculation
        imeds= dg[iPeaks].median()
        # shot noise std dev of ratios
        relShotNoise = np.sqrt((1/(imeds*dg['integration_time'].median().median())).sum(axis=1))
        # relShotNoise = np.sqrt(((np.sqrt(imeds*dg['integration_time'].median().median())/imeds)**2).sum(axis=1))
    
        # dfRelShots = pd.DataFrame(data=relShotNoise, columns=[filtRatBase])
        shotNoise = (Rmode*relShotNoise).rename('R_shot_noise')
        
        # compute deltas for these bois
        deltaModes = (Rmode[idx[:,:,1]]/Rmode[idx[:, :, 0]]-1)*1000
        deltaMeds = (Rmed[idx[:,:,1]]/Rmed[idx[:, :, 0]]-1)*1000
    
        deltaModes = pd.DataFrame(deltaModes.rename('delta_mode'))
        deltaModes['is_sample'] = 1
        dms = deltaModes.set_index('is_sample', append=True)
    
        grps = dg[filtRatBase].groups
            
            
        
        grpIndex = ['block', 'acq_number', 'cycle_number', 'is_sample']
        dg2 = dr.groupby(grpIndex, group_keys=False)
        nRoll = 5
        dgRoll = dg2[filtRatBase].rolling(nRoll, center=True).mean()
        dgRoll = pd.DataFrame(dgRoll)
        dgRoll = dgRoll.merge(Rmode, how='left', left_index=True, right_index=True)
        dgRoll = dgRoll.merge(shotNoise, how='left', left_index=True, right_index=True)
        dgRoll['z_score'] = (dgRoll[filtRatBase]- dgRoll['R_mode'])/(dgRoll['R_shot_noise']/np.sqrt(nRoll))
        dgRoll['is_on_peak'] = np.abs(dgRoll['z_score']) < sigmaFilter
        # invert to make is_off_peaks
        dgRoll['is_off_peak'] = ~dgRoll['is_on_peak']
        # reset index
        dRoll = dgRoll.reset_index()
        dRoll = dRoll.set_index('meas_line')
        # merge just the key col back into main df
        if 'is_off_peak' in dr.columns:
            dr = dr.drop(columns='is_off_peak')
        dr = dr.merge(dRoll['is_off_peak'], how='left', left_index=True, right_index=True)
        # cleanup: because of how the roll works, first n before the roll starts are assigned nan
        # if first one is, nan, assume the previous are, too (becuase integration started on a bad peak)
        goodStarts = dr.loc[(dr['integration_number'] ==int(nRoll/2)) & (~dr['is_off_peak']), 'measure_line'].values
        dr.loc[(dr['measure_line'].isin(goodStarts)) & (dr['integration_number'] < int(nRoll/2)), 'is_off_peak'] = False
        totalInts = dr['integration_number'].max()
        goodEnds = dr.loc[(dr['integration_number'] == (totalInts - int(nRoll/2))) & (~dr['is_off_peak']), 'measure_line'].values
        dr.loc[(dr['measure_line'].isin(goodEnds)) & (dr['integration_number'] > (totalInts - int(nRoll/2))), 'is_off_peak'] = False


        
        dr[filtRatApply] = dr[filtRatBase].copy()
        dr.loc[dr['is_off_peak'],filtRatApply] = np.nan
        
        # do one more groupby with cleaned data
        dg = dr.groupby(['block', 'acq_number', 'is_sample'], group_keys=True)
        RmedOnPeak = dg[filtRatApply].median().rename('R_med_on_peak')
        deltaMedOnPeak = (RmedOnPeak[idx[:,:,1]]/RmedOnPeak[idx[:, :, 0]]-1)*1000
        # refit modes after filtering
        skewFits = dg[filtRatApply].apply(lambda x: skewnorm.fit(x.dropna()))
        # next, get an xrange for each data set
        Rranges = dg[filtRatApply].apply(lambda x: np.linspace(x.min(), x.max(), num=int(1e4)))
        # combine these into a df
        skewModes = pd.DataFrame(data={'params': skewFits, 'x': Rranges})
        # compute pdf
        skewModes['pdf'] = skewModes.apply(lambda y: skewnorm.pdf(y.x, *y.params), axis=1)
        RmodeOnPeak = skewModes.apply(lambda y: y.x[np.argmax(y.pdf)], axis=1).rename('R_mode_on_peak')
        deltaModeOnPeak = (RmodeOnPeak[idx[:,:,1]]/RmodeOnPeak[idx[:, :, 0]]-1)*1000

        dms['delta_median'] = deltaMeds
        dms['delta_median_on_peak'] = deltaMedOnPeak
        dms['delta_mode_on_peak'] = deltaModeOnPeak

    
        # combine into a df and save
        dmSum = pd.concat([imeds, Rmode, Rmed, RmedOnPeak, RmodeOnPeak, dms], axis=1)
        dmSum.to_excel('delta{0}_by_acq.xlsx'.format(mass), merge_cells=False)
        
        if makePlots:
            # make compute deltas and make a df of this to export bc possibly more robust
            fig, ax = plt.subplots(sharex='col', nrows=len(dr['acq_number'].unique()), ncols=2, figsize=(8,len(grps)), sharey=True)
            xBins = [np.linspace(dr.loc[dr['is_sample']==0, filtRatBase].min(), dr.loc[dr['is_sample']==0, filtRatBase].max(), 50),
                     np.linspace(dr.loc[dr['is_sample']==1, filtRatBase].min(), dr.loc[dr['is_sample']==1, filtRatBase].max(), 50)]
            for grpID in grps:
                thisAx = ax[int(grpID[1]),int(grpID[2])]
                theseData = dg.get_group(grpID)
                thisAx.plot('x', 'pdf', '-', data=skewModes.loc[grpID], color='C{0}'.format(grpID[2]))
                thisAx.hist(theseData[filtRatBase], bins=xBins[grpID[2]], color='C{0}'.format(grpID[2]), alpha=0.4, density=True)
                thisAx.hist(theseData[filtRatApply], bins=xBins[grpID[2]], color='C{0}'.format(grpID[2]), alpha=0.4, density=True)
        
                # plot median, mode, and skewnorm fit
                yLims = thisAx.get_ylim()
                thisAx.plot([Rmode[grpID], Rmode[grpID]], yLims, ':',color='C{0}'.format(grpID[2]))
                thisAx.plot([Rmed[grpID], Rmed[grpID]], yLims, '--',color='C{0}'.format(grpID[2]))
                thisAx.set_ylim(*yLims)
                if grpID[2]==1:
                    thisAx.text(0.95, 0.95, r'$\delta_{{mode}}$:{0:.2f}‰'.format(dms.loc[grpID, 'delta_mode'])
                                +'\n'+ r'$\delta_{{median}}$:{0:.2f}‰'.format(dms.loc[grpID, 'delta_median'])
                                + '\n' + r'$\delta_{{mode, on peak}}$:{0:.2f}‰'.format(dms.loc[grpID, 'delta_mode_on_peak'])
                                + '\n' + r'$\delta_{{median, on peak}}$:{0:.2f}‰'.format(dms.loc[grpID, 'delta_median_on_peak']),
                                transform=thisAx.transAxes, ha='right', va='top')
            title1= r'$\delta_{{mode}}$:{0:.2f} ± {1:.2f}‰'.format(
                dms['delta_mode'].mean(),dms['delta_mode'].std()/np.sqrt(len(dms['delta_mode'])))
            title2 = r'$\delta_{{median}}$:{0:.2f} ± {1:.2f}‰'.format(
                dms['delta_median'].mean(),dms['delta_median'].std()/np.sqrt(len(dms['delta_median'])))
            title3 = r'$\delta_{{mode, on peak}}$:{0:.2f} ± {1:.2f}‰'.format(
                dms['delta_mode_on_peak'].mean(),dms['delta_mode_on_peak'].std()/np.sqrt(len(dms['delta_mode_on_peak'])))
    
            title4 = r'$\delta_{{median, on peak}}$:{0:.2f} ± {1:.2f}‰'.format(
                dms['delta_median_on_peak'].mean(),dms['delta_median_on_peak'].std()/np.sqrt(len(dms['delta_median_on_peak'])))
            fig.suptitle(title1 + '\n' + title2 + '\n' + title3 + '\n' + title4)
            fig.savefig('delta{0}_by_acq.pdf'.format(mass))
    
    
    return(dr)


def compare_sample_and_std_bgs(dbr, peakIDs_obs, group_by=['block', 'cycle_number', 'is_sample'], isNH3=False):
    dbr_groups = dbr.loc[~dbr['is_outlier']].groupby(group_by)
    bg_mean = dbr_groups[peakIDs_obs].mean()
    bg_std = dbr_groups[peakIDs_obs].std()
    bg_se = bg_std/np.sqrt(dbr_groups[peakIDs_obs].count())
    
    bgs = pd.merge(bg_mean, bg_std, how='left', left_index=True, right_index=True, suffixes=['', '_std'])
    bgs = pd.merge(bgs, bg_se, how='left', left_index=True, right_index=True, suffixes=['', '_se'])
    # save bgs
    bglabel = 'backgrounds_'
    if isNH3:
        bglabel +='NH3_'
    while True:
        try:
            dbr.to_excel(bglabel + 'all.xlsx')
            break
        except(PermissionError):
            close_sheet = input(
                'Spreadsheet: backgrounds_all.xlsx is open. '
                '\n Close it and press ENTER to continue... ')
    # save bgs    
    while True:
        try:
            bgs[bgs.columns.sort_values()].to_excel(bglabel + 'mean.xlsx')
            break
        except(PermissionError):
            close_sheet = input(
                'Spreadsheet: backgrounds_mean.xlsx is open. '
                '\n Close it and press ENTER to continue... ')
    return(bgs)    

def calculate_deltas(dr, peakIDs):
    """
    Compute delta values from sample-standard cycles

    Parameters
    ----------
    dr : pd.DataFrame
        DataFrame containing all data, background-corrected.

    Returns
    -------
    drm : pd.DataFrame
        New DataFrame averaged per integration, including delta values based
        on std-sa-std brackets.

    """
    basePeak = peakIDs[0]
    sigs_to_rd = {}
    for peak in peakIDs[1:]:
        mass = peak.strip('i_')
        sigs_to_rd[peak] = ('R'+mass, 'd'+mass)
    # sigs_to_rd = {'i16': ('R16','d16'), 'i17': ('R17','d17')}
    cols_needed = [ 'cycle_number', 'acq_number',
                    'integration_time', basePeak, 'percent_N2', 'R15']
    for i in sigs_to_rd.keys():
        if i in dr.columns:
            r,d = sigs_to_rd[i]             
            dr[r+'_std'] = dr.loc[dr['is_sample'] == 0, r + '_stable']
            dr[r+'_sample'] = dr.loc[dr['is_sample'] == 1, r + '_stable']
            cols_needed.extend([i,r+'_unfiltered', r+'_on_peak', r+'_stable', r+'_cln'])
    dg =dr.groupby(['block', 'measure_line', 'is_sample'])
    drm = pd.DataFrame(data = dg[cols_needed].mean())
    drm = drm.reset_index()
    
    drm['P_imbalance'] = np.nan
    i_sample = drm.loc[drm['is_sample']==True, :].index
    drm['percent_on_peak'] = (dr.loc[(dr['is_off_peak'] == False) & (
        dr['signal_is_stable'])].groupby(['block','measure_line', 'is_sample'])[
            basePeak].count()/dg[
                basePeak].count()*100).values
    # also compute shot noise multiplier per row
    medCounts = dg[['integration_time']].median().values*dg[[peakIDs[0], peakIDs[-1]]].median()
    shotNoise = np.sqrt((1/medCounts).sum(axis=1))*dg[r+'_stable'].median()
    drm['shot_noise_multiplier'] = (dg[r+'_stable'].std()/shotNoise).values
    
    while True:
        try: 
            drm.loc[i_sample, 'P_imbalance'] = (drm.loc[i_sample, basePeak]/(
                (drm.loc[i_sample-1, basePeak].values + drm.loc[i_sample+1,basePeak].values
                 )/2)-1)*100
            break
        except(KeyError):
            # occurs if measurement fails in the middle of a cycle
            print('Incorrect number of sample/std cycles. \nDropping last row and trying again')
            # drop last sample row, try again
            i_sample = i_sample[:-1]
            
        
    for i in sigs_to_rd.keys():
        if i in dr.columns:
            r,d = sigs_to_rd[i]             
            dr[r+'_std'] = dr.loc[dr['is_sample'] == 0, r+'_stable']
            dr[r+'_sample'] = dr.loc[dr['is_sample'] == 1, r+'_stable']
            for to_append in ['_stable', '_on_peak', '_cln', '_unfiltered']:
                this_d = d + to_append
                this_r = r + to_append
                drm[this_d] = np.nan
                drm.loc[i_sample, this_d] = (drm.loc[i_sample, this_r]/(
                    (drm.loc[i_sample-1,this_r].values + drm.loc[
                        i_sample+1,this_r].values)/2)-1)*1000

    return(drm)

def export_data(dr, drm, file_name, peakIDs):
    """
    Exports all data.

    Parameters
    ----------
    dr : pd.DataFrame
        DataFrame with data at a sub-integration level.
    drm : pd.DataFrame
        DataFrame averaged per integration, includes delta values.
    file_name : str
        Name of file.


    Returns
    -------
    None.

    """
    basePeak = peakIDs[0]
    sigs_to_rd = {}
    for peak in peakIDs[1:]:
        mass = peak.strip('i_')
        sigs_to_rd[peak] = ('R'+mass, 'd'+mass)

    cols_to_export = ['measure_line', 'block', 'meas', 'cycle_number',
                      'is_sample', 'integration_number', 'is_outlier',
                      'is_off_peak', basePeak + '_raw', basePeak]
    # sigs_to_rd = {'i16': ('R16','d16'), 'i17': ('R17','d17')}
    for i in sigs_to_rd.keys():
        if i in dr.columns:
            r,d = sigs_to_rd[i]
            cols_to_export.extend([i+'_raw', i, r+'_unfiltered', r+'_stable',
                                   r+'_cln', r+'_on_peak', r+'_std', r+'_sample'])
    
    cols_to_export_all = cols_to_export.copy()
    dr.to_excel('{0}_processed_export_all.xlsx'.format(file_name),
                index=False, columns=cols_to_export_all,
                freeze_panes=(1,0), header=True)
    return()

def parse_log_file(data_file):
    """
    Reads the Qtegra logfile and extracts data for pressure-balances
    and peak centering.

    Parameters
    ----------
    data_file : str
        filepath to data import.

    Returns
    -------
    d : pd.DataFrame
        DataFrame of all logfile data, one line per row
    d_pc : pd.DataFrame
        DataFrame of just the peak center data

    """
    data_folder=os.path.dirname(data_file)
    log_files = [i for i in os.listdir(data_folder) if 'logbook.xls' in i]

    if len(log_files):
        log_file_name = os.path.abspath(os.path.join(data_folder, log_files[0]))
        file_name = os.path.basename(log_file_name).split('.xlsx')[0]
    else:
        while True:
            log_file_name = input('drag excel log file... ').replace(
                '\\ ',' ').strip("'").strip('"').strip()
            log_file_name = os.path.abspath(log_file_name)
            if os.path.exists(log_file_name) and log_file_name.endswith('.xlsx'):
                file_name = os.path.basename(log_file_name).split('.xlsx')[0]
                break
            else:
                print('Not an .xlsx file ')
    d = pd.read_excel(log_file_name, header=None,
                      names=['Logged at', 'Level', 'Message',
                             'Time', 'Category', 'Sub Category'])
    p_adjust_start = 'Start Volume Adjust'
    p_adjust_stop = 'Pressure adjust finished'

    start = d.loc[d['Message'].str.contains(p_adjust_start),'Time']
    stop = d.loc[d['Message'].str.contains(p_adjust_stop),'Time']
    i_p_adjust = (d['Message'].str.contains(p_adjust_start) | d['Message'].str.contains(p_adjust_stop))

    i_peak_center = d['Message'].str.lower().str.contains(
        'peak center') & ~d['Message'].str.startswith('Found more than one peak')
    i_pc = d.loc[(i_peak_center) & (d['Level'].isin(
        ['UserInfo', 'UserError'])),:].index
    i_pc_success = d.loc[(i_peak_center) & (d['Level'] == 'UserInfo'),:].index
    if len(i_pc_success) > 0:
        match_string = '(-?[0-9]+.[0-9]+[E-]*[0-9]+)'
        mass_data = d.loc[i_pc_success,'Message'].str.extractall(match_string).unstack()[0]
        center_mass = mass_data[0]
        delta_offset = mass_data[1]
        mass_offset = mass_data[2]
        d['center_mass'] = np.nan
        d.loc[center_mass.index, 'center_mass'] = center_mass.astype(float)
        d['delta_offset'] = np.nan
        d.loc[delta_offset.index, 'delta_offset'] = delta_offset.astype(float)
        d['mass_offset'] = np.nan
        d.loc[mass_offset.index, 'mass_offset'] = mass_offset.astype(float)
        d.loc[i_pc, ['Time', 'Message', 'center_mass', 'delta_offset',
                     'mass_offset']].to_excel(
                         '{0}_peak_center_data.xlsx'.format(file_name))
        d_pc = d.loc[i_pc, ['Time', 'Message', 'center_mass',
                            'delta_offset', 'mass_offset']]        
    d.loc[i_p_adjust, ['Time', 'Message']].to_excel(
        '{0}_pressure_adjust_data.xlsx'.format(file_name))
    return(d, d_pc)


def get_list_of_files_to_import():
    acq_name =  input('Drag all Qtegra files to process ').replace('\\ ', ' ').strip("'").strip('"').strip()
    acq_name=acq_name.strip().strip("'").strip('"')
    acq_name_list = acq_name.split(' ')
    acq_name_list = [l.strip(',').strip("'") for l in acq_name_list]
    return(acq_name_list)

###########################################################################
#
# Main script
#
###########################################################################


drs = []
drms = []
file_names = []

acq_name_list = get_list_of_files_to_import()
cycle_num, integration_time, integration_num, peakIDs, blockIDs, repeatBlocks = get_measurement_params(
    file_name=acq_name_list[0], auto_detect=True)
for i in acq_name_list:
    if not os.path.exists(os.path.join(os.path.dirname(i),'d_data_all_summary.xlsx')):
        os.chdir(os.path.dirname(i))
    if i.endswith('N2_corr.csv'):
        N2_fits = compute_N2_contents(i)
    else:
        dr, drm, file_name = process_Qtegra_csv_file(i, peakIDs, blockIDs, sigma_filter=3, prompt_for_params=False,
                                                     cycle_num=cycle_num, integration_time=integration_time,
                                                     integration_num=integration_num, prompt_for_backgrounds=False,
                                                     input_tail_D2_background=True, repeatBlocks=repeatBlocks)
        drs.append(dr)
        drms.append(drm)
        file_names.append(file_name)
        export_data(dr, drm, file_name, peakIDs)
        if input('parse log file for {0}: (y/n)? '.format(file_name)).lower() != 'n':
            log_file, peak_centers = parse_log_file(i)
            # test if peak centered every time
            colsToAdd= ['center_mass', 'delta_offset', 'mass_offset']
            if len(peak_centers) == len(drms[-1]):
                for i, col in enumerate(colsToAdd):
                    drms[-1].insert(i+5, col, peak_centers[col].values)
            # else, test if peak centered every acquisition
            elif len(peak_centers) == len(drms[-1]['acq_number'].unique()):
                peak_centers['acq_number'] = range(len(peak_centers))
                peak_centers['cycle_number'] = 0
                colsToAlign = ['acq_number', 'cycle_number']
                drms[-1] = drms[-1].merge(peak_centers[colsToAdd + colsToAlign],
                                          how='left', on=colsToAlign)
            # D17O now does two peak centers per acq, catch this, use the second one
            elif len(peak_centers)/2 == len(drms[-1]['acq_number'].unique()):
                peak_centers['acq_number'] = range(len(peak_centers))
                peak_centers['acq_number'] = peak_centers['acq_number']/2-0.5
                peak_centers['cycle_number'] = 0
                colsToAlign = ['acq_number', 'cycle_number']
                drms[-1] = drms[-1].merge(peak_centers[colsToAdd + colsToAlign],
                                          how='left', on=colsToAlign)
            # catch case where peak centers occur between successive std measurements
            elif len(peak_centers) - (drms[-1]['is_sample'].diff()==0).sum() < 3:
                peakCenterLocs = drms[-1].loc[(drms[-1]['is_sample'].diff()==0) | (drms[-1]['is_sample'].diff().isnull()), :].index
                peak_centers['measure_line_index'] = np.nan
                peakCentersLastX = peak_centers[-len(peakCenterLocs):].index
                peak_centers.loc[peakCentersLastX, 'measure_line_index'] = peakCenterLocs
                peak_centers = peak_centers.set_index('measure_line_index')
                drms[-1] = drms[-1].merge(peak_centers[colsToAdd], how='left', left_index=True, right_index=True)
             # catch case where one or two coarse peak centers before main event, deal with this
            elif len(peak_centers) - len(drms[-1]) < 5:
                nExtra =  len(peak_centers) - len(drms[-1])
                peak_centers = peak_centers[nExtra:]
                for i, col in enumerate(colsToAdd):
                    drms[-1].insert(i+5, col, peak_centers[col].values)
               
            else:
                try:
                    print('Data and peak centers are misaligned. Using only the first {0} rows of the peak center data'.format(len(drms[-1])))
                    peak_centers_short = peak_centers.iloc[:len(drms[-1]), :]
                    for i, col in enumerate(colsToAdd):
                        drms[-1].insert(i+5, col, peak_centers_short[col].values)
                    drms[-1]['max_delta_offset'] = np.nan
                    i_sample = drms[-1].loc[drms[-1]['is_sample'] == True].index
                    drms[-1].loc[i_sample, 'max_delta_offset'] = np.stack(
                        [drms[-1].loc[i_sample, 'delta_offset'].values, drms[-1].loc[
                            i_sample+1, 'delta_offset'].values]).max(axis = 0)
                except(ValueError):
                    print('Unable to align peak centers')

if len(drms)>0:    
    # consolidate all data frames
    dr_all = drs[0].copy()
    drm_all = drms[0].copy()
    for i in range(1,len(drs)):
        print(i)
        dr_all = dr_all.append(drs[i])
        drm_all = drm_all.append(drms[i])
    # re_index to preserve full order
    dr_all.reset_index(drop=False, inplace=True)    
    drm_all.reset_index(drop=True, inplace=True)
    dr_all.to_excel('d_data_all.xlsx', freeze_panes=(1,0), header=True)
    drm_all.to_excel('d_data_all_summary.xlsx', freeze_panes=(1,0), header=True)
    
    deltas = []
    yints = []
    yses = []
    slopes = []
    useY = []
    Rkeys = []
    Rs = []
    for peakID in peakIDs[1:]:
        delta = 'd{0}_on_peak'.format(peakID[1:])
        deltas.append(delta)
        fig, ax = plt.subplots()
        thisR = 'R{0}_on_peak'.format(peakID[1:])
        Rs.append(drm_all[thisR].mean())
        
        
        thisDr = drm_all[['P_imbalance', delta]].dropna(how='any').sort_values(by='P_imbalance')
        
        X = thisDr['P_imbalance'].values
        X = sm.add_constant(X)
        # get Y
        y = thisDr[delta].values
    
        resRLM = sm.RLM(y, X,  M=sm.robust.norms.TrimmedMean()).fit()
        resOLS = sm.OLS(y, X).fit()
        preds = resOLS.get_prediction(X)
        dfPreds = preds.summary_frame(alpha=0.05)
        
        ax.plot(X[:,1], y, '.', alpha=0.5)
        ax.plot(X[:,1], dfPreds['mean'], '-', color='grey')
        ax.plot(X[:,1], resRLM.predict(X), '--', color='grey')
        ax.fill_between(X[:,1], dfPreds['mean_ci_lower'], dfPreds['mean_ci_upper'], alpha=0.2, color='grey')
        
        ax.set_xlabel('P imbalance (%)')
        ax.set_ylabel(delta + ' (‰ vs. wg)')
        fig.savefig('{0}_vs_P-imbalance.pdf'.format(delta))
        
        
        # save y-intercept, SE
        yints.append(resOLS.params[0])
        yses.append(np.sqrt(np.diag(resOLS.cov_params()))[0])
        slopes.append(resOLS.params[1])
        # test if slope significantly different than 0
        useY.append(not ((0 >= resOLS.conf_int()[1,0]) and (0 <=resOLS.conf_int()[1,1])))
    
    dsum = pd.DataFrame(data={'r': Rs, 'delta':deltas, 'Y-intercept': yints,
                              'Y-intercept SE': yses, 'slope': slopes,
                              'use Y-int':useY}).set_index('delta')
    
    dsum['Mean'] = drm_all[deltas].mean()
    dsum['SD'] = drm_all[deltas].std()
    dsum['N'] =  drm_all[deltas].count()
    dsum['SE'] = dsum['SD']/np.sqrt(dsum['N'])
    dsum['file_name'] = file_name
    
    dspiv = pd.pivot(dsum.reset_index(), columns='delta', index='file_name').swaplevel(axis=1)     
    dspiv = dspiv.sort_index(axis=1)
    toMean = [peakIDs[0], 'P_imbalance']
    for key in toMean:
        dspiv[key] = drm_all[key].mean()
    dN2 = drm_all[['is_sample', 'percent_N2', 'R15']].groupby('is_sample').mean()
    dN2['file_name'] = file_name
    dN2s = pd.merge(dN2.loc[[0],:], dN2.loc[[1],:], on='file_name', suffixes=['_wg', '_sa']).set_index('file_name')
    dspiv[dN2s.columns] = dN2s

    dspiv.to_excel(file_name + '_summary_line.xlsx')    
        
