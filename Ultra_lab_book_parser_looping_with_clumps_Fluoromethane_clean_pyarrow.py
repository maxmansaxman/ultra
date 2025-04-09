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
import pandas as pd
import os
from scipy.special import erf, erfinv
plt.close('all')

def get_measurement_params(auto_detect=False, file_name=''):
    ''' Retreives three key measurement parameters (number of cylces,
    sub-integration time, number of sub-integrations) by either prompting for 
    input or assuming based on the filename.
    '''
    if auto_detect and len(file_name) > 0:
        # assumes parameters based on file name and standard configuration
        if '_dD_' in file_name:
            (cycle_num, integration_time, integration_num) = (10, 0.524, 90)
            return(cycle_num, integration_time, integration_num)
        elif '_d13CD_' in file_name:
            (cycle_num, integration_time, integration_num) = (10, 1.048, 60)
            return(cycle_num, integration_time, integration_num)
        elif '_dD2_' in file_name:
            (cycle_num, integration_time, integration_num) = (10, 1.048, 60)
            return(cycle_num, integration_time, integration_num)
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
    return(cycle_num, integration_time, integration_num)

def import_qtegra_file_legacy(d_data_file):

    d = pd.read_csv(d_data_file, sep=None, engine='python')
    print('Valid file, now calculating...')
    dt = d.transpose()
    dt['f']= dt[3].astype(float, errors='ignore')
    # use regexp to extract intensity data from each row
    i = dt['f'].str.extract(r'(-*[0-9]+\.[0-9]+)',
                            expand=False).dropna().index
    sigs = dt.loc[i,'f'].astype(float)
    meta = pd.DataFrame(dt.loc[i,1].str.split(':').values.tolist(),
                        index=i,
                        columns=['block', 'meas', 'peak'])
    measure_line = dt.loc[i,0].astype(int)
    d = pd.concat([measure_line, meta, sigs], axis = 1)
    d['block'] = d['block'].astype(int)
    # measurements with backgrounds will contain multiple blocks
    # so, assume largest block is a measurement block,
    # and all others are backgrounds
    measure_block = d.groupby('block').count().idxmax()['peak'] 
    db = d.loc[d['block']!=measure_block, :]
    d = d.loc[d['block'] == measure_block,:]
    dr = pd.DataFrame(data = d.loc[d['peak'] == '12CH3',:])
    dr.columns = ['measure_line', 'block', 'meas', 'peak', 'i15']
    dr['measure_line'] = dr['measure_line']- dr.iloc[0,:]['measure_line']
    dr['block'] = dr['block'].astype(int)
    dr['meas_line'] = dr['measure_line'].copy()
    dr.set_index('meas_line', inplace = True)
    return(d, db, dr)

def import_qtegra_file_pyarrow(d_data_file):
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
    basePeak = peakIDs_obs[0]
    dc = dn.loc[(dn[3]=='Y [cps]'), [1, 'block', 'meas', 'peak', 4]].copy()
    #rename columns to match legacy processer
    d = dc.rename(columns={1: 'measure_line', 4:'f'})
    d['f'] = d['f'].astype(float)
    d['measure_line'] = d['measure_line'].astype(int)
    d = d.sort_values(by=['measure_line', 'peak'])
    d['block'] = d['block'].astype(int)
    # commands that could make processing simpler later
    dp = d.pivot(columns='peak', values='f', index='measure_line')
    dp = dp.reset_index()
    dp = dp.merge(d.loc[d['peak']==basePeak,['measure_line', 'block', 'meas']],
                  how='left', on='measure_line')
    dp[peakIDs_obs] = dp[peakIDs_obs].astype(float)
    measure_block = dp.groupby('block').count().idxmax()[basePeak] 
    db = d.loc[d['block']!=measure_block, :].copy()
    d = d.loc[d['block'] == measure_block,:]
    
    dr = pd.DataFrame(data = d.loc[d['peak'] == '12CH3',:])
    dr.columns = ['measure_line', 'block', 'meas', 'peak', 'i15']  
    dr['measure_line'] = dr['measure_line']- dr.iloc[0,:]['measure_line']
    dr['meas_line'] = dr['measure_line'].copy()
    d['meas_line'] = d['measure_line'].copy()
    dr = dr.set_index('meas_line')
    return(d, db, dr)
    
def process_Qtegra_csv_file(d_data_file, prompt_for_params=False,
                            integration_time=16.7, integration_num=10,
                            cycle_num=10, sigma_filter=2,
                            prompt_for_backgrounds=True,
                            input_tail_D2_background=False, promptForPyarrow=True,
                            usePyarrow=False):
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
    if promptForPyarrow:
        pyarChoice = input("Press l if a (l)egacy Qtegra export file or any other key to continue...").strip().lower()
        if pyarChoice =='l':
            usePyarrow = False
        else:
            usePyarrow = True
    if usePyarrow:
        d, db, dr = import_qtegra_file_pyarrow(d_data_file)
    else:
        d, db, dr = import_qtegra_file_legacy(d_data_file)
    if prompt_for_params:
        cycle_num, integration_time, integration_num = get_measurement_params()    
    dr = sort_by_cycles(dr, integration_num, cycle_num)
    
    # 2. Process and apply backgrounds measurements
    if db.size > 0:
        dbr = pd.DataFrame(data = db.loc[db['peak'].isin(
            ['b_i15', 'b2_i15','frag13_i15', '12CH3_plusFrag', '12CH3']),:])
        dbr.columns = ['measure_line', 'block', 'meas', 'peak', 'i15']
        dbr['measure_line'] = dbr['measure_line'] - dbr.iloc[0,:]['measure_line']
        dbr['block'] = dbr['block'].astype(int)
        dbr['meas_line'] = dbr['measure_line'].copy()
        dbr.set_index('meas_line', inplace = True)
        background_masses = ['i16', 'i17']
        # prep columns for big background data merge
        db['meas_line'] = db['measure_line'].copy()
        db['meas_line'] = db['meas_line']- db.iloc[0,:]['meas_line']

        for background_mass in background_masses:
            peaks_to_merge = []
            for p in db['peak'].unique():
                if background_mass in p:
                    peaks_to_merge.append(p)
            dbr = dbr.join(db.loc[
                db.peak.isin(peaks_to_merge),['meas_line','f']
                ].set_index('meas_line').rename(
                    columns={'f': background_mass}), how='left'
                    ).fillna(value=0.0)
            
        # need to treat backgrounds differently if clumped 13C, or D
        b_blocks = dbr['block'].unique()
        # if more than one background block
        if (len(b_blocks) >1) :
            dbr = sort_by_cycles(dbr, integration_num, 3)
            dbr['is_outlier'] = False
            #first, filter for signal stability per sample block
            dbr = filter_for_signal_stability(dbr, integration_num,
                                              sigma_filter=3,
                                              count_backwards=30,
                                              signals_to_use=['i15'],
                                              b_block=b_blocks[-1])
            # then again per sample or std
            dbr = filter_for_signal_stability_per_side(dbr,
                                                       sigma_filter=2,
                                                       count_backwards=2,
                                                       signals_to_use=['i15'],
                                                       b_block=b_blocks[-1])
            # now the same on mass 16, on second-to-last background block
            dbr = filter_for_signal_stability(dbr, integration_num,
                                              sigma_filter=3,
                                              count_backwards=30,
                                              signals_to_use=['i16'],
                                              b_block = b_blocks[-2])
            dbr = filter_for_signal_stability_per_side(dbr,
                                                       sigma_filter=2,
                                                       count_backwards=2,
                                                       signals_to_use=['i16'],
                                                       b_block=b_blocks[-2])

            # i_15 background comes from last one
            dbg_15 = dbr.loc[(dbr['block'] == b_blocks[-1]) & (
                dbr['is_outlier'] == False),:].groupby('is_sample')
            # i16 and i17 will from second to last
            dbg = dbr.loc[(dbr['block'] == b_blocks[-2]) & (
                dbr['is_outlier'] == False),:].groupby('is_sample')
        # else, everybody comes from same one
        else:
            try:
                dbr = sort_by_cycles(dbr, integration_num, 2)
                dbr['is_outlier'] = False
                #first, filter for signal stability per sample block
                dbr = filter_for_signal_stability(dbr, integration_num,
                                                  sigma_filter=3,
                                                  count_backwards=30,
                                                  signals_to_use=['i15'],
                                                  b_block=b_blocks[-1])
                # then again per sample or std
                dbr = filter_for_signal_stability_per_side(dbr,
                                                           sigma_filter=2,
                                                           count_backwards=2,
                                                           signals_to_use=['i15'],
                                                           b_block=b_blocks[-1])
                dbg_15 = dbr.loc[(dbr['block'] == b_blocks[-1]) & (
                    dbr['is_outlier'] == False),:].groupby('is_sample')
                dbg = dbr.loc[(dbr['block'] == b_blocks[-1]) & (
                    dbr['is_outlier'] == False),:].groupby('is_sample')
            except(KeyError):
                # Catches case where background data are incomplete or missing
                print("Incomplete background data found...assigning backgrounds to zero")
                db_b = pd.DataFrame(data={'i15': [0.0, 0.0, 0.0, 0.0],
                                          'i16': [0.0, 0.0,0.0, 0.0],
                                          'i17': [0.0, 0.0,0.0, 0.0],
                                          'is_sample': [0, 0, 1, 1]})
                dbg = db_b.groupby('is_sample')
                dbg_15 = db_b.copy().groupby('is_sample')
                prompt_for_backgrounds = True

        # 2.1 apply mass 15 backgrounds
        # first, copy old
        dr['i15_raw'] = dr['i15'].copy()
        # first, check if backgrounds are significantly different at 3sigma level:
        if np.abs(dbg_15['i15'].mean()[1] - dbg_15['i15'].mean()[0]
                  ) > (3*dbg['i15'].std()/np.sqrt(dbg['i15'].count())).sum():
            print('Backgrounds on {0} are significantly different outside of '
                  'uncertainty \n std: '
                  '{1:.3f}, sa: {2:.3f}, pm {3:.3f}'.format(
                      '12CH3',
                      dbg['i15'].mean()[0],
                      dbg['i15'].mean()[1],
                      (2*dbg['i15'].std()/np.sqrt(dbg['i15'].count())).sum()))
            bckgrnd_choice = input('Apply (s)ame, (d)ifferent, '
                                   'or (c)ustom background corrections? ').lower()
            # then, apply to sample and standard separately
            if bckgrnd_choice == 'd':
                dr.loc[dr['is_sample']==False,'i15'] = dr.loc[
                    dr['is_sample']==False,'i15_raw'] - dbg_15['i15'].mean()[0]
                dr.loc[dr['is_sample']==True,'i15'] = dr.loc[
                    dr['is_sample']==True,'i15_raw'] - dbg_15['i15'].mean()[1]
            elif bckgrnd_choice == 'c':
                manual_background_std = input(
                    'Input a background value for peak {0} on wg, '
                    'or press enter to calculate automatically...'.format(
                        '12CH3')).strip()
                manual_background_sa = input(
                    'Input a background value for peak {0} on sample, '
                    'or press enter to calculate automatically...'.format(
                        '12CH3')).strip()
                if len(manual_background_std) > 0:
                    dr.loc[dr['is_sample']==False,'i15'] = dr.loc[
                        dr['is_sample']==False,'i15_raw'] - float(manual_background_std)
                    dr.loc[dr['is_sample']==True,'i15'] = dr.loc[
                        dr['is_sample']==True,'i15_raw'] - float(manual_background_sa)

            else:
                # otherwise, apply the same to both
                dr.loc[dr['is_sample']==False,'i15'] = dr.loc[
                    dr['is_sample']==False,'i15_raw'] - dbg_15['i15'].mean().mean()
                dr.loc[dr['is_sample']==True,'i15'] = dr.loc[
                    dr['is_sample']==True,'i15_raw'] - dbg_15['i15'].mean().mean()
             
        else:
            # otherwise, apply the same to both
            dr.loc[dr['is_sample']==False,'i15'] = dr.loc[
                dr['is_sample']==False,'i15_raw'] - dbg_15['i15'].mean().mean()
            dr.loc[dr['is_sample']==True,'i15'] = dr.loc[
                dr['is_sample']==True,'i15_raw'] - dbg_15['i15'].mean().mean()   
        while True:
            try:
                dbr.to_excel('backgrounds_clump.xlsx')
                break
            except(PermissionError):
                close_sheet = input(
                    'Spreadsheet: backgrounds_clump.xlsx is open. '
                    '\n Close it and press ENTER to continue... ')
                continue
        while True:
            try:
                dbg[['i15', 'i16', 'i17']].mean().to_excel(
                    'backgrounds_mean.xlsx')
                break
            except(PermissionError):
                close_sheet = input(
                    'Spreadsheet: backgrounds_mean.xlsx is open. '
                    '\n Close it and press ENTER to continue... ')
                continue
    else:
        manual_background_std = input(
            'Input a background value for peak {0} on wg, '
            'or press enter to calculate automatically...'.format(
                '12CH3')).strip()
        manual_background_sa = input(
            'Input a background value for peak {0} on sample, '
            'or press enter to calculate automatically...'.format(
                '12CH3')).strip()
        if len(manual_background_std) > 0:
            dr['i15_raw'] = dr['i15'].copy()
            dr.loc[dr['is_sample']==False,'i15'] = dr.loc[
                dr['is_sample']==False,'i15_raw'] - float(manual_background_std)
            dr.loc[dr['is_sample']==True,'i15'] = dr.loc[
                dr['is_sample']==True,'i15_raw'] - float(manual_background_sa)

    d.rename(columns={0:'meas_line'}, inplace=True)
    for peak in d['peak'].unique():
        this_peak = 'i_' + peak 
        # one-line to find peaks and merge, with correct name
        # and fill nas as 0.0
        dr = dr.join(d.loc[d.peak == peak,['meas_line','f']].set_index(
            'meas_line').rename(columns={'f': this_peak}), how='left'
            ).fillna(value=0.0)
        # 2.2 Apply mass 16 backggrounds
        if peak in ['13CH3', '12CH2D', '12CH4']:        
            dr['i16_raw'] = dr[this_peak]
            dr['i16'] = dr['i16_raw'].copy()
            # if background scans processed:
            if db.size > 0:
                if prompt_for_backgrounds:
                    manual_background_std = input(
                        'Input a background value for peak {0} on wg, '
                        'or press enter to calculate automatically...'.format(
                            peak)).strip()
                    manual_background_sa = input(
                        'Input a background value for peak {0} on sample, '
                        'or press enter to calculate automatically...'.format(
                            peak)).strip()
                    if len(manual_background_std) > 0:
                        dr.loc[dr['is_sample']==False,'i16'] = dr.loc[
                            dr['is_sample']==False,'i16'] - float(manual_background_std)
                        dr.loc[dr['is_sample']==True,'i16'] = dr.loc[
                            dr['is_sample']==True,'i16'] - float(manual_background_sa)
                    else:
                        prompt_for_backgrounds = False
                if not prompt_for_backgrounds:
                    # first, check if backgrounds are significantly different at 3sigma level:
                    if np.abs(dbg['i16'].mean()[1] - dbg['i16'].mean()[0]) > (
                            3*dbg['i16'].std()/np.sqrt(dbg['i16'].count())).sum():
                        print('Backgrounds on {0} are significantly different '
                              'outside of uncertainty \n std: '
                              '{1:.3f}, sa: {2:.3f}, pm {3:.3f}'.format(
                                  peak,dbg['i16'].mean()[0],
                                  dbg['i16'].mean()[1],
                                  (2*dbg['i16'].std()/np.sqrt(dbg['i16'].count())).sum()))
                        bckgrnd_choice = input(
                            'Apply (s)ame or (d)ifferent background corrections? ').lower()
                        # then, apply to sample and standard separately
                        if bckgrnd_choice == 'd':
                            dr.loc[dr['is_sample']==False,'i16'] = dr.loc[
                                dr['is_sample']==False,'i16_raw'] - dbg['i16'].mean()[0]
                            dr.loc[dr['is_sample']==True,'i16'] = dr.loc[
                                dr['is_sample']==True,'i16_raw'] - dbg['i16'].mean()[1]
                        else:
                            # otherwise, apply the same to both
                            dr.loc[dr['is_sample']==False,'i16'] = dr.loc[
                                dr['is_sample']==False,'i16_raw'] - dbg['i16'].mean().mean()
                            dr.loc[dr['is_sample']==True,'i16'] = dr.loc[
                                dr['is_sample']==True,'i16_raw'] - dbg['i16'].mean().mean()    
                    else:
                        # otherwise, apply mean of both to each:
                        dr.loc[dr['is_sample']==False,'i16'] = dr.loc[
                            dr['is_sample']==False,'i16'] - dbg['i16'].mean().mean()
                        dr.loc[dr['is_sample']==True,'i16'] = dr.loc[
                            dr['is_sample']==True,'i16'] - dbg['i16'].mean().mean()
            else:
                manual_background_std = input(
                    'Input a background value for peak {0} on wg, '
                    'or press enter to calculate automatically...'.format(
                        peak)).strip()
                manual_background_sa = input(
                    'Input a background value for peak {0} on sample, '
                    'or press enter to calculate automatically...'.format(
                        peak)).strip()
                if len(manual_background_std) > 0:
                    dr.loc[dr['is_sample']==False,'i16'] = dr.loc[
                        dr['is_sample']==False,'i16'] - float(manual_background_std)
                    dr.loc[dr['is_sample']==True,'i16'] = dr.loc[
                        dr['is_sample']==True,'i16'] - float(manual_background_sa)

            dr['R16'] = dr['i16']/dr['i15']
        # 2.3 Apply mass 17 backgrounds
        elif peak in ['12CHD2','13CH2D','12CH2D2']:
            dr['i17_raw'] = dr[this_peak]
            dr['i17'] = dr['i17_raw'].copy()
             # if background scans processed:           
            if db.size > 0:
                if peak == '13CH2D':
                    # first, check if backgrounds are significantly different at 3sigma level:
                    if np.abs(dbg_15['i17'].mean()[1] - dbg_15['i17'].mean()[0]) > (
                            3*dbg_15['i17'].std()/np.sqrt(dbg_15['i17'].count())).sum():
                        print('Backgrounds on {0} are significantly different '
                              'outside of uncertainty \n std: '
                              '{1:.3f}, sa: {2:.3f}, pm {3:.3f}'.format(
                                  peak,dbg_15['i17'].mean()[0],
                                  dbg_15['i17'].mean()[1],
                                  (2*dbg_15['i17'].std()/np.sqrt(dbg_15['i17'].count())).sum()))
                        bckgrnd_choice = input(
                            'Apply (s)ame, (d)ifferent, or (c)ustom background corrections? ').lower()
                        # then, apply to sample and standard separately
                        if bckgrnd_choice == 'd':
                            dr.loc[dr['is_sample']==False,'i17'] = dr.loc[
                                dr['is_sample']==False,'i17_raw'] - dbg_15['i17'].mean()[0]
                            dr.loc[dr['is_sample']==True,'i17'] = dr.loc[
                                dr['is_sample']==True,'i17_raw'] - dbg_15['i17'].mean()[1]
                        elif bckgrnd_choice == 'c':
                            manual_background_std = input(
                                'Input a background value for peak {0} on wg, '
                                'or press enter to calculate automatically...'.format(
                                    peak)).strip()
                            manual_background_sa = input(
                                'Input a background value for peak {0} on sample, '
                                'or press enter to calculate automatically...'.format(
                                    peak)).strip()
                            if len(manual_background_std) > 0:
                                dr.loc[dr['is_sample']==False,'i17'] = dr.loc[
                                    dr['is_sample']==False,'i17_raw'] - float(manual_background_std)
                                dr.loc[dr['is_sample']==True,'i17'] = dr.loc[
                                    dr['is_sample']==True,'i17_raw'] - float(manual_background_sa)

                        else:
                            # otherwise, apply the same to both
                            dr.loc[dr['is_sample']==False,'i17'] = dr.loc[
                                dr['is_sample']==False,'i17_raw'] - dbg_15['i17'].mean().mean()
                            dr.loc[dr['is_sample']==True,'i17'] = dr.loc[
                                dr['is_sample']==True,'i17_raw'] - dbg_15['i17'].mean().mean()
    
                    else:
                        # otherwise, apply mean of both to each:
                        dr.loc[dr['is_sample']==False,'i17'] = dr.loc[
                            dr['is_sample']==False,'i17'] - dbg_15['i17'].mean().mean()
                        dr.loc[dr['is_sample']==True,'i17'] = dr.loc[
                            dr['is_sample']==True,'i17'] - dbg_15['i17'].mean().mean()
            # 2.4 Special treatment just for D2 peak, which has a tailing corr
            if peak in ['12CH2D2', '12CHD2']:
                if input_tail_D2_background:
                    try:
                        print('Measured scattered ion background on 12CHD2 is: '
                              '{0:.3f} cps'.format(
                                dbg['i17'].mean().mean()))
                    except(UnboundLocalError):
                        print('No background scans detected')
                    D2_tail_background_wg = input(
                        'Input total background for 12CH2D2 on WG \n '
                        'or press ENTER to continue... ').strip()
                    D2_tail_background_sample = input(
                        'Input total background for 12CH2D2 on SAMPLE \n '
                        'or press ENTER to continue... ').strip()                    
                    if len(D2_tail_background_wg) > 0:
                        if len(D2_tail_background_sample) == 0:
                            D2_tail_background_sample = D2_tail_background_wg
                        print('Applying backgrounds of {0:.3f} and {1:.3f} '
                              'cps to wg and sample 12CHD2 peaks, '
                              'respectively'.format(
                                  float(D2_tail_background_wg),
                                  float(D2_tail_background_sample)))
                        # if tail background added, apply this to voltages on both sides
                        dr.loc[dr['is_sample']==False,'i17'] = dr.loc[
                            dr['is_sample']==False,'i17'] - float(D2_tail_background_wg)
                        dr.loc[dr['is_sample']==True,'i17'] = dr.loc[
                            dr['is_sample']==True,'i17'] -  float(D2_tail_background_sample)       
            dr['R17'] = dr['i17']/dr['i15']
    dr['integration_time'] = integration_time
    # 3. Filter for outlier sub-integrations, calculate delta values
    # Calculate confidence interval expected given the number of observations. 
    # I.e., sigma window in which all observations should fall
    proportional_confidence_interval = np.sqrt(2)*erfinv(
        1.0 - 1.0/(integration_num*2.0))
    dr = filter_for_outliers(dr, sigma_filter=sigma_filter)
    if db.size > 0:
        dr = filter_for_max_ratios(dr,
                                   sigma_filter=proportional_confidence_interval*2,
                                   dbr=dbr)
    else:    
        dr = filter_for_max_ratios(dr,
                                   sigma_filter=proportional_confidence_interval*2)
    if db.size >0:
        if 'frag13_i15' in dbr['peak'].unique():
            frag_rate = calculate_k_factor(dr, dbr, dbg_15)
    drm = calculate_deltas(dr)
    return(dr, drm, file_name)

def calculate_k_factor(dr, dbr, dbg_15, plot_results=False):
    """
    Computes the k-factor (secondary methyl fragmentation rate) using the data
    from the background scans. I.e., k where i_CH2 = k*i_CH3

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
    frags = dbr.loc[(dbr['peak'] == 'frag13_i15'),[
        'measure_line','is_sample','is_outlier','i15']].copy()
    peaks = dr.loc[(dr['measure_line'].max()-dr['measure_line'] <5), [
        'measure_line','is_sample','is_outlier','i16']].copy()
    # divide up sample vs std  
    frags.loc[frags['is_sample'] == True,'i15_sa'] = frags.loc[
        (frags['is_sample'] == True) & (frags['is_outlier'] == False),'i15'] - dbg_15['i15'].mean()[1]
    frags.loc[frags['is_sample'] == False,'i15_std'] = frags.loc[
        (frags['is_sample'] == False) & (frags['is_outlier'] == False),'i15'] - dbg_15['i15'].mean()[0]
    # same for i16
    peaks.loc[peaks['is_sample'] == True, 'i16_sa'] = peaks.loc[
        (peaks['is_sample'] == True) & (peaks['is_outlier'] == False), 'i16']
    peaks.loc[peaks['is_sample'] == False, 'i16_std'] = peaks.loc[
        (peaks['is_sample'] == False)  &(peaks['is_outlier'] == False), 'i16']
    # now, append frags to end of peaks. 
    # Timing is such that this works out perfectly because both peaks and
    # frags took the same amount of time
    # this corrects for bleedout
    pf = peaks.append(frags, ignore_index = True)
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
        
    

def sort_by_cycles(dr, integration_num, cycle_num):
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
    dr['integration_number'] = (dr['measure_line']%integration_num).astype(int)
    dr['measure_line'] = (dr['measure_line']/integration_num).apply(
        np.floor).astype(int)
    dr['cycle_number'] = (dr['measure_line']%(cycle_num*2+1)).astype(int)
    dr['acq_number'] = (dr['measure_line']/(cycle_num*2+1)).astype(int)
    dr['is_sample'] = dr['cycle_number']%2
    return(dr)    

def filter_for_outliers(dr, sigma_filter=3):
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
    dg =dr.groupby('measure_line')
    drm = dr.set_index('measure_line')
    sigs_to_rd = {'i16': ('R16','d16'), 'i17': ('R17','d17')}
    for i in sigs_to_rd.keys():
        if i in dr.columns:
            filter_ratio,d = sigs_to_rd[i]
            dr['is_outlier'] = ((
                np.abs(drm[filter_ratio]-dg[filter_ratio].mean())
                )/dg[filter_ratio].std()> sigma_filter
                ).reset_index()[filter_ratio].copy()
            dr[filter_ratio+'_unfiltered'] = dr[filter_ratio].copy()
            dr.loc[dr['is_outlier'],filter_ratio] = np.nan
    return(dr)
    
def filter_for_max_ratios(dr, sigma_filter=6 , dbr=[]):
    """
    Outlier test for off-peak drifts. Flags sub-integrations where measurement
    drifts off-peak, based on observing ratios below a certain threshold
    from the third-to-highest value.

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
    dg =dr.groupby('measure_line')
    drm = dr.set_index('measure_line')
    sigs_to_rd = {'i16': ('R16','d16'), 'i17': ('R17','d17')}
    for i in sigs_to_rd.keys():
        if i in dr.columns:
            shot_noise_signal = i
            frf,d = sigs_to_rd[i]
            filter_ratio = frf+'_unfiltered'
            if len(dbr) == 0:
                shot_noise_std_devs = np.sqrt(
                    dg[shot_noise_signal].mean()*dg['integration_time'].mean()
                    )/(dg['i15'].mean())
            else:
                shot_noise_std_devs = np.sqrt(
                    dg[shot_noise_signal].mean()*dg['integration_time'].mean()+dbr.loc[
                        dbr['is_outlier'] == False, shot_noise_signal].std()**2
                    )/(dg['i15'].mean())                
        # use third-highest to buffer angainst strays
            dr['is_off_peak'] = ((dg[filter_ratio].apply(
                lambda x: x.sort_values().iloc[-3])-drm[filter_ratio]
                )/shot_noise_std_devs > sigma_filter).reset_index()[0].copy()
            dr[frf+'_on_peak'] = dr[filter_ratio].copy()
            dr.loc[dr['is_off_peak'],frf+'_on_peak'] = np.nan
    return(dr)

def filter_for_signal_stability(dbr, integration_num,
                                sigma_filter=3,
                                count_backwards=30,
                                signals_to_use=['i15'],
                                b_block=2):
    '''
    filters backgrounds based on last 30 integrations of each cycle 
    due to pressure switching issues during first ones
    '''
    # group based on last 30 integrations in each cycle
    dbg_last = dbr.loc[(dbr['integration_number']>(
        integration_num-count_backwards)) & (dbr['block'] == b_block),:
            ].groupby('measure_line')
    dbr_m = dbr.loc[(dbr['block'] == b_block),:].set_index('measure_line')
    for i in signals_to_use:
        if i in dbr.columns:
            dbr.loc[dbr['block'] == b_block,'is_outlier'] = (
                (dbr_m[i]-dbg_last[i].mean())/dbg_last[i].std()>sigma_filter
                ).reset_index().set_index(dbr.loc[
                    dbr['block'] == b_block].index)[i].copy()
    return(dbr)

def filter_for_signal_stability_per_side(dbr, sigma_filter=3,
                                         count_backwards=2,
                                         signals_to_use=['i15'],
                                         b_block=3):
    '''
    filters backgrounds based on last 30 integrations of each cycle due to
    pressure switching issues during first ones
    '''
    # group based on last integration block on each side
    dbg_last = dbr.loc[(dbr['measure_line']>(dbr.loc[
        dbr.block == b_block,'measure_line'].max()-count_backwards)) & (
            dbr['block'] == b_block) & (dbr['is_outlier'] == False),:
                ].groupby('is_sample')
    for i in signals_to_use:
        if i in dbr.columns:
            for j in [0,1]:
                dbr.loc[(dbr['block'] == b_block) & (dbr['is_sample'] == j) & (dbr['is_outlier'] == False), 'is_outlier'] = (
                        dbr.loc[(dbr['block'] == b_block) & (dbr['is_sample'] == j) & (dbr['is_outlier'] == False), i]-dbg_last[i].mean()[j])/dbg_last[i].std()[j] > sigma_filter
    return(dbr)    
    

def calculate_deltas(dr):
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
    sigs_to_rd = {'i16': ('R16','d16'), 'i17': ('R17','d17')}
    cols_needed = ['measure_line', 'cycle_number', 'acq_number',
                   'is_sample', 'integration_time', 'i15']
    for i in sigs_to_rd.keys():
        if i in dr.columns:
            r,d = sigs_to_rd[i]             
            dr[r+'_std'] = dr.loc[dr['is_sample'] == 0, r]
            dr[r+'_sample'] = dr.loc[dr['is_sample'] == 1, r]
            cols_needed.extend([i,r+'_unfiltered', r+'_on_peak', r])
    dg =dr.groupby('measure_line')
    drm = pd.DataFrame(data = dg[cols_needed].mean())
    drm['P_imbalance'] = np.nan
    i_sample = drm.loc[drm['is_sample']==1,'measure_line'].values
    drm['percent_on_peak'] = dr.loc[dr['is_off_peak'] == False].groupby(
        'measure_line')['is_off_peak'].count()/dr.groupby(
            'measure_line')['is_off_peak'].count()*100
    drm.loc[i_sample, 'P_imbalance'] = (drm.loc[i_sample, 'i15']/(
        (drm.loc[i_sample-1,'i15'].values + drm.loc[i_sample+1,'i15'].values
         )/2)-1)*100
    for i in sigs_to_rd.keys():
        if i in dr.columns:
            r,d = sigs_to_rd[i]             
            dr[r+'_std'] = dr.loc[dr['is_sample'] == 0, r]
            dr[r+'_sample'] = dr.loc[dr['is_sample'] == 1, r]
            for to_append in ['', '_on_peak', '_unfiltered']:
                this_d = d + to_append
                this_r = r + to_append
                drm[this_d] = np.nan
            # drm[d+'_on_peak'] = np.nan
            # drm[d+'_unfiltered'] = np.nan
                drm.loc[i_sample, this_d] = (drm.loc[i_sample, this_r]/(
                    (drm.loc[i_sample-1,this_r].values + drm.loc[
                        i_sample+1,this_r].values)/2)-1)*1000
            # drm.loc[i_sample, d+'_unfiltered'] = (
            #     drm.loc[i_sample, r+'_unfiltered']/((
            #         drm.loc[i_sample-1,r+'_unfiltered'].values + drm.loc[
            #             i_sample+1,r+'_unfiltered'].values)/2)-1)*1000
            # drm.loc[i_sample, d+'_on_peak'] = (
            #     drm.loc[i_sample, r+'_on_peak']/((drm.loc[
            #         i_sample-1,r+'_on_peak'].values + drm.loc[
            #             i_sample+1,r+'_on_peak'].values)/2)-1)*1000
    return(drm)

def export_data(dr, drm, file_name):
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
    cols_to_export = ['measure_line', 'block', 'meas', 'cycle_number',
                      'is_sample', 'integration_number', 'is_outlier',
                      'is_off_peak', 'i15_raw','i15']
    sigs_to_rd = {'i16': ('R16','d16'), 'i17': ('R17','d17')}
    for i in sigs_to_rd.keys():
        if i in dr.columns:
            r,d = sigs_to_rd[i]
            cols_to_export.extend([i+'_raw', i, r+'_unfiltered', r,
                                   r+'_on_peak', r+'_std', r+'_sample'])
    
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
        'peak center')
    i_pc = d.loc[(i_peak_center) & (d['Level'].isin(
        ['UserInfo', 'UserError'])),:].index
    i_pc_success = d.loc[(i_peak_center) & (d['Level'] == 'UserInfo'),:].index
    if len(i_pc_success) > 0:
        match_string = r'(-?[0-9]\.[0-9]+[E-]*[0-9]+)'
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
cycle_num, integration_time, integration_num = get_measurement_params(
    file_name=acq_name_list[0], auto_detect=True)
for i in acq_name_list:
    if not os.path.exists(os.path.join(os.path.dirname(i),'d_data_all_summary.xlsx')):
        os.chdir(os.path.dirname(i))
    dr, drm, file_name = process_Qtegra_csv_file(i,sigma_filter=3, prompt_for_params=False,
                                                 cycle_num=cycle_num, integration_time=integration_time,
                                                 integration_num=integration_num, prompt_for_backgrounds=False,
                                                 input_tail_D2_background=True,
                                                 promptForPyarrow=True)
    drs.append(dr)
    drms.append(drm)
    file_names.append(file_name)
    export_data(dr, drm, file_name)
    if input('parse log file for {0}: (y/n)? '.format(file_name)).lower() != 'n':
        log_file, peak_centers = parse_log_file(i)
        try:
            drms[-1].insert(5, 'center_mass', peak_centers['center_mass'].values)
            drms[-1].insert(6, 'delta_offset', peak_centers['delta_offset'].values)
            drms[-1].insert(7, 'mass_offset', peak_centers['mass_offset'].values)
        except(ValueError):
            print('Data and peak centers are misaligned. Using only the first {0} rows of the peak center data'.format(len(drms[-1])))
            peak_centers_short = peak_centers.iloc[:len(drms[-1]), :]
            drms[-1].insert(5, 'center_mass', peak_centers_short['center_mass'].values)
            drms[-1].insert(6, 'delta_offset', peak_centers_short['delta_offset'].values)
            drms[-1].insert(7, 'mass_offset', peak_centers_short['mass_offset'].values)            
        drms[-1]['max_delta_offset'] = np.nan
        i_sample = drms[-1].loc[drms[-1]['is_sample'] == True].index
        drms[-1].loc[i_sample, 'max_delta_offset'] = np.stack(
            [drms[-1].loc[i_sample, 'delta_offset'].values, drms[-1].loc[
                i_sample+1, 'delta_offset'].values]).max(axis = 0)
        

  
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
