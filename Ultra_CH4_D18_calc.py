#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:53:59 2018

@author: Max
"""

import numpy as np
import pandas as pd
import random
import os


def FMCI_r_to_c(rD,r13):
    ''' func for caculating methane stochastic
    isotopologue concentrations from ratios '''

    c12 = 1/(1+r13)
    c13 = r13/(1+r13)
    cH = 1/(1+rD)
    cD = rD/(1+rD)

    c12CH4 = c12*cH**4
    c12CH3D = 4*c12*cH**3*cD
    c13CH4 = c13*cH**4
    c13CH3D = 4*c13*cH**3*cD
    c12CH2D2 = 6*c12*cH**2*cD**2

    return(c12CH4, c12CH3D, c13CH4, c13CH3D, c12CH2D2)

def CH4_bulk_comp(ref_gas_ID, dD_wg, d13C_wg, d13CD_wg=np.nan, dD2_wg=np.nan, frag_rate=0.79):
    ''' Solves the linear equations to accurately calculate
    the bulk composition of d13C and d18O. '''
    r13C_vpdb = 0.0112372 # craig (1957)

    rD_vsmow = 0.00015576
    # # Define wg values
    d13C_ref_dict = {'GTS-1': -34.14}
    d13C_ref = d13C_ref_dict[ref_gas_ID]
    dD_ref_dict = {'GTS-1': -169.5}
    dD_ref = dD_ref_dict[ref_gas_ID]
    # # Calculate wg ratios and errors
    r13C_ref = (d13C_ref/1000 + 1)*r13C_vpdb
    rD_ref = (dD_ref/1000 + 1)*rD_vsmow
    # calculate concentrations of ref gas
    m12CH3D_sa = (dD_wg/1000 + 1)*rD_ref*4
    rD_sa = m12CH3D_sa/4

    
    
    m13CH4_sa = (d13C_wg/1000 + 1)*r13C_ref/(1+frag_rate*(r13C_ref + 3*rD_ref))
    r13C_sa = m13CH4_sa*(1 + 3*frag_rate*rD_sa)/(1 - m13CH4_sa*frag_rate)

    
    #wg clumped compositions from Eldridge et al. 2019 ACS Earth Space Chem.
    D13CD_ref = 0
    r13CD_ref_stoch = 4*r13C_ref*rD_ref
    r13CD_ref = (D13CD_ref/1000 + 1)*r13CD_ref_stoch

    DD2_ref = 0
    rD2_ref_stoch = 6*rD_ref*rD_ref
    rD2_ref = (DD2_ref/1000 + 1)*rD2_ref_stoch

    # # now, calculate for sample side

    r13CD_sa = ((d13CD_wg/1000 +1)*r13CD_ref/(1+frag_rate*(r13C_ref + 3*rD_ref)))*(1 + frag_rate*(r13C_sa + 3*rD_sa))
    rD2_sa = ((dD2_wg/1000 +1)*rD2_ref/(1+frag_rate*(r13C_ref + 3*rD_ref)))*(1 + frag_rate*(r13C_sa + 3*rD_sa))
    
    D13CD_wg = (r13CD_sa/(4*r13C_sa*rD_sa)-1)*1000
    DD2_wg = (rD2_sa/(6*rD_sa*rD_sa)-1)*1000

    dD_vsmow = (rD_sa/rD_vsmow-1)*1000
    d13C_vpdb = (r13C_sa/r13C_vpdb-1)*1000

    return(pd.DataFrame({'dD_vsmow':dD_vsmow, 'd13C_vpdb': d13C_vpdb, 'D13CD_wg': D13CD_wg, 'DD2_wg': DD2_wg}))

def calculate_randoms(row, mean, std_dev):
    return(row['seed'].gauss(mean, std_dev))


def CH4_bulk_comp_by_row(row):
    [dD_vsmow, d13C_vpdb, D13CD_wg] = CH4_bulk_comp(row['ref gas ID'], row.dD_wg, row.d13C_wg, row.d13CD_wg)
    row['dD_vsmow'] = dD_vsmow
    row['d13C_vpdb'] = d13C_vpdb
    row['D13CD_wg'] = D13CD_wg
    return(row)

if __name__ == '__main__':
    # change to correct location
    summaryPath = '/Users/Thermo/Documents/Ultra/Data/CH4'
    os.chdir(summaryPath)
    d = pd.read_excel('methane_summary_sheet.xlsx',
                      header=[0,1], index_col=0)

    # get indices of samples that are unprocessed
    i_to_process = d.loc[((d[('dD'), 'mean vs. wg'].notnull()) & (
        d[('Calculated data'), 'dD_vsmow'].isnull()))|(
            (d[('d13C'), 'mean vs. wg'].notnull()) & (
                d[('Calculated data'), 'd13C_vpdb'].isnull()))|(
                    (d[('d13CD'), 'mean vs. wg'].notnull()) & (
                        d[('Calculated data'), 'D13CD_wg'].isnull()))|(
                            (d[('dD2'), 'mean vs. wg'].notnull()) & (
                                d[('Calculated data'), 'DD2_wg'].isnull()))].index

    if len(i_to_process) == 0:
        print('No columns to process. Goodbye! ')
    else:
        print('Processing rows: {0}'.format([j for j in i_to_process]))
        N = int(1e5)
        dmc = pd.DataFrame(data = {'i': np.arange(N), 'seed': [random.SystemRandom() for j in range(N)]})
        cols_to_make = ['dD_vsmow', 'd13C_vpdb', 'D13CD_wg', 'DD2_wg']
        for i in i_to_process:
            print('MC error propagation for row {0}... '.format(i))
            ref_gas_ID = d.loc[i,'Info']['ref gas ID']
            dmc['dD_wg'] = dmc.apply(calculate_randoms, axis=1,
                                     args=(d.loc[i,'dD']['mean vs. wg'],
                                           d.loc[i,'dD']['std error']))
            dmc['d13C_wg'] = dmc.apply(calculate_randoms, axis=1,
                                       args=(d.loc[i,'d13C']['mean vs. wg'],
                                             d.loc[i,'d13C']['std error']))
            dmc['d13CD_wg'] = dmc.apply(calculate_randoms, axis=1,
                                        args=(d.loc[i,'d13CD']['mean vs. wg'],
                                              d.loc[i,'d13CD']['std error']))
            dmc['dD2_wg'] = dmc.apply(calculate_randoms, axis=1,
                                      args=(d.loc[i,'dD2']['mean vs. wg'],
                                            d.loc[i,'dD2']['std error']))
            
            if d.loc[i,'dD']['frag rate'] > -1:
                frag_rate = d.loc[i,'dD']['frag rate']
            else:
                frag_rate = 0.79
            # catch case where dD not measured
            # needed so we can still do d13C meas with frag
            # if dmc['dD_wg'].isnull().prod() == True:
            #     dmc['dD_wg'] = 0
            results = CH4_bulk_comp(ref_gas_ID, dmc['dD_wg'],
                                     dmc['d13C_wg'],
                                     d13CD_wg=dmc['d13CD_wg'],
                                     dD2_wg=dmc['dD2_wg'],
                                     frag_rate=frag_rate)
            d.loc[i,('Calculated data', cols_to_make)] = results[cols_to_make].mean().round(3).values
            d.loc[i,('Calculated data', ['dD_std_error',
                                         'd13C_std_error',
                                         'D13CD_std_error',
                                         'DD2_std_error'])] = results[cols_to_make].std().round(3).values
            d.loc[i,('Calculated data', ['D13CD_dD_var',
                                         'D13CD_d13C_var'])] = results.cov()['D13CD_wg'].values[0:2]

        print('MC calcs complete. Writing new spreadsheet... ')
        while True:
            try:
                d.loc[:,(['Info', 'Calculated data'])].to_excel(
                    'methane_summary_sheet_calc.xlsx')
                break
            except(PermissionError):
                is_it_closed = input('Could not write to spreadsheet '
                                     '\n fluoromethane_summary_sheet_calc.xlsx '
                                     '\n because it is open. \n '
                                     'Close it and press ENTER to continue... ')
