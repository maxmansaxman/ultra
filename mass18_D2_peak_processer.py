#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:08:09 2019

@author: Max
"""

import numpy as np
import pandas as pd
import random
import os
import matplotlib as mpl
# mpl.use('PDF')
import matplotlib.pyplot as plt
import scipy.stats
# mpl.rcParams.update({'figure.autolayout': True})
from scipy.special import erf, erfinv
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline, LSQUnivariateSpline
# plt.style.use('ggplot')
mpl.rcParams.update({'mathtext.default': 'regular'})
#plt.ion()
plt.close('all')

def peak_shape_model(mass, peak_center, amplitude, sigma, cup_width=0.00048):
    ''' model of a narrow peak suing the difference between two erfs'''
    intensity = amplitude/2*(erf((mass - peak_center + cup_width)/sigma)
                             - erf((mass - peak_center - cup_width)/sigma))
    return(intensity)


# def three_peak_model(mass, center_13C, amplitude_13C, amplitude_D,
#                      amplitude_adduct, sigma, cup_width):
#     intensity = peak_shape_model(mass, center_13C, amplitude_13C, sigma,
#                                  cup_width=cup_width) \
#                 + peak_shape_model(mass, center_13C + 0.00292*16.43/17.03,
#                                    amplitude_D, sigma, cup_width=cup_width) \
#                 + peak_shape_model(mass, center_13C + 0.00447*16.53/17.03,
#                                    amplitude_adduct, sigma,
#                                    cup_width=cup_width)
#     return(intensity)
    
def four_peak_model(mass, center_13CD, amplitude_13CD, amplitude_13C_adduct,
                     amplitude_D2, amplitude_D_adduct,
                     sigma, cup_width):
    intensity = peak_shape_model(mass, center_13CD, amplitude_13CD, sigma,
                                 cup_width=cup_width) \
                + peak_shape_model(mass, center_13CD + 0.00155*17.09/18.041,
                                   amplitude_13C_adduct, sigma, cup_width=cup_width) \
                + peak_shape_model(mass, center_13CD + 0.00292*17.09/18.041,
                                   amplitude_D2, sigma,
                                   cup_width=cup_width) \
                + peak_shape_model(mass, center_13CD + 0.00447*17.09/18.041,
                                   amplitude_D_adduct, sigma,
                                   cup_width=cup_width)
    return(intensity)


# def three_peak_minimizer(p, *extra_args):
#     masses, signal = extra_args
#     model = three_peak_model(masses, p[0], p[1]*1e7, p[2]*1e5, p[3]*1e5,
#                              p[4]/1e4, p[5]/1e4)
#     misfit = np.abs(model - signal)
#     return(np.sum(misfit))
    
def one_peak_minimizer(p, *extra_args):
    masses, signal = extra_args
    model = peak_shape_model(masses, p[0], p[1],
                             p[2], p[3])
    misfit = np.abs(model - signal)
    return(np.sum(misfit))


def four_peak_minimizer(p, *extra_args):
    masses, signal, sigma, cup_width = extra_args
    model = four_peak_model(masses, p[0], p[1], p[2], p[3], p[4],
                            sigma, cup_width)
    misfit = np.abs(model - signal)
    return(np.sum(misfit))
    
# def double_peak_minimizer(p, *extra_args):
#     masses, signal = extra_args
#     model = peak_shape_model(masses, p[0], p[1],
#                              p[2], p[3]) + peak_shape_model(masses, p[0], p[5], p[4], p[3])
#     misfit = np.abs(model - signal)
#     return(np.sum(misfit))  
    
def find_and_report_backgrounds(OH_peak):
    # find mass of max OH_peak
    max_mass = OH_peak.loc[OH_peak['H4 CDD'].idxmax(),'Mass H4 CDD']
    mass_lower_lim = 0.002
    true_bg = OH_peak.loc[OH_peak['Mass H4 CDD'] > (max_mass + mass_lower_lim),'H4 CDD'].mean()
    return(true_bg)
 
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
    return(resEval)

        
        


ask_for_files = True
# import files
OH_file = 'C:/Users/Thermo/Documents/Ultra/Data/CH4/2025/05/05-09_plusD/dD2/OH_scan_wg.csv'

CH4_file_wg = 'C:/Users/Thermo/Documents/Ultra/Data/CH4/2025/05/05-09_plusD/dD2/D2_scan_wg.csv'
CH4_file_sample = 'C:/Users/Thermo/Documents/Ultra/Data/CH4/2025/05/05-09_plusD/dD2/D2_scan_sample.csv'

scan_defaults = [OH_file, CH4_file_wg, CH4_file_sample]
scans_to_request = ['OH peak', 'wg CH3 peaks', 'sample CH3 peaks']
scan_files = []
scans = []
if ask_for_files:
    for i in range(len(scans_to_request)):
        while True:
            s_path = input('Drag a scan file for {0}, or press ENTER to use default... '.format(scans_to_request[i])).replace('\\ ', ' ').strip("'").strip('"').strip()
            if len(s_path) > 0:
                s_path = os.path.abspath(s_path)
                if os.path.exists(s_path) and s_path.endswith('.csv'):
                    file_name = os.path.basename(s_path).split('.csv')[0]
                    scan_files.append(s_path)
                    break
                else:
                    print('Not a .csv file ')
            else:
                scan_files.append(scan_defaults[i])
                file_name = 'default'
                break

home_folder = os.path.split(s_path)[0]
os.chdir(home_folder)

# then, loop through and clean up
for sf in scan_files:
    try:
        s = pd.read_csv(sf, header=1, sep=';', index_col=0)
        s = s.drop(columns=s.columns[s.columns.str.contains('(', regex=False)])
        s = s.dropna(axis=1, how='all').dropna(axis=0, how='all')
    except(AttributeError):
        s = pd.read_csv(sf, header=0, sep=',', index_col=0)
        s = s.drop(columns=s.columns[s.columns.str.contains('(', regex=False)])
        s = s.dropna(axis=1, how='all').dropna(axis=0, how='all')
    # if a .csv file has already been opened, needs to be treated different
    if len(s) < 10:
        s = pd.read_csv(sf, header=0, sep=',', index_col=0)
        s = s.drop(columns=s.columns[s.columns.str.contains('(', regex=False)])
        s = s.dropna(axis=1, how='all').dropna(axis=0, how='all')        

    
    # check if scan order is reversed
    if s['Mass H4 CDD'].diff().mode()[0] < 0:
        # if so, reverse it
        s = s.iloc[::-1]
        s = s.set_index(s.index.sort_values()) 
    scans.append(s)

[OH_peak, CH3_peak_wg, CH3_peak_sample] = scans
# now, get actual bg from OH peak
sbg_wg = find_and_report_backgrounds(OH_peak)
while True:
    sbg_wg_choice = input('Input a scattered ion background for wg or press ENTER to calculate from OH peak scan... ').strip()
    if len(sbg_wg_choice) > 0:
        try:
            sbg_wg = float(sbg_wg_choice)
        except(ValueError):
            print('Invalid input, try again ')
        break
    break
while True:
    sbg_sa_choice = input('Input a scattered ion background for sample or press ENTER to use wg value... ').strip()
    if len(sbg_sa_choice) > 0:
        try:
            sbg_sa = float(sbg_sa_choice)
        except(ValueError):
            print('Invalid input, try again ')
        break
    else:
        sbg_sa = sbg_wg
        break
# finally, apply bgs
OH_peak['H4_bg_corr'] = OH_peak['H4 CDD'] - sbg_wg
CH3_peak_wg['H4_bg_corr'] = CH3_peak_wg['H4 CDD'] - sbg_wg
CH3_peak_sample['H4_bg_corr'] = CH3_peak_sample['H4 CDD'] - sbg_sa  
        


# now, fit and plot the OH peak
cup_width = 0.000455
extra_args = (OH_peak['Mass H4 CDD'].values, OH_peak['H4_bg_corr'].values)
params_guess = np.array([OH_peak.loc[OH_peak['H4 CDD'].idxmax(), 'Mass H4 CDD'],
                         OH_peak['H4 CDD'].max(), 0.0004,
                         cup_width])
res = minimize(one_peak_minimizer, params_guess, args=extra_args)
OH_peak['H4_model'] = peak_shape_model(
        OH_peak['Mass H4 CDD'], res.x[0], res.x[1],
        res.x[2], res.x[3])


sigma = res.x[2]
cup_width = res.x[3]


OH_interp = UnivariateSpline(OH_peak['Mass H4 CDD'], OH_peak['H4_bg_corr'], s=300)


hrMasses = np.linspace(OH_peak['Mass H4 CDD'].min(), OH_peak['Mass H4 CDD'].max(), num=1000)
hrOH =  peak_shape_model(hrMasses, res.x[0], res.x[1], res.x[2], res.x[3])
hrdf = pd.DataFrame(data={'Mass_center': hrMasses,
                                  'H4_model': hrOH})
hrdf['Mass'] = hrdf['Mass_center']*18.041/17.09
hrdf['H4_interp'] = OH_interp(hrdf['Mass_center'])

resEval = compute_resolution(hrdf)


# next, fit and plot wg

max_signal = CH3_peak_wg['H4 CDD'].max()
four_peak_extra_args = (CH3_peak_wg['Mass H4 CDD'].values,
                        CH3_peak_wg['H4_bg_corr'].values, sigma,
                        cup_width)
four_peak_guess = np.array([CH3_peak_wg.loc[CH3_peak_wg['H4 CDD'].idxmax(),
                                            'Mass H4 CDD'],
    max_signal, max_signal/2, 35, max_signal/20])
res_four_wg = minimize(four_peak_minimizer, four_peak_guess, args=four_peak_extra_args)
CH3_peak_wg['H4_model'] = four_peak_model(
        CH3_peak_wg['Mass H4 CDD'], res_four_wg.x[0], res_four_wg.x[1],
        res_four_wg.x[2], res_four_wg.x[3], res_four_wg.x[4], sigma, cup_width)

mass_scale_points = np.array([0.00137, -0.00155])
cps_interp_OH = OH_interp(res.x[0] + mass_scale_points*17.09/18.041)
cps_interp_OH =np.array([OH_peak.loc[(OH_peak['Mass H4 CDD'] < res.x[0] + mass_scale_points[0]*17.09/18.041+0.0001) & (OH_peak['Mass H4 CDD'] > res.x[0] + mass_scale_points[0]*17.09/18.041-0.0001), 'H4_bg_corr'].mean(),
                                     OH_peak.loc[(OH_peak['Mass H4 CDD'] < res.x[0] + mass_scale_points[1]*17.09/18.041+0.0001) & (OH_peak['Mass H4 CDD'] > res.x[0] + mass_scale_points[1]*17.09/18.041-0.0001), 'H4_bg_corr'].mean()])

cps_interp_wg = cps_interp_OH/res.x[1]*res_four_wg.x[[2,4]]


# finally, fit and plot sample
max_signal = CH3_peak_sample['H4 CDD'].max()
four_peak_extra_args = (CH3_peak_sample['Mass H4 CDD'].values,
                        CH3_peak_sample['H4_bg_corr'].values, sigma,
                        cup_width)
four_peak_guess = np.array([CH3_peak_sample.loc[CH3_peak_sample['H4 CDD'].idxmax(),
                                            'Mass H4 CDD'],
    max_signal, max_signal/2, 35, max_signal/20])
res_four_sa = minimize(four_peak_minimizer, four_peak_guess, args=four_peak_extra_args)
CH3_peak_sample['H4_model'] = four_peak_model(
        CH3_peak_sample['Mass H4 CDD'], res_four_sa.x[0], res_four_sa.x[1],
        res_four_sa.x[2], res_four_sa.x[3], res_four_sa.x[4], sigma, cup_width)
try:
    i_16_sa = CH3_peak_sample.loc[(np.abs(CH3_peak_sample['Mass L3']-res_four_sa.x[0]) < 0.0006),'L3'].mean()
    i_16_wg = CH3_peak_wg.loc[(np.abs(CH3_peak_wg['Mass L3']-res_four_sa.x[0]) < 0.0006),'L3'].mean()

    cps_interp_sa = cps_interp_OH/res.x[1]*res_four_sa.x[[2,4]]*i_16_wg/i_16_sa
except(KeyError):
    cps_interp_sa = cps_interp_OH/res.x[1]*res_four_sa.x[[2,4]]

tailing_factors = {'13CH5':cps_interp_OH[0]/res.x[1], '12CH4D': cps_interp_OH[1]/res.x[1]}

bgs = pd.DataFrame({'scattered bg': [sbg_wg, sbg_sa, np.nan],
                    '13CH5 bg': [cps_interp_wg[0], cps_interp_sa[0], tailing_factors['13CH5']],
                    '12CH4D bg': [cps_interp_wg[1], cps_interp_sa[1], tailing_factors['12CH4D']],
                    'total bg': [cps_interp_wg.sum() + sbg_wg,
                                 cps_interp_sa.sum()+ sbg_sa, np.nan]},
                                 index=['wg', 'sample', 'relative tailing'])
print('-----Tailing factor-----\n 13CH5: {0:.2e} \n 12CH4D {1:.2e}'.format(
        tailing_factors['13CH5'], tailing_factors['12CH4D']))
print('-----WG-----\n 13CH5 bg: {0:.3f} \n 12CH4D bg: {1:.3f} \n scattered bg: {2:.2f} \n Total bg: {3:.3f}'.format(
        bgs.loc['wg', '13CH5 bg'], bgs.loc['wg', '12CH4D bg'], bgs.loc['wg', 'scattered bg'], bgs.loc['wg', 'total bg']))    
    
print('-----Sample-----\n 13CH5 bg: {0:.3f} \n 12CH4D bg: {1:.3f} \n scattered bg: {2:.2f} \n Total bg: {3:.3f}'.format(
        bgs.loc['sample', '13CH5 bg'], bgs.loc['sample', '12CH4D bg'], bgs.loc['sample', 'scattered bg'], bgs.loc['sample', 'total bg']))    
    
bgs.to_excel('D2_backgrounds_{0}.xlsx'.format(file_name))



fig, [ax0, ax1] = plt.subplots(1,2, figsize=(12,6))
ax0.plot(OH_peak['Mass H4 CDD'], OH_peak['H4_bg_corr'], '.', label='data')
ax0.plot(OH_peak['Mass H4 CDD'], OH_peak['H4_model'], '--', label='erf fit')
#ax0.plot(OH_peak['Mass H4 CDD'], OH_peak['H4_model_dbl'], '--', label='erf fit_dbl')

ax0.plot(OH_peak['Mass H4 CDD'], OH_peak['Mass H4 CDD'].apply(OH_interp), '-', label='interpolation')
ax0.plot(OH_peak['Mass H4 CDD'], cps_interp_OH[0]*np.ones(OH_peak['Mass H4 CDD'].shape), ':', label='contaminant cps', color='C3')
ax0.plot((res.x[0] + mass_scale_points[0]*17.09/18.041)*np.ones(50), np.linspace(0, cps_interp_OH[0], num=50), ':', label='__nolegend__', color='C3')
ax0.legend(loc='upper left')
ax0.set_yscale('log')
ax0.set_ylim((0.01, ax0.get_ylim()[1]))

ax1.plot(CH3_peak_wg['Mass H4 CDD'], CH3_peak_wg['H4_bg_corr'], '.', label='wg data')
ax1.plot(CH3_peak_wg['Mass H4 CDD'], CH3_peak_wg['H4_model'], '-', label='wg model')

ax1.plot(CH3_peak_sample['Mass H4 CDD'], CH3_peak_sample['H4_bg_corr'], '.', label='sample data')
ax1.plot(CH3_peak_sample['Mass H4 CDD'], CH3_peak_sample['H4_model'], '-', label='sample model')
ax1.text(CH3_peak_sample['Mass H4 CDD'].max()*0.75, CH3_peak_wg['H4_bg_corr'].max()*0.75, 'Total D2 bg wg: {0:.3f} cps \n Total D2 bg sample: {1:.3f}'.format((cps_interp_wg.sum()+ sbg_wg),(cps_interp_sa.sum()+ sbg_sa)))
ax1.legend(loc='upper left')

while True:
    try:
        fig.savefig('D2_tailing_factors_{0}.pdf'.format(file_name))
        break
    except(PermissionError):
        wait_for_close = input('Plot file is open. Close it and it <Enter>')


fig, [ax0, ax1] = plt.subplots(1,2, figsize=(12,6))
ax0.plot(OH_peak['Mass H4 CDD'], OH_peak['H4_bg_corr'], '.', label='data')
ax0.plot(OH_peak['Mass H4 CDD'], OH_peak['H4_model'], '--', label='erf fit')
#ax0.plot(OH_peak['Mass H4 CDD'], OH_peak['H4_model_dbl'], '--', label='erf fit_dbl')

ax0.plot(OH_peak['Mass H4 CDD'], OH_peak['Mass H4 CDD'].apply(OH_interp), '-', label='interpolation')
ax0.plot(OH_peak['Mass H4 CDD'], cps_interp_OH[0]*np.ones(OH_peak['Mass H4 CDD'].shape), ':', label='contaminant cps', color='C3')
ax0.plot((res.x[0] + mass_scale_points[0]*17.09/18.041)*np.ones(50), np.linspace(0, cps_interp_OH[0], num=50), ':', label='__nolegend__', color='C3')
ax0.legend(loc='upper left')
#ax0.set_yscale('log')
ax0.set_ylim((-1.5,2.5) )

ax1.plot(CH3_peak_wg['Mass H4 CDD'], CH3_peak_wg['H4_bg_corr'], '.', label='wg data')
ax1.plot(CH3_peak_wg['Mass H4 CDD'], CH3_peak_wg['H4_model'], '-', label='wg model')

ax1.plot(CH3_peak_sample['Mass H4 CDD'], CH3_peak_sample['H4_bg_corr'], '.', label='sample data')
ax1.plot(CH3_peak_sample['Mass H4 CDD'], CH3_peak_sample['H4_model'], '-', label='sample model')
ax1.text(CH3_peak_sample['Mass H4 CDD'].max()*0.75, CH3_peak_wg['H4_bg_corr'].max()*0.75, 'Total D2 bg wg: {0:.3f} cps \n Total D2 bg sample: {1:.3f}'.format((cps_interp_wg.sum()+ sbg_wg),(cps_interp_sa.sum()+ sbg_sa)))
ax1.legend(loc='upper left')


while True:
    try:
        fig.savefig('D2_tailing_factors_{0}_lin.pdf'.format(file_name))
        break
    except(PermissionError):
        wait_for_close = input('Plot file is open. Close it and it <Enter>')
#plt.show()
#plt.show()
while True:
    q = input('Press ENTER to quit... ')
    break
