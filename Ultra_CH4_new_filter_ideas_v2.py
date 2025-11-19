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
from scipy.stats import zscore, iqr, skewtest, gaussian_kde

plt.close('all')
homeDir = os.getcwd()

df = pd.read_feather('d_data_all_runs_2025.feather')

df['is_sample'] = df['is_sample'].astype(bool)        
 
# assume R18 is the rounded peak and then fill with peak of interest
df['i_rare'] = df['i18'].copy()
df['i_rare'] = df['i_rare'].fillna(df['i17'])
df['i_base'] = df['i16'].copy()

df['R_stable'] = df['R18_stable'].copy()
df['R_stable'] = df['R_stable'].fillna(df['R18_unfiltered'])
df['R_stable'] = df['R_stable'].fillna(df['R17_stable'])
df['R_stable'] = df['R_stable'].fillna(df['R17_unfiltered'])

# df.loc[]
# compute shot noise per meas

grpIndex = ['sample_ID', 'meas_date', 'meas_type', 'block', 'acq_number', 'cycle_number', 'is_sample']
dg = df.groupby(grpIndex)


toMedian = ['integration_time', 'i_base', 'i_rare', 'R_stable']

dm = dg[toMedian].median()

nlarge = 10
dgni = dg['R_stable'].nlargest(nlarge)
dm['R_expected'] = dgni.groupby(grpIndex).nth(nlarge-1).droplevel(-1)
dm['i_expected'] = dm['R_expected']*dm['i_base']
dm['R_shot_noise'] = np.sqrt(dm['i_expected']*dm['integration_time'])/dm['i_base']

# merge into df
df2 = df.merge(dm.reset_index(), how='left', left_on=grpIndex, right_on=grpIndex, suffixes=['', '_median'])

# groupby again
dg2 = df2.groupby(grpIndex)
# compute rolling
nRoll = 5
dgRoll = dg2['R_stable'].rolling(nRoll).mean()
# drop the unnecessary index levels
dgRoll.index = dgRoll.index.droplevel(level=grpIndex)
# add back to df
df2['R_rolling'] = dgRoll
# apply filter
sigmaFilter = 4
df2['R_lower_limit'] = df2['R_expected'] - 2*sigmaFilter*df2['R_shot_noise']/np.sqrt(nRoll)
df2['low_rolling'] = df2['R_rolling'] < df2['R_lower_limit']


dfCln = df2.loc[~df2['low_rolling'], :]
dg3 = dfCln.groupby(['sample_ID', 'meas_date', 'meas_type', 'block', 'acq_number', 'is_sample'])

dgmf = pd.DataFrame(dg3['R_stable'].mean())
dgmf = dgmf.merge(dg3['R_stable'].median(), how='left', left_index=True, right_index=True, suffixes=['_mean', '_median'])
dgmc = dgmf.reset_index()

dgmc = dgmc.set_index(['sample_ID', 'meas_date', 'meas_type', 'block', 'acq_number'])
deltas = (dgmc.loc[dgmc['is_sample'], ['R_stable_mean', 'R_stable_median']]/dgmc.loc[~dgmc['is_sample'], ['R_stable_mean', 'R_stable_median']]-1)*1000

deltag = deltas.groupby(['sample_ID', 'meas_date', 'meas_type'])
deltam = deltag.mean()
deltam = deltam.merge(deltag.std(), left_index=True, right_index=True, suffixes=['', '_SD'])
deltam['n'] = deltag['R_stable_mean'].count()
deltam.to_excel('rolling_mean_byacq_summary.xlsx', merge_cells=False)