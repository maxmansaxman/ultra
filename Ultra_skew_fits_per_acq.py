# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 11:43:01 2025

@author: Max
"""

import os
import matplotlib as mpl
import re
mpl.rcParams.update({'mathtext.default': 'regular'})
mpl.rcParams.update({'lines.markeredgecolor': 'black'})
mpl.rcParams.update({'lines.markersize': 10})
import matplotlib.pyplot as plt
# import OB1_pd_py3_no_plp as OB1_pd
import numpy as np
import pandas as pd
# from scipy import integrate
import matplotlib.transforms as mtransforms

from scipy.stats import skewnorm, bootstrap


plt.ion()
plt.close('all')

samplePath = 'C:/Users/Max/Downloads/10/10/10-1_plus13CD/dD_MRAI/d_data_all.xlsx'

d = pd.read_excel(samplePath, index_col=0)

fits = []
isSample = []

for line in d['measure_line'].unique():
    theseData = d.loc[(d['measure_line']==line) & (d['signal_is_stable']), :]
    thisFit = skewnorm.fit(theseData['R17_stable'].values)
    fits.append(thisFit)
    isSample.append(bool(theseData['is_sample'].unique()[0]))

aFits = np.stack(fits)
f = pd.DataFrame(data=aFits, columns=['skew', 'mean', 'var'])
f['isSample'] = isSample
f.to_excel('skewFits.xlsx')
