#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import pickle
import numpy as np
import pylab as plt

basepath = '/Users/dedan/projects/master/results/spectra'

res_gamess = pickle.load(open(os.path.join(basepath, 'plots_gamess', 'res.pckl')))
res_gaussian = pickle.load(open(os.path.join(basepath, 'plots_gaussian', 'res.pckl')))
res_gamess_sel = pickle.load(open(os.path.join(basepath, 'plots_gamess_sel', 'res.pckl')))
res_gaussian_sel = pickle.load(open(os.path.join(basepath, 'plots_gaussian_sel', 'res.pckl')))

plt.close('all')
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(141)
data = {}
for kernel_width in res_gamess[res_gamess.keys()[0]]['oob']:
    data[kernel_width] = []
    for glom in res_gamess:
        data[kernel_width].append(res_gamess[glom]['oob'][kernel_width])
ax.boxplot(data.values())
ax.set_xticklabels(data.keys())
ax.set_ylim([0, 0.5])
ax.set_ylabel('gamess')

ax = fig.add_subplot(142)
data = {}
for kernel_width in res_gaussian[res_gaussian.keys()[0]]['oob']:
    data[kernel_width] = []
    for glom in res_gaussian:
        data[kernel_width].append(res_gaussian[glom]['oob'][kernel_width])
ax.boxplot(data.values())
ax.set_xticklabels(data.keys())
ax.set_ylim([0, 0.5])
ax.set_ylabel('gaussian')


ax = fig.add_subplot(143)
data = {}
for kernel_width in res_gamess_sel[res_gamess_sel.keys()[0]]['oob']:
    data[kernel_width] = []
    for glom in res_gamess_sel:
        data[kernel_width].append(res_gamess_sel[glom]['oob_sel'][kernel_width])
ax.boxplot(data.values())
ax.set_xticklabels(data.keys())
ax.set_ylim([0, 0.5])
ax.set_ylabel('gamess_sel')

ax = fig.add_subplot(144)
data = {}
for kernel_width in res_gaussian_sel[res_gaussian_sel.keys()[0]]['oob']:
    data[kernel_width] = []
    for glom in res_gaussian_sel:
        data[kernel_width].append(res_gaussian_sel[glom]['oob_sel'][kernel_width])
ax.boxplot(data.values())
ax.set_xticklabels(data.keys())
ax.set_ylim([0, 0.5])
ax.set_ylabel('gaussian_sel')
plt.show()
