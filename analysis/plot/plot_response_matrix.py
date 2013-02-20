#!/usr/bin/env python
# encoding: utf-8
"""
hinton diagram of response matrix

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import os
import numpy as np
import pylab as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import NullLocator
from master.libs import read_data_lib as rdl

subtract_sfr = False
outpath = '/Users/dedan/projects/master/results/summary/'
rm_path = 'data/response_matrix.csv'

cases, gloms, rm = rdl.load_response_matrix(rm_path, door2id=None)

# read standard firing rates
if subtract_sfr:
    sfrs = open(rm_path).readlines()[1].split(',')[1:]
    sfrs = np.array([float(s) if not s == 'NA' else 0 for s in sfrs])
    rm = np.subtract(rm, sfrs)

# only look at a slice of the matrix
rm = rm[50:120]
rm[np.isnan(rm)] = 0

fig = plt.figure(figsize=(rm.shape[1]/7, rm.shape[0]/7))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.xaxis.set_major_locator(NullLocator())
ax.yaxis.set_major_locator(NullLocator())

for (x,y),w in np.ndenumerate(rm.T):
    if w > 0: color = 'black'
    else:     color = 'g'
    size = np.max(np.abs(w)-0.1, 0)
    if not w == 0:
        rect = Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color)
        ax.add_patch(rect)
ax.autoscale_view()
ax.set_xlim([-1, rm.shape[1]])
ax.set_ylim([-1, rm.shape[0]])

fig.savefig(os.path.join(outpath, 'rm.png'), dpi=300)
plt.show()