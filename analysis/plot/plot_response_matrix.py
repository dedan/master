#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import numpy as np
import pylab as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import NullLocator
from master.libs import read_data_lib as rdl

cases, gloms, rm = rdl.load_response_matrix('data/response_matrix.csv', door2id=None)

# # read standard firing rates
# sfrs = open('data/response_matrix.csv').readlines()[1].split(',')[1:]
# sfrs = np.array([float(s) if not s == 'NA' else 0 for s in sfrs])
# rm = np.subtract(rm, sfrs)

rm[np.isnan(rm)] = 0

fig = plt.figure(figsize=(9, 30))
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

fig.savefig('bla.png', dpi=300)

# for i_cas, cas in enumerate(cases):
#     for i_glom, glom in enumerate(gloms):
#         ax.plot(i_cas, i_glom, 'o', markersize=rm[i_cas, i_glom])
#     print i_cas
plt.show()