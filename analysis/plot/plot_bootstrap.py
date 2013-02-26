#!/usr/bin/env python
# encoding: utf-8
"""
illustration of the bootstrap precedure

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import os
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from master.libs import utils

outpath = '/Users/dedan/projects/master/results/visualization/'
rm_path = 'data/response_matrix.csv'
glom = 'Or35a'

cases, gloms, rm = rdl.load_response_matrix(rm_path, door2id=None)

data = rm[:, gloms.index(glom)]
data = data[~np.isnan(data)]
fig = plt.figure()
ax = fig.add_subplot(111)
counts, bins, _ = ax.hist(data, bins=np.arange(0, 1.01, 0.1), color='0.5')

center = np.min(data) + (np.max(data) - np.min(data)) / 2
ax.plot([center, center], [0, np.max(counts) / 2], color='r')
utils.simple_axis(ax)

print('points in total: {}'.format(len(data)))
print('points above {:.2f}: {}'.format(center, np.sum(data > center)))
print('points below {:.2f}: {}'.format(center, np.sum(data <= center)))
fig.savefig(os.path.join(outpath, 'bootstrap.png'), dpi=300)


plt.show()
