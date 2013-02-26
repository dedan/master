#!/usr/bin/env python
# encoding: utf-8
"""
illustrate how the EVA descriptor is computed

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import pickle
import numpy as np
import pylab as plt
from master.libs import features_lib as fl
from master.libs import utils
from master.libs import run_lib
from scipy.spatial.distance import pdist,squareform
reload(utils)

k_width = 20
molid = '1'
outpath = '/Users/dedan/projects/master/results/visualization/'
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
feature_file = os.path.join(data_path, 'spectral_features', 'large_base', 'parsed.pckl')
spectra = pickle.load(open(feature_file))

fig = plt.figure()
ax = fig.add_subplot(211)
ax.hist(spectra[molid]['freq'], bins=4000, edgecolor='0.4', facecolor='0.4')
ax.set_xlim([0, 4000])
ax.set_xticks([0, 4000])
ax.set_xticklabels([0, 4000])
ax.set_yticks(ax.get_ylim())
utils.simple_axis(ax)

print('loading..')
features = fl.get_spectral_features(spectra,
                                    use_intensity=False,
                                    kernel_widths=k_width,
                                    bin_width=10)
print('plotting..')
ax = fig.add_subplot(212)
ax.plot(features[molid], color='0.4', label='sigma: 20')
ax.set_xticks([ax.get_xticks()[0], ax.get_xticks()[-1]])
ax.set_xticklabels([0, 4000])
utils.simple_axis(ax)
ax.set_yticks(ax.get_ylim())
ax.set_xlabel('wavenumber (1/cm)')
ax.legend(loc='upper left', frameon=False, numpoints=1)
fig.savefig(os.path.join(outpath, 'eva.png'), dpi=300)
