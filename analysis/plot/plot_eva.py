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
reload(utils)

molid = '1'
outpath = '/Users/dedan/projects/master/results/visualization/'
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
feature_file = os.path.join(data_path, 'spectral_features', 'large_base', 'parsed.pckl')
spectra = pickle.load(open(feature_file))

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(511)
ax.hist(spectra[molid]['freq'], bins=4000)
ax.set_xlim([0, 4000])
ax.set_xticks([])
ax.set_yticks([])
utils.simple_axis(ax)

for i, k_width in enumerate([2, 20, 50, 100]):
    print('loading..')
    features = fl.get_spectral_features(spectra,
                                        use_intensity=False,
                                        kernel_widths=k_width,
                                        bin_width=10)
    print('plotting..')
    ax = fig.add_subplot(5, 1, i + 2)
    ax.plot(features[molid])
    if not i == 3:
        ax.set_xticks([])
    else:
        ax.set_xticks([ax.get_xticks()[0], ax.get_xticks()[-1]])
        ax.set_xticklabels([0, 4000])
    utils.simple_axis(ax)
    ax.set_yticks([])
fig.savefig(os.path.join(outpath, 'eva.png'))