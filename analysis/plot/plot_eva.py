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

glomeruli = ['Or43b', 'Or22a', 'Or19a', 'Or35a']
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


for glom in glomeruli:
    config = {
        "glomerulus": glom,      # good performance for vib100
        "data_path": data_path
    }

    features = fl.get_spectral_features(spectra,
                                        use_intensity=False,
                                        kernel_widths=100,
                                        bin_width=150)
    features = fl.normalize_features(features)
    data, targets, molids = run_lib.load_data_targets(config, features)
    ligands = [m for i, m in enumerate(molids) if targets[i] >= 0.5]
    inactive = [m for i, m in enumerate(molids) if targets[i] < 0.2]

    ligands_matrix = np.array([features[m] for m in ligands])
    inactive_matrix = np.array([features[m] for m in inactive])
    num_lig = ligands_matrix.shape[0]
    most_simil = np.argmin(squareform(pdist(np.vstack((ligands_matrix, inactive_matrix))))[num_lig:,:num_lig],0)
    inactive_matrix = inactive_matrix[most_simil, :]

    N = ligands_matrix.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.mean(ligands_matrix, axis=0), 'r')
    ax.fill_between(range(N),
                     np.mean(ligands_matrix, axis=0),
                     np.max(ligands_matrix, axis=0),
                     facecolor='r', alpha=0.3)
    ax.fill_between(range(N),
                     np.mean(ligands_matrix, axis=0),
                     np.min(ligands_matrix, axis=0),
                     facecolor='r', alpha=0.3)

    plt.plot(np.mean(inactive_matrix, axis=0), 'b')
    ax.fill_between(range(N),
                     np.mean(inactive_matrix, axis=0),
                     np.max(inactive_matrix, axis=0),
                     facecolor='b', alpha=0.3)
    ax.fill_between(range(N),
                     np.mean(inactive_matrix, axis=0),
                     np.min(inactive_matrix, axis=0),
                     facecolor='b', alpha=0.3)
    fig.savefig(os.path.join(outpath, glom + '_eva_investigation.png'))
plt.show()
