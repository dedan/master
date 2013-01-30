#!/usr/bin/env python
# encoding: utf-8
"""
I wanted to see the shape of our feature space in reduced dimensionality
and also where the molecules I use for training and the ligands are located.

Furthermore, how does the eigenvalue spectrum of a higher dimensional space look
like. When looking at the spectrum for ALL features, is there an elbow at
dim(saito/haddad)?

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import numpy as np
import pylab as plt
from master.libs import run_lib
from master.libs import read_data_lib as rdl
from master.libs import utils

descriptor = 'saito_desc'
threshold = 0.5
outpath = '/Users/dedan/projects/master/results/features/'
config = {
    "data_path": os.path.join(os.path.dirname(__file__), '..', 'data'),
    "features": {
        "type": "conventional",
        "kernel_width": 100,
        "bin_width": 150,
        "use_intensity": False,
        "spec_type": "ir",

        "descriptor": descriptor,
        "normalize": True,
        "properties_to_add": []
    }
 }

# load features stuff
door2id = json.load(open(os.path.join(config['data_path'], 'door2id.json')))
used_glomeruli = json.load(open(os.path.join(config['data_path'], 'used_glomeruli.json')))
path_to_csv = os.path.join(config['data_path'], 'response_matrix.csv')
cas_numbers, glomeruli, rm = rdl.load_response_matrix(path_to_csv, door2id)
features = run_lib.prepare_features(config)

# do PCA
feature_matrix = np.matrix(features.values())
U, d, Vt = np.linalg.svd(feature_matrix, full_matrices=False)
projected_data = d[:2] *  np.asarray(feature_matrix * Vt[:2, :].T)
eigen_spectrum = d**2 / np.sum(d**2)

# plot PCA results
fig = plt.figure()
fig.suptitle(descriptor)
ax = fig.add_subplot(121)
ax.imshow(np.log(feature_matrix))
ax.set_title('input matrix log scaled')
ax.axis('off')
ax = fig.add_subplot(122)
ax.plot(eigen_spectrum[:50])
ax.set_title('first 50 entries of eigenvalue spectrum')
fig.savefig(os.path.join(outpath, descriptor + '_eigenvalue_spectrum.png'))

# plot molecules and ligands for each glomerulus in PCA space
fig = plt.figure()
fig.suptitle(descriptor)
N = utils.ceiled_root(len(used_glomeruli))
for i, glom in enumerate(used_glomeruli):
    glom_idx = glomeruli.index(glom)
    tmp_rm, tmp_cas_numbers = rdl.get_avail_targets_for_glom(rm, cas_numbers, glom_idx)
    avail_molids = [door2id[tc][0] for tc in tmp_cas_numbers]
    avail_mask = np.array([True if molid in avail_molids else False for molid in features],
                          dtype=bool)
    resp_idx = [door2id[tc][0] for j, tc in enumerate(tmp_cas_numbers) if tmp_rm[j] > threshold]

    # TODO: scale by eigen
    ax = fig.add_subplot(N, N, i+1)
    ax.plot(projected_data[~avail_mask,0], projected_data[~avail_mask,1], 'x', color='0.5')
    ax.plot(projected_data[avail_mask,0], projected_data[avail_mask,1], 'xb')
    ax.plot(projected_data[resp_idx,0], projected_data[resp_idx,1], 'xr')
    ax.axis('off')
    ax.set_title(glom)
plt.subplots_adjust(hspace=0.4)
fig.savefig(os.path.join(outpath, descriptor + '_pca_space.png'))


