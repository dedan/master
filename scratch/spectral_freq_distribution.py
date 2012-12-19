#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import __builtin__
import sys
import os
import pickle, json
import itertools as it
import numpy as np
import pylab as plt
from collections import defaultdict
from master.libs import run_lib
from master.libs import features_lib as flib
from master.libs import utils
from scipy.spatial.distance import pdist
reload(run_lib)
plt.close('all')

glomerulus = 'Or22a'
active_thresh = 0.5
fpath = '/Users/dedan/projects/master/data/spectral_features/large_base/'

plt.close('all')
features = pickle.load(open(os.path.join(fpath, 'parsed.pckl')))

# only for molecules we always use and for a single glomerulus
config = {'data_path': os.path.join(os.path.dirname(__file__), '..', 'data'),
          'glomerulus': glomerulus}
data, targets, molids = run_lib.load_data_targets(config, features)

# some molids map to two CAS numbers for some molecules, use only first
first_molids_idx = sorted([molids.index(m) for m in set(molids)])
data = data[first_molids_idx]
targets = targets[first_molids_idx]
molids = [molids[i] for i in first_molids_idx]
assert len(molids) == len(targets) == data.shape[0]

freqs = {k: v for k, v in features.items() if k in molids}


# # histogram of frequencies
# fig = plt.figure()
# all_freqs = list(it.chain(*freqs.values()))
# ax = fig.add_subplot(111)
# ax.hist(all_freqs, 4000, range=[0, 4000])

# # target value histogram
# fig = plt.figure()
# ax = fig.add_subplot(211)
# ax.hist(targets)

# # frequency distribution of active targets
# active = [m for i, m in enumerate(molids) if targets[i] > active_thresh]
# act_freqs = list(it.chain(*[v for k, v in freqs.items() if k in active]))
# ax = fig.add_subplot(212)
# # do ligand share bins for resolution of 0.1?
# ax.hist(act_freqs, 4000, range=[0, 4000])

def pdist_1d(values, thresh=0.6):
    """like pdist but for 1-d lists"""
    res = []
    ligands = []
    for i in range(len(values)):
        for j in range(i+1, len(values)):
            res.append(np.abs(values[i] - values[j]))
            if values[i] > thresh and values[j] > thresh:
                ligands.append(True)
            else:
                ligands.append(False)
    return np.array(res), np.array(ligands)


fig = plt.figure()
ma = flib._place_waves_in_vector(freqs, 0.01, True, 'ir')
target_distances, ligands = pdist_1d(targets)

f_select_config = {'feature_selection': {'method': 'linear'}}
sel_scores = run_lib.get_selection_score(f_select_config, ma, targets)

ma_sel = flib.select_k_best(ma, sel_scores, 2**11)
feature_distances_sel = pdist(ma_sel, 'cosine')
ax = fig.add_subplot(111)
ax.plot(feature_distances_sel[~ligands], target_distances[~ligands], 'xb')
ax.plot(feature_distances_sel[ligands], target_distances[ligands], 'xr')

plt.show()