#!/usr/bin/env python
# encoding: utf-8
"""
fit SVR for high-resolution spectral descriptor

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import pickle
import json
import numpy as np
import pylab as plt
from master.libs import run_lib
from master.libs import features_lib as flib
from master.libs import read_data_lib as rdl
reload(flib)
plt.close('all')
data_path = '/Users/dedan/projects/master/data/'

config = {
    "feature_selection": {
        "method": "linear"
    },
    "features": {
        "normalize": True,
        "normalize_samples": False,
        "type": "spectral",
        "resolution": 0.1,
        "spec_type": "ir",
        "use_intensity": True,
        "kernel_width": 1,
        "properties_to_add": []
    },
    "methods": {
        "svr": {
            "cross_val": True,
            "C": 1.0,
            "n_folds": 10
        },
        "svr_ens": {
            "n_estimators": 500,
            "oob_score": True
        }
    },
    "data_path": "/Users/dedan/projects/master/data",
    "glomerulus": "Or22a",
    "randomization_test": False
}

# only get features for available molecules, otherwise matrix too large
fpath = '/Users/dedan/projects/master/data/spectral_features/large_base/'
door2id = json.load(open(os.path.join(data_path, 'door2id.json')))
csv_path = os.path.join(data_path, 'response_matrix.csv')
cas_numbers, glomeruli, rm = rdl.load_response_matrix(csv_path, door2id)
glom_idx = glomeruli.index(config['glomerulus'])
targets , tmp_cas_numbers = rdl.get_avail_targets_for_glom(rm, cas_numbers, glom_idx)
molids = [str(door2id[cas_number][0]) for cas_number in tmp_cas_numbers]

# spectra = pickle.load(open(os.path.join(fpath, 'parsed.pckl')))
# spectra = {k: v for k, v in spectra.items() if k in molids}
# features = flib.get_spectral_features(spectra, 0.5, use_intensity=True,
#                                                spec_type='ir',
#                                                kernel_widths=1)
# features = flib.remove_invalid_features(features)
# features = flib.normalize_features(features)
features = run_lib.prepare_features(config)

data, targets, molids = run_lib.load_data_targets(config, features)

# some molids map to two CAS numbers for some molecules, use only first
first_molids_idx = sorted([molids.index(m) for m in set(molids)])
targets = targets[first_molids_idx]
data = data[first_molids_idx]
molids = [molids[i] for i in first_molids_idx]


# freqs = {k: v for k, v in features.items() if k in molids}
# data = flib._place_waves_in_vector(freqs, 0.01, True, 'ir')
assert len(molids) == len(targets) == data.shape[0]


# fit model
sel_scores = run_lib.get_selection_score(config, data, targets)
data = flib.select_k_best(data, sel_scores, 2**9)
tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
model = tmp_res['svr_ens']['model']