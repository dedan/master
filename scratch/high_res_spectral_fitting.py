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
        'normalize_samples': False
    },
    "methods": {
        "svr": {
            "cross_val": True,
            "C": 1.0,
            "n_folds": 10
        }
    },
    "data_path": "/Users/dedan/projects/master/data",
    "glomerulus": "Or22a",
    "randomization_test": False
}

# get the molids
spectra = pickle.load(open('/Users/dedan/projects/master/data/spectral_features/large_base/parsed.pckl'))
_, targets, molids = run_lib.load_data_targets(config, {str(i): [] for i in spectra.keys()})

# some molids map to two CAS numbers for some molecules, use only first
first_molids_idx = sorted([molids.index(m) for m in set(molids)])
targets = targets[first_molids_idx]
data_orig = data_orig[first_molids_idx]
molids = [molids[i] for i in first_molids_idx]

# place the intensity values in a high-resolution vector
freqs = {k: v for k, v in spectra.items() if k in molids}
data_orig = flib._place_waves_in_vector(freqs, 0.01, True, 'ir')
assert len(molids) == len(targets) == data_orig.shape[0]

# fit model
sel_scores = run_lib.get_selection_score(config, data_orig, targets)
data = flib.select_k_best(data_orig, sel_scores, 2**9)
tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
model = tmp_res['svr']['model']