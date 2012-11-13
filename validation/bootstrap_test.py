#!/usr/bin/env python
# encoding: utf-8
"""
    do a randomization test on a feature_selection-preprocessing-model combination

    by shuffling the data within columns and the re-evaluating the result N times.
    The idea is that the result should be much worse for shuffled data because
    the observations should now be meaningless and not helpful to predict any
    target values. It would only perform equally well if the unshuffled
    observations were already meaningless (without target related structure)

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys
import os
from master.libs import run_lib
import numpy as np
import pylab as plt

n_repetitions = 100
config = {
    "feature_selection": {
        "method": "forest",
        "k_best": 50
    },
    "features": {
        "type": "conventional",
        "descriptor": "BURDEN_EIGENVALUES_DESCRIPTORS",
        "normalize": True,
        "normalize_samples": False
    },
    "methods": {
        "forest": {
            "n_estimators": 500,
            "oob_score": True,
            "random_state": 0
        },
    },
    "data_path": "/Users/dedan/projects/master/data",
    "glomerulus": "Or19a"
}

# load a previous feature_selection-preprocessing-model combination (its config)
config['randomization_test'] = False
features = run_lib.prepare_features(config)
true_res = run_lib.run_runner(config, features)

# add shuffle data to the config and run the runner for N times
config['randomization_test'] = True
rand_res = []
for i in range(n_repetitions):
    features = run_lib.prepare_features(config)
    tmp_res = run_lib.run_runner(config, features)
    rand_res.append(tmp_res['forest']['gen_score'])

# readout the resulting files and analyze them

# create a plot of the resulting distribution and the original value
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(rand_res)
plt.hold(True)
ax.plot([true_res['forest']['gen_score']], [1], 'r*')


