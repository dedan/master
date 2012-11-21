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
import json
import sys
import os
from master.libs import run_lib
from master.libs import read_data_lib as rdl
from master.libs import features_lib as flib
import numpy as np
import pylab as plt

n_repetitions = 100
desc = 'haddad_desc'
selection = 'linear'
method = 'svr'
# TODO: maybe search for a good glomerulus
glom = 'Or19a'
path = '/Users/dedan/mnt/numbercruncher/results/param_search/conv_features/'

# get the best parameters from the parameter search
search_res, max_overview, sc = rdl.read_paramsearch_results(path, [method])

# get the base config
config = sc['runner_config_content']
config['features']['descriptor'] = desc
mat = search_res[desc][selection][glom][method]
k_best_idx = np.argmax(np.max(mat, axis=1))
reg_idx = np.argmax(np.max(mat, axis=0))
print sc
print 'svr' in sc
if 'svr' in method:
    config['methods'][method]['C'] = sc['svr'][reg_idx]
else:
    config[method]['n_estimators'] = sc['forest'][reg_idx]
config['feature_selection']['method'] = selection


config['randomization_test'] = False
features = run_lib.prepare_features(config)
data, targets = run_lib.load_data_targets(config, features)
sel_scores = run_lib.get_selection_score(config, data, targets)
data = flib.select_k_best(data, sel_scores, sc['k_best'][k_best_idx])
true_res = run_lib.run_runner(config, data, targets)

# add shuffle data to the config and run the runner for N times
config['randomization_test'] = True
rand_res = []
for i in range(n_repetitions):
    data, targets = run_lib.load_data_targets(config, features)
    sel_scores = run_lib.get_selection_score(config, data, targets)
    data = flib.select_k_best(data, sel_scores, sc['k_best'][k_best_idx])
    tmp_res = run_lib.run_runner(config, data, targets)
    rand_res.append(tmp_res['forest']['gen_score'])

# readout the resulting files and analyze them

# create a plot of the resulting distribution and the original value
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(rand_res)
plt.hold(True)
ax.plot([true_res['forest']['gen_score']], [1], 'r*')


