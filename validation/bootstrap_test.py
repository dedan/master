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
import copy
from master.libs import run_lib
from master.libs import read_data_lib as rdl
from master.libs import features_lib as flib
import numpy as np
import pylab as plt
reload(run_lib)

plt.close('all')
n_repetitions = 10
desc = 'haddad_desc'
selection = 'linear'
method = 'svr'
# TODO: maybe search for a good glomerulus
glom = "Or22a"
path = '/Users/dedan/mnt/numbercruncher/results/param_search/conv_features/'

# get the best parameters from the parameter search
search_res, max_overview, sc, _ = rdl.read_paramsearch_results(path)

# get the base config
config_orig = sc['runner_config_content']
config_orig['features']['descriptor'] = desc
mat = search_res[desc][selection][glom][method]
k_best_idx = np.argmax(np.max(mat, axis=1))
reg_idx = np.argmax(np.max(mat, axis=0))
if 'svr' in method:
    config_orig['methods'][method]['C'] = sc['svr'][reg_idx]
else:
    config_orig[method]['n_estimators'] = sc['forest'][reg_idx]
config_orig['feature_selection']['method'] = selection
config_orig['glomerulus'] = glom


for method in config_orig['methods']:

    print method
    config = copy.deepcopy(config_orig)
    for m in list(config['methods'].keys()):
        if not method in m:
            del(config['methods'][m])
    config['randomization_test'] = False
    features = run_lib.prepare_features(config)
    data, targets, _ = run_lib.load_data_targets(config, features)
    sel_scores = run_lib.get_selection_score(config, data, targets)
    data = flib.select_k_best(data, sel_scores, sc['k_best'][k_best_idx])
    true_res = run_lib.run_runner(config, data, targets)

    # add shuffle data to the config and run the runner for N times
    config['randomization_test'] = True
    rand_res = []
    fig = plt.figure()
    fig.suptitle(method)
    ax = fig.add_subplot(211)
    for i in range(n_repetitions):
        orig_data, orig_targets, _ = run_lib.load_data_targets(config, features)
        data, targets = orig_data.copy(), orig_targets.copy()
        sel_scores = run_lib.get_selection_score(config, data, targets)
        data = flib.select_k_best(data, sel_scores, sc['k_best'][k_best_idx])
        tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
        rand_res.append(tmp_res[method]['gen_score'])
        ax.plot(orig_targets, tmp_res[method]['model'].predict(orig_data), 'x')

    # create a plot of the resulting distribution and the original value
    ax = fig.add_subplot(212)
    ax.hist(rand_res)
    ax.plot([true_res[method]['gen_score']], [1], 'r*')
    plt.show()

