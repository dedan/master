#!/usr/bin/env python
# encoding: utf-8
"""
We decided to use oob score for parameter selection but real cross validation

to estimate the generalization error. This script reads from an already
existing results file, reads out the optimal parameters from the parameter
search matrix and computes the cross validation generalization score for it.

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import pickle
import glob
import copy
import __builtin__
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from master.libs import run_lib
from master.libs import utils
from master.libs import plot_lib as plib

inpath = '/Users/dedan/projects/master/results/final_plots/svr_overview'
method = 'forest'
selection = 'forest'
n_folds = 30
boxplot = False

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
search_res, max_overview, sc, k_best_dict = rdl.read_paramsearch_results(inpath)
glomeruli = json.load(open(os.path.join(data_path, 'used_glomeruli.json')))

new_maxes = np.zeros((len(search_res), len(glomeruli)))
for i, descriptor in enumerate(search_res):

    print descriptor
    # load param search config
    res = json.load(open(os.path.join(inpath, descriptor + '.json')))
    config = copy.deepcopy(res['sc']['runner_config_content'])
    config['methods'][method]['cross_val'] = 'xval'
    config['methods'][method]['n_folds'] = n_folds
    features = run_lib.prepare_features(config)

    # no regularization
    if 'max_depth' in config['methods'][method]:
        del config['methods'][method]['max_depth']

    for j, glom in enumerate(glomeruli):

        print glom
        # read optimal k_best parameter from previous parameter search
        config['glomerulus'] = glom
        best_params = rdl.get_best_params(max_overview, sc, k_best_dict,
                                          descriptor, glom, method, selection)
        k_best = best_params['feature_selection']['k_best']
        config['feature_selection']['k_best'] = k_best

        # recompute results with cross validation
        data, targets, molids = run_lib.load_data_targets(config, features)
        run_res = run_lib.run_runner(config, data, targets)
        new_maxes[i, j] = run_res[method]['gen_score']
        print run_res[method]['gen_score']

# save and plot the results
pickle.dump(new_maxes, open(os.path.join(inpath, 'xval_maxes.pckl'), 'w'))
max_overview[method][selection]['max'] = new_maxes
fig = plt.figure(figsize=(15,5))
plib.new_descriptor_performance_plot(fig, max_overview, sc, boxplot)
fig.subplots_adjust(bottom=0.3)
fig.savefig(os.path.join(inpath, 'desc_comparison_xval.png'))


