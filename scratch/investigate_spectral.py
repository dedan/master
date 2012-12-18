#!/usr/bin/env python
# encoding: utf-8
'''

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import os
import json
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from master.libs import run_lib
from master.libs import features_lib as flib
reload(rdl)

plt.close('all')
method = 'svr'
selection = 'linear'
kwidth = 1
inpath = '/Users/dedan/projects/master/results/param_search/spectral_features_new'
data_path = "/Users/dedan/projects/master/data"
glom = 'Or22a'


# get the best parameters from the search result
search_res, max_overview, sc, k_best = rdl.read_paramsearch_results(inpath)
config = sc['runner_config_content']
config['features']['kernel_width'] = kwidth
config['glomerulus'] = glom
config['feature_selection']['method'] = selection

cur_max = max_overview[method][selection]
desc_idx = cur_max['desc_names'].index(repr(kwidth))
glom_idx = cur_max['glomeruli'].index(glom)
best_c_idx = int(cur_max['c_best'][desc_idx, glom_idx])
best_kbest_idx = int(cur_max['k_best'][desc_idx, glom_idx])
config['methods']['svr']['C'] = 1 #sc['svr'][best_c_idx]
config['feature_selection']['k_best'] = k_best[repr(kwidth)][best_kbest_idx]


features = run_lib.prepare_features(config)
data, targets, molids = run_lib.load_data_targets(config, features)


# fit model
print("use {} molecules for training".format(data.shape[0]))
sel_scores = run_lib.get_selection_score(config, data, targets)
data = flib.select_k_best(data, sel_scores, config['feature_selection']['k_best'])
tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
model = tmp_res[method]['model']

fig = plt.figure()
# histogram of the model coefficients
c = model.dual_coef_[0]
ax = fig.add_subplot(311)
ax.hist(c)

ax = fig.add_subplot(312)
ax.plot(model.support_vectors_[c > 0.5, :].T, linewidth=2)
ax.plot(model.support_vectors_[c < -0.5, :].T, linewidth=0.5)

# get top N ligands
N = 5
idx = np.argsort(targets)
top_ligands = data[idx[-N:], :]
top_targets = targets[idx[-N:]]

for i in range(N):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(model.support_vectors_[c > 0.5, :].T, linewidth=2)
    ax.plot(model.support_vectors_[c < -0.5, :].T, linewidth=0.5)
    ax.plot(-top_ligands[i])
    ax.set_ylabel('{:.2f}'.format(top_targets[i]), rotation='0')
    plt.subplots_adjust(hspace=0.4)

