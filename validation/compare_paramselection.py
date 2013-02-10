#!/usr/bin/env python
# encoding: utf-8
"""
Compare the result of optimal feature selection and regularization values with
the results for some fixed settings. I had a look at this because we had the
feeling that for the SVR we don't need any feature selection and special
regularization. It works already good with standard values, but still we want
to see the drop of performance we'll have.

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import glob
import __builtin__
import numpy as np
import pylab as plt
from scipy import stats
from master.libs import read_data_lib as rdl
from master.libs import utils
import itertools as it

inpath = '/Users/dedan/projects/master/results/param_search/merged'
method = 'svr'
selection = 'linear'

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
used_glomeruli = json.load(open(os.path.join(data_path, 'all_glomeruli.json')))
search_res, max_overview, sc, k_best_dict = rdl.read_paramsearch_results(inpath)

out_res = {}
for i, descriptor in enumerate(search_res):

    res = json.load(open(os.path.join(inpath, descriptor + '.json')))

    # param selection values to compare against
    if method == 'forest':
        sc[method] = res['sc'][method]
        sc['svr'] = res['sc']['svr']
        c_k_best = np.max(sc['k_best'])
        c_reg_idx = np.min((len(sc[method])-1, len(sc['svr'])-1))
    else:
        c_k_best = np.max(res['sc']['k_best'])
        c_regularization = 1.0
        c_reg_idx = sc[method].index(c_regularization)

    best_genscore = []
    picked_genscore = []
    for glom in used_glomeruli:
        best_params = rdl.get_best_params(max_overview, sc, k_best_dict,
                                          descriptor, glom, method, selection)
        k_best = best_params['feature_selection']['k_best']
        reg_idx = res['sc'][method].index(best_params['methods'][method]['regularization'])
        best_res = res['res'][selection][glom][str(k_best)][str(reg_idx)]
        best_genscore.append(best_res[method]['gen_score'])
        if method == 'forest':
            picked_res = res['res'][selection][glom][str(k_best)][str(c_reg_idx)]
        else:
            picked_res = res['res'][selection][glom][str(c_k_best)][str(c_reg_idx)]
        picked_genscore.append(picked_res[method]['gen_score'])
    out_res[descriptor] = {'best_genscore': best_genscore,
                           'picked_genscore': picked_genscore,
                           'labels': max_overview[method][selection]['glomeruli']}


fig = plt.figure()
ax = fig.add_subplot(211)

# use only glomeruli for which paramsearch result > 0
best_descs = ['haddad_desc', 'GETAWAY', 'all', 'vib_100']
vals = [out_res[k] for k in best_descs]
all_best = np.array(utils.flatten([r['best_genscore'] for r in vals]))
all_picked = np.array(utils.flatten([r['picked_genscore'] for r in vals]))
# only look at gloms for which paramsearch yields genscores above 0
all_picked = all_picked[all_best > 0]
all_best = all_best[all_best > 0]
# everything below 0 is equally bad
all_picked[all_picked < 0] = 0
diffs = np.abs(all_best - all_picked)
score_perc = stats.scoreatpercentile(diffs, 90)

ax.plot(all_picked, diffs, 'ko', alpha=0.6, markersize=4)
ax.set_ylim([-0.1, 0.4])
ax.plot([0, 1], [score_perc, score_perc], color='0.5')
ax.set_xlabel('fixed parameter result')
ax.set_ylabel('performance gain')
ax.set_xlim([0, 0.8])
utils.simple_axis(ax)


fig = plt.figure()
ax = fig.add_subplot(212)
line_max = 0.9
vals = [out_res[k] for k in best_descs]
all_best = np.array(utils.flatten([r['best_genscore'] for r in vals]))
all_picked = np.array(utils.flatten([r['picked_genscore'] for r in vals]))
# only look at gloms for which paramsearch yields genscores above 0
all_picked = all_picked[all_best > 0]
all_best = all_best[all_best > 0]

ax.plot(all_best, all_picked, 'ko', alpha=0.6, markersize=4)
ax.plot([0, line_max], [0, line_max], color='0.5')
ax.set_xlabel('param search results')
ax.set_ylabel('fixed parameter')
ax.set_title(method)
utils.simple_axis(ax)

ax.set_xlim([0, 0.8])
ax.set_ylim([0, 0.8])
fig.savefig(os.path.join(inpath, 'plots', method + '_param_selection_overview.png'))
plt.show()


