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
from master.libs import read_data_lib as rdl
from master.libs import utils

inpath = '/Users/dedan/projects/master/results/param_search/forest'
method = 'forest'
selection = 'forest'

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
used_glomeruli = json.load(open(os.path.join(data_path, 'used_glomeruli.json')))
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


fig = plt.figure(figsize=(10, 10))
n_sub = utils.ceiled_root(len(out_res))
sorted_out_res = sorted(out_res.items(),
                        key=lambda (k, v): np.mean(v['best_genscore']),
                        reverse=True)
for i, (descriptor, results) in enumerate(sorted_out_res):
    ax = fig.add_subplot(n_sub, n_sub, i + 1)
    n_gloms = len(results['best_genscore'])
    ax.bar(range(0, n_gloms*2, 2), results['best_genscore'], color='#D95B43')
    ax.bar(range(1, n_gloms*2 + 1, 2), results['picked_genscore'], color='#ECD078')
    ax.set_title('{} ({:.2f})'.format(descriptor[:15], np.mean(results['best_genscore'])) , fontsize=8)
    ax.set_xticks(np.arange(0, n_gloms*2, 2)+1.)
    ax.set_xticklabels(results['labels'], rotation='90', fontsize=7)
    ax.set_yticks([0, 0.5, 0]) if (i + 1) % n_sub == 1 else ax.set_yticks([])
    ax.set_ylim([0, 1])
fig.subplots_adjust(hspace=0.7)
fig.savefig(os.path.join(inpath, 'plots', method + '_param_selection_comp.png'))

fig = plt.figure()
ax = fig.add_subplot(111)
all_best_genscores = __builtin__.sum([r['best_genscore'] for r in out_res.values()], [])
all_picked_genscores = __builtin__.sum([r['picked_genscore'] for r in out_res.values()], [])
ax.plot(all_best_genscores, all_picked_genscores, 'o', alpha=0.6, markersize=4)
ax.plot([0, 0.8], [0, 0.8], color='0.5')
ax.plot([0.1, 0.9], [0, 0.8], color='0.5')
ax.set_xlabel('param search results')
ax.set_ylabel('fixed parameter')
ax.set_title(method)
utils.simple_axis(ax)

ax.set_xlim([0, 0.8])
ax.set_ylim([0, 0.8])
fig.savefig(os.path.join(inpath, 'plots', method + '_param_selection_overview.png'))
plt.show()


