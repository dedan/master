#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import glob
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from master.libs import utils

inpath = '/Users/dedan/projects/master/results/new_param_search/conv_features'
method = 'svr'
selection = 'linear'

search_res, max_overview, sc, k_best_dict = rdl.read_paramsearch_results(inpath)

out_res = {}
for i, descriptor in enumerate(search_res):

    res = json.load(open(os.path.join(inpath, descriptor + '.json')))

    # param selection values to compare against
    c_k_best = np.max(res['sc']['k_best'])
    c_regularization = 1.0
    c_reg_idx = sc[method].index(c_regularization)

    best_genscore = []
    picked_genscore = []
    for glom in search_res[descriptor][selection]:
        best_params = rdl.get_best_params(max_overview, sc, k_best_dict,
                                          descriptor, glom, method, selection)
        k_best = best_params['feature_selection']['k_best']
        reg_idx = res['sc'][method].index(best_params['methods'][method]['regularization'])
        best_res = res['res'][selection][glom][str(k_best)][str(reg_idx)]
        best_genscore.append(best_res[method]['gen_score'])
        picked_res = res['res'][selection][glom][str(c_k_best)][str(c_reg_idx)]
        picked_genscore.append(picked_res[method]['gen_score'])
    out_res[descriptor] = {'best_genscore': best_genscore,
                           'picked_genscore': picked_genscore,
                           'labels': res['res'][selection].keys()}


fig = plt.figure(figsize=(10, 10))
n_sub = utils.ceiled_root(len(out_res))
for i, descriptor in enumerate(out_res):
    # ax = fig.add_subplot(111)
    ax = fig.add_subplot(n_sub, n_sub, i + 1)
    n_gloms = len(out_res[descriptor]['best_genscore'])
    ax.bar(range(0, n_gloms*2, 2), out_res[descriptor]['best_genscore'], color='#D95B43')
    ax.bar(range(1, n_gloms*2 + 1, 2), out_res[descriptor]['picked_genscore'], color='#ECD078')
    ax.set_title(descriptor[:15], fontsize=8)
    ax.set_xticks(np.arange(0, n_gloms*2, 2)+1.)
    ax.set_xticklabels(out_res[descriptor]['labels'], rotation='90', fontsize=7)
    ax.set_yticks([0, 0.5, 0]) if (i + 1) % n_sub == 1 else ax.set_yticks([])
    ax.set_ylim([0, 1])
fig.subplots_adjust(hspace=0.7)
plt.show()
