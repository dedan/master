#!/usr/bin/env python
# encoding: utf-8
"""
Compare the result of optimal feature selection and regularization values with
the results for some fixed settings. I had a look at this because we had the
feeling that for the SVR we don't need any feature selection and special
regularization. It works already good with standard values, but still we want
to see the gain of performance we'll have.

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

inpath = '/Users/dedan/projects/master/results/param_search/nusvr'
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
        c_k_best = np.max(res['sc']['k_best'])              # all features
        c_regularization = 1.0
        c_reg_idx = sc[method].index(c_regularization)      # regularization 1.0

    out_res[descriptor] = {'both_opt': [], 'none_opt': [], 'k_opt': [], 'reg_opt': [],
                           'labels': max_overview[method][selection]['glomeruli']}
    for glom in used_glomeruli:
        best_params = rdl.get_best_params(max_overview, sc, k_best_dict,
                                          descriptor, glom, method, selection)
        k_best = best_params['feature_selection']['k_best']
        reg_idx = res['sc'][method].index(best_params['methods'][method]['regularization'])

        # optimal k_best and regularization
        tmp = res['res'][selection][glom][str(k_best)][str(reg_idx)]
        out_res[descriptor]['both_opt'].append(tmp[method]['gen_score'])
        # use optimal k_best with fixed regularization
        tmp = res['res'][selection][glom][str(k_best)][str(c_reg_idx)]
        out_res[descriptor]['k_opt'].append(tmp[method]['gen_score'])
        # use optimal regularization with fixed k_best
        tmp = res['res'][selection][glom][str(c_k_best)][str(reg_idx)]
        out_res[descriptor]['reg_opt'].append(tmp[method]['gen_score'])
        # use fixed regularization and k_best
        tmp = res['res'][selection][glom][str(c_k_best)][str(c_reg_idx)]
        out_res[descriptor]['none_opt'].append(tmp[method]['gen_score'])


# which search dimension is more important
best_descs = ['haddad_desc', 'GETAWAY', 'all', 'vib_100']
nice_names = {'none_opt': 'default parameters (q2)',
              'k_opt': 'k_best optimized (q2)',
              'reg_opt': 'C optimized (q2)',
              'both_opt': 'both optimized (q2)',
              }
line_max = 0.9
ticks = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
ticklabels = ['0', '.2', '.4', '.6', '.8', '']
vals = [out_res[k] for k in best_descs]
reference_name = 'none_opt'
to_compare = ['k_opt', 'reg_opt'] #, 'both_opt']
fig = plt.figure()
for i, pick_type in enumerate(to_compare):

    reference = np.array(utils.flatten([r[reference_name] for r in vals]))
    improved = np.array(utils.flatten([r[pick_type] for r in vals]))
    # only plot values which reach genscore > 0 after paramsearch
    reference = reference[improved > 0]
    improved = improved[improved > 0]
    # don't care how negative it was before search, all < 0 are equally bad
    reference[reference < 0] = 0

    ax = fig.add_subplot(1, len(to_compare), i+1)
    ax.plot(reference, improved, 'ko', alpha=0.6, markersize=4)
    ax.plot([0, line_max-0.05], [0, line_max-0.05], color='0.5')
    ax.set_xlabel(nice_names[reference_name])
    ax.set_ylabel(nice_names[pick_type])
    ax.set_title(method.upper())
    plt.axis('scaled')
    ax.set_xlim([-0.05, line_max])
    ax.set_ylim([0, line_max])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    utils.simple_axis(ax)

fig.savefig(os.path.join(inpath, 'plots', 'param_selection_overview.png'))
plt.show()


