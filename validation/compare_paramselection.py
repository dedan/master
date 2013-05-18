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
import string
import __builtin__
import numpy as np
import pylab as plt
from scipy import stats
from master.libs import read_data_lib as rdl
from master.libs import utils
import itertools as it

params = {'axes.labelsize': 6,
    'font.size': 6,
    'legend.fontsize': 7,
    'xtick.labelsize':6,
    'ytick.labelsize': 6,
    'text.usetex': False,
    'mathtext.fontset':'stixsans'}
plt.rcParams.update(params)
print plt.rcParams['mathtext.fontset']
fig = plt.figure(figsize=(2.5, 1.8))
gs = plt.GridSpec(1,2)

inpath = '/Users/dedan/projects/master/results/param_search/nusvr'
method = 'svr'
selection = 'linear'
example_desc = 'all'
example_gloms = ['Or43b', 'Or67c']

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
used_glomeruli = json.load(open(os.path.join(data_path, 'all_glomeruli.json')))
search_res, max_overview, sc, k_best_dict = rdl.read_paramsearch_results(inpath)

out_res = {}
for i, descriptor in enumerate(search_res):

    res = json.load(open(os.path.join(inpath, descriptor + '.json')))

    # param selection values to compare against
    if method == 'forest':
        c_k_best = -1
        c_reg_idx = -1
    else:
        c_k_best = -1              # all features
        c_regularization = 1.0
        c_reg_idx = sc[method].index(c_regularization)      # regularization 1.0

    out_res[descriptor] = {'both_opt': [], 'none_opt': [], 'k_opt': [], 'reg_opt': [],
                           'labels': max_overview[method][selection]['glomeruli']}
    for glom in used_glomeruli:

        cur_res = search_res[descriptor][selection][glom][method]

        # optimal k_best and regularization
        out_res[descriptor]['both_opt'].append(np.max(cur_res))

        # use optimal k_best with fixed regularization
        k_best = np.argmax(cur_res[:, c_reg_idx])
        out_res[descriptor]['k_opt'].append(cur_res[k_best, c_reg_idx])

        # use optimal regularization with fixed k_best
        reg_idx = np.argmax(cur_res[c_k_best, :])
        out_res[descriptor]['reg_opt'].append(cur_res[c_k_best, reg_idx])

        # use fixed regularization and k_best
        out_res[descriptor]['none_opt'].append(cur_res[c_k_best, c_reg_idx])
    assert not (np.array(out_res[descriptor]['none_opt']) > out_res[descriptor]['k_opt']).any()


minimum = 0
maximum = np.max([np.max(search_res[example_desc][selection][glom][method]) for glom in example_gloms])
ad = ['A)', 'B)']
for i, glom in enumerate(example_gloms):
    mat = search_res[example_desc][selection][glom][method]
    ax = plt.subplot(gs[0,i])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['0.1', '1', '10'], rotation=90)
    ax.set_yticks(range(mat.shape[0]))
    yticklabels = [str(2**j) if j%2==0 else '' for j in range(mat.shape[0])]
    yticklabels[-1] = str(k_best_dict[example_desc][-1])
    ax.set_yticklabels(yticklabels)
    ax.set_title('{}'.format(glom))
    if i == 0:
        ax.set_ylabel('k best')
        ax.set_xlabel('$C$')
    else:
        ax.set_yticks([])
    ax.tick_params('both', length=0, width=2, which='major')
    im = ax.imshow(mat, interpolation='nearest', cmap=plt.cm.gray,
                    vmin=minimum, vmax=maximum)
    ax.text(-3,-1.2, ad[i], fontsize=9, weight='bold')
    if i == 1:
        ticks = [0, maximum]
        cbar = fig.colorbar(im, ticks=ticks, use_gridspec=True)#
        cbar.ax.set_yticklabels(['{:.2f}'.format(t) for t in ticks])
        cbar.set_label('$q^{2}$', fontsize=9)
gs.tight_layout(fig)
fig.savefig(os.path.join(inpath, 'plots', 'Fig3ab.png'), dpi=600)

fig = plt.figure(figsize=(3.5, 1.8))
gs = plt.GridSpec(1,2)
ad = ['C)', 'D)']
# which search dimension is more important
best_descs = ['haddad_desc', 'all', 'vib_100']
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
to_compare = ['k_opt', 'reg_opt']
titles = ['c)', 'd)']
for i, pick_type in enumerate(to_compare):

    reference = np.array(utils.flatten([r[reference_name] for r in vals]))
    improved = np.array(utils.flatten([r[pick_type] for r in vals]))
    # only plot values which reach genscore > 0 after paramsearch
    reference = reference[improved > 0]
    improved = improved[improved > 0]
    # don't care how negative it was before search, all < 0 are equally bad
    reference[reference < 0] = 0
    improved[improved < 0] = 0

#    fig = plt.figure(figsize=(2.23, 2))
    ax = plt.subplot(gs[0,i])#fig.add_subplot(1, 3, i+1)
    ax.plot(reference, improved, 'ko', alpha=0.6, markersize=4)
    ax.plot([0, line_max-0.05], [0, line_max-0.05], color='0.5')
    ax.set_ylabel(nice_names[pick_type])
    plt.axis('scaled')
    ax.set_xlim([-0.05, line_max])
    ax.set_ylim([0, line_max])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    #ax.set_title(titles[i])
    utils.simple_axis(ax)
    ax.set_xlabel(nice_names[reference_name])
    ax.text(-0.2, 0.91, ad[i], fontsize=9, weight='bold')
fig.tight_layout()
fig.savefig(os.path.join(inpath, 'plots', 'Fig3cd.png'), dpi=600)

