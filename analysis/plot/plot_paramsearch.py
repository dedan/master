#!/usr/bin/env python
# encoding: utf-8
"""
    plot: regularization on x axis, number of k_best features on y

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys
import os
import json
import numpy as np
import pylab as plt
import itertools as it
from libs import plot_lib as plib
from libs import read_data_lib as rdl
from libs import utils
import matplotlib.gridspec as gridspec
reload(plib)
reload(rdl)


params = {'axes.labelsize': 6,
    'font.size': 6,
    'legend.fontsize': 7,
    'xtick.labelsize':6,
    'ytick.labelsize': 6}
plt.rcParams.update(params)

config = json.load(open(sys.argv[1]))
outpath = os.path.join(config['inpath'], 'plots')
if not os.path.exists(outpath):
    os.mkdir(outpath)

# variables for results
plt.close('all')
search_res, max_overview, sc, _ = rdl.read_paramsearch_results(config['inpath'],
                                                               p_selection=config.get('selection', {}))

if config['plot_param_space']:
    for desc in search_res:
        fig = plt.figure()
        plib.plot_search_matrix(fig, search_res[desc], config['fselection'],
                                config['method'], config.get('glomeruli', []))
        fig.savefig(os.path.join(outpath, config['method'] + '_' + desc + '.' + config['format']))

# descriptor method performance plots
fig = plt.figure(figsize=(3.35, 2))
ptype = config['descriptor_plot_type']
plib.new_descriptor_performance_plot(fig, max_overview, config['fselection'],
                                     config['method'],
                                     config.get('glomeruli', []),
                                     ptype)
fig.subplots_adjust(bottom=0.25)
fig.savefig(os.path.join(outpath, ptype + '_desc_comparison.' + config['format']), dpi=600)
#plt.show()


# ML method comparison plot
markers = ['1', '0']
desc2comp = ['EVA_100', 'all']
fig = plt.figure(figsize=(3.35, 1.8))
ax = fig.add_subplot(111)
desc1_collect, desc2_collect = [], []
for i, desc in enumerate(desc2comp):
    desc_idx1 = max_overview['svr']['linear']['desc_names'].index(desc)
    desc_idx2 = max_overview['forest']['forest']['desc_names'].index(desc)
    desc1_collect.extend(max_overview['svr']['linear']['p_selection'][desc_idx1, :])
    desc2_collect.extend(max_overview['forest']['forest']['p_selection'][desc_idx2, :])
    ax.plot(max_overview['svr']['linear']['p_selection'][desc_idx1, :],
            max_overview['forest']['forest']['p_selection'][desc_idx2, :],
            'o', mfc=markers[i],
            label=desc,
            markersize=5)
ax.plot([0, 0.8], [0, 0.8], color='0.5')
plt.axis('scaled')
ax.set_xlim([0, .9])
ax.set_ylim([0, .9])
ax.set_xlabel('SVR (q2)')
ax.set_ylabel('RFR (q2)')
utils.simple_axis(ax)
ax.legend(loc='upper left', numpoints=1, frameon=False, prop={'size': 'small'}, bbox_to_anchor=(0.01, 1))
ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ticklabels = ['0', '', '.2', '', '.4', '', '.6', '', '.8', '']
ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels)
fig.subplots_adjust(bottom=0.2)
fig.tight_layout()
fig.savefig(os.path.join(outpath, 'best_method_comparison.' + config['format']), dpi=600)

assert len(desc1_collect) == len(desc2_collect)
svr_better = np.sum([1 for d1, d2 in zip(desc1_collect, desc2_collect) if d1 > d2])
rfr_better = np.sum([1 for d1, d2 in zip(desc1_collect, desc2_collect) if d1 < d2])
ratio = float(svr_better) / (np.sum(rfr_better) + np.sum(svr_better))
print('svr better than rfr in {:.2f} \% of the cases'.format(ratio))

if utils.run_from_ipython():
    plt.show()
