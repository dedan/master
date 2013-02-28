#!/usr/bin/env python
# encoding: utf-8
"""
plot the predictions made by compute_predictions.py

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import pickle
import numpy as np
import pylab as plt
from libs import utils
from scipy import stats

params = {'axes.labelsize': 6,
    'font.size': 6,
    'legend.fontsize': 7,
    'xtick.labelsize':6,
    'ytick.labelsize': 6}
plt.rcParams.update(params)

inpath = os.path.join(os.environ['HOME'], 'Projects', 'EVA', 'PhyspropVsEVA', 
                      'data')
reference = 'all'
example_receptor = 'ac3a'
comparison_desc = 'eva'
res = pickle.load(open(os.path.join(inpath, 'predictions.pkl')))
del res['getaway']
to_compare = set(res.keys()).difference([reference])

fig = plt.figure()
plot_res = {desc:{'ps': [], 'corrs': []} for desc in to_compare}
for desc in to_compare:

    for glom in res[reference]:
        corr, p = stats.pearsonr(res[reference][glom]['predictions'],
                                 res[desc][glom]['predictions'])
        plot_res[desc]['corrs'].append(corr)
        plot_res[desc]['ps'].append(p)

for desc, pres in plot_res.items():
    print '{} mean r: {:.2f}'.format(desc, np.mean(pres['corrs']))


ax = fig.add_subplot(1,2,1)
ax.hist(utils.flatten([v['ps'] for v in plot_res.values()]))

fig = plt.figure(figsize=(3.35, 1.8))
marker = ['o', 's', 'd']
xticks = [0, 0.2, 0.4, 0.6, 0.8]
xticklabels = ['0', '.2', '.4', '.6', '.8']
ax = fig.add_subplot(1,2,1)
ax.plot([-0.05, 0.8], [-0.05, 0.8], color='0.6')
col_ref = []
col_comp = []
for i, (desc, pres) in enumerate(plot_res.items()):
    compare_scores = np.array([res[desc][g]['score'] for g in res[desc]])
    ref_scores = np.array([res[reference][g]['score'] for g in res[reference]])
    compare_scores[compare_scores < 0] = 0
    ref_scores[ref_scores < 0] = 0
    col_ref.extend(ref_scores)
    col_comp.extend(compare_scores)
    ax.plot(compare_scores, ref_scores,
            '.', marker=marker[i], color='0.5', markeredgecolor='0.3',
            markersize=4,
            label=desc.upper() if not desc == 'eva' else 'EVA_100')
comp_score = res[comparison_desc][example_receptor]['score']
ref_score = res[reference][example_receptor]['score']
ax.plot(comp_score, ref_score, 'o', color='0.5', markeredgecolor='0.3', markersize=4)
ax.plot(comp_score, ref_score, 'x', color='0.0', markersize=3)
plt.axis('scaled')
ax.annotate(example_receptor, xy=(comp_score, ref_score), xytext=(0.65, 0.465),
            arrowprops=dict(facecolor='black', shrink=0.2, width=1, frac=0.2, headwidth=3))
ax.text(0.55, 0.75, 'r:{:.2f}'.format(stats.pearsonr(col_comp, col_ref)[0]))
print stats.pearsonr(col_comp, col_ref)
ax.set_yticks(xticks)
ax.set_yticklabels(xticklabels)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim([-0.05, 0.9])
ax.set_ylim([-0.05, 0.9])
ax.set_xlabel('comparison descriptor (q2)')
ax.set_ylabel('ALL (q2)')
ax.legend(loc='lower right', numpoints=1, frameon=False, fancybox=True, prop={'size': 'small'})
utils.simple_axis(ax)
#fig.subplots_adjust(bottom=0.2)
#ax.set_title('a)')
ax.text(-0.3,0.85, 'A)', fontsize=9, weight='bold')
#fig.savefig(os.path.join(inpath, 'q2_comparison.svg'), dpi=300)

ax = fig.add_subplot(1,2,2)
ax.plot([-0.1, 0.8], [-0.1, 0.8], color='0.6')
ref_predictions = res[reference][example_receptor]['predictions']
comp_predictions = res[comparison_desc][example_receptor]['predictions']
ax.plot(ref_predictions, comp_predictions, 'ko', color='0.5',
        markeredgecolor='0.3',
        markersize=4,
        label=example_receptor)
plt.axis('scaled')
ax.set_yticks(xticks)
ax.set_yticklabels(xticklabels)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlabel('EVA_100 predictions\n(normalized response)')
ax.set_ylabel('ALL predictions')
ax.set_xlim([-0.1, 0.9])
ax.set_ylim([-0.1, 0.9])
# ax.set_title(example_receptor)
ax.legend(loc='lower right', numpoints=1, frameon=False, fancybox=True,
          prop={'size': 'small'})
utils.simple_axis(ax)
ax.text(0.55, 0.75, 'r:{:.2f}'.format(stats.pearsonr(ref_predictions, comp_predictions)[0]))
print stats.pearsonr(ref_predictions, comp_predictions)
fig.subplots_adjust(bottom=0.2)
#ax.set_title('b)')
ax.text(-0.3,0.85, 'B)', fontsize=9, weight='bold')
#fig.savefig(os.path.join(inpath, 'prediction_comparison_{}.svg'.format(example_receptor)), dpi=300)
fig.tight_layout()
fig.savefig(os.path.join(inpath, 'Fig2.png'.format(example_receptor)), dpi=600)

if utils.run_from_ipython():
    plt.show()