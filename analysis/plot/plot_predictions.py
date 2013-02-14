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
from master.libs import utils
from scipy import stats

inpath = '/Users/dedan/projects/master/results/predict/'
reference = 'all'
res = pickle.load(open(os.path.join(inpath, 'predictions.pkl')))
to_compare = set(res.keys()).difference([reference])

fig = plt.figure()
xticks = np.arange(0, 1.1, 0.25)
xticklabels = ['0', '', '.5', '', '1']
bins = np.arange(0, 1.01, 0.05)
plot_res = {desc:{'corr_both_pos': [], 'corr_one_neg': [], 'all': []} for desc in to_compare}
for desc in to_compare:

    for glom in res[reference]:
        corr, _ = stats.pearsonr(res[reference][glom]['predictions'],
                                 res[desc][glom]['predictions'])
        plot_res[desc]['all'].append(corr)
        if res[reference][glom]['score'] <= 0 or res[desc][glom]['score'] <=0:
            plot_res[desc]['corr_one_neg'].append(corr)
        else:
            plot_res[desc]['corr_both_pos'].append(corr)

for desc, pres in plot_res.items():
    scor_perc = stats.scoreatpercentile(pres['all'], 10)
    print desc, scor_perc

fig = plt.figure()
marker = ['s', 'o', 'd']
xticks = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
xticklabels = ['0', '.2', '.4', '.6', '.8', '']
ax = fig.add_subplot(111)
ax.plot([0, 0.8], [0, 0.8], color='0.6')
for i, (desc, pres) in enumerate(plot_res.items()):
    compare_scores = np.array([res[desc][g]['score'] for g in res[desc]])
    ref_scores = np.array([res[reference][g]['score'] for g in res[reference]])
    compare_scores[compare_scores < 0] = 0
    ref_scores[ref_scores < 0] = 0
    ax.plot(compare_scores, ref_scores,
            '.', marker=marker[i], color='0.5', markeredgecolor='0.3',
            label=desc.upper() if not desc == 'eva' else 'VIB_100')
ax.set_yticks(xticks)
ax.set_yticklabels(xticklabels)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlim([-0.05, 0.9])
ax.set_ylim([-0.05, 0.9])
ax.set_xlabel('comparison descriptor (q2)')
ax.set_ylabel('ALL (q2)')
ax.legend(loc='lower right', numpoints=1, frameon=True, fancybox=True)
utils.simple_axis(ax)


example_receptor = 'Or22a'
comparison_desc = 'eva'
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([0, 0.8], [0, 0.8], color='0.6')
ref_predictions = res[reference][example_receptor]['predictions']
comp_predictions = res[comparison_desc][example_receptor]['predictions']
ax.plot(ref_predictions, comp_predictions, 'ko', color='0.5', markeredgecolor='0.3')
ax.set_yticks(xticks)
ax.set_yticklabels(xticklabels)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlabel('EVA_100 (prediction)')
ax.set_ylabel('ALL (prediction)')
utils.simple_axis(ax)
ax.text(0.6, 0.7, 'r:{:.2f}'.format(stats.pearsonr(ref_predictions, comp_predictions)[0]))

if utils.run_from_ipython():
    plt.show()