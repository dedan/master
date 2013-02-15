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
example_receptor = 'Or22a'
comparison_desc = 'eva'
res = pickle.load(open(os.path.join(inpath, 'predictions.pkl')))
del res['getaway']
to_compare = set(res.keys()).difference([reference])

fig = plt.figure()
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
    print '{} mean r: {:.2f}'.format(desc, np.mean(pres['all']))


fig = plt.figure()
marker = ['o', 's', 'd']
xticks = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
xticklabels = ['0', '.2', '.4', '.6', '.8', '']
ax = fig.add_subplot(111)
ax.plot([0, 0.8], [0, 0.8], color='0.6')
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
            label=desc.upper() if not desc == 'eva' else 'EVA_100')
comp_score = res[comparison_desc][example_receptor]['score']
ref_score = res[reference][example_receptor]['score']
ax.plot(comp_score, ref_score, 'ko', markersize=8)
ax.annotate(example_receptor, xy=(comp_score, ref_score), xytext=(0.65, 0.4),
            arrowprops=dict(facecolor='black', shrink=0.2, width=1, frac=0.2, headwidth=4),
            )

ax.text(0.65, 0.75, 'r:{:.2f}'.format(stats.pearsonr(col_comp, col_ref)[0]))
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

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([0, 0.8], [0, 0.8], color='0.6')
ref_predictions = res[reference][example_receptor]['predictions']
comp_predictions = res[comparison_desc][example_receptor]['predictions']
ax.plot(ref_predictions, comp_predictions, 'ko', color='0.5', markeredgecolor='0.3', label=example_receptor)
ax.set_yticks(xticks)
ax.set_yticklabels(xticklabels)
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
ax.set_xlabel('EVA_100 predictions (q2: {:.2f})'.format(ref_score))
ax.set_ylabel('ALL predictions (q2: {:.2f})'.format(ref_score))
ax.legend(loc='lower right', numpoints=1, frameon=True, fancybox=True)
utils.simple_axis(ax)
ax.text(0.6, 0.7, 'r:{:.2f}'.format(stats.pearsonr(ref_predictions, comp_predictions)[0]))
print stats.pearsonr(ref_predictions, comp_predictions)

xticks = np.arange(0, 1.1, 0.25)
xticklabels = ['0', '', '.5', '', '1']
bins = np.arange(0, 1.01, 0.05)
fig = plt.figure()
ylim = np.max([np.histogram(v['all'], bins=bins)[0] for v in plot_res.values()])
for i, (desc, pres) in enumerate(plot_res.items()):
    ax = fig.add_subplot(1, 2, i+1)
    c_both, _ = np.histogram(pres['all'], bins=bins)
    plt.bar(bins[:-1], c_both, width=bins[1]-bins[0], color='0.5')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, ylim])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(desc)
    if i == 0:
        ax.set_ylabel(reference.upper())
    else:
        ax.set_yticklabels([])
    utils.simple_axis(ax)


if utils.run_from_ipython():
    plt.show()