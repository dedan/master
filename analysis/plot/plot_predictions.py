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

inpath = '/Users/dedan/projects/master/results/predict/'
reference = 'all'
res = pickle.load(open(os.path.join(inpath, 'predictions.pkl')))
to_compare = set(res.keys()).difference([reference])

fig = plt.figure()
xticks = np.arange(0, 1.1, 0.25)
xticklabels = ['0', '', '0.5', '', '1']
for i, desc in enumerate(to_compare):

    ax = fig.add_subplot(2, len(to_compare), i+1)
    correlations_both_pos = []
    correlations_one_neg = []
    for glom in res[reference]:
        corr = np.corrcoef(res[reference][glom]['predictions'],
                           res[desc][glom]['predictions'])[0, 1]
        if res[reference][glom]['score'] <= 0 or res[desc][glom]['score'] <=0:
            correlations_one_neg.append(corr)
        else:
            correlations_both_pos.append(corr)
    bins = np.arange(0, 1.01, 0.05)
    c_both, _ = np.histogram(correlations_both_pos, bins=bins)
    plt.bar(bins[:-1], c_both, width=bins[1]-bins[0], color='0.5')
    c_one, _ = np.histogram(correlations_one_neg, bins=bins)
    plt.bar(bins[:-1], c_one, bottom=c_both, width=bins[1]-bins[0], color='0.8')
    annotation_text = '{}/{}'.format(len(correlations_both_pos), len(res[reference]))
    ax.text(0.1, 0.9, annotation_text, transform=ax.transAxes)
    ax.set_xlim([0, 1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    if i == 0:
        ax.set_ylabel(reference)
    else:
        ax.set_yticklabels([])
    utils.simple_axis(ax)

    ax = fig.add_subplot(2, len(to_compare), len(to_compare) + i + 1)
    compare_scores = np.array([res[desc][g]['score'] for g in res[desc]])
    ref_scores = np.array([res[reference][g]['score'] for g in res[reference]])
    compare_scores[compare_scores < 0] = 0
    ref_scores[ref_scores < 0] = 0
    ax.plot(compare_scores, ref_scores, 'ko', alpha=0.6)
    ax.plot([0, 0.8], [0, 0.8], color='0.5')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(desc)
    if i == 0:
        ax.set_ylabel(reference)
    else:
        ax.set_yticklabels([])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    utils.simple_axis(ax)
