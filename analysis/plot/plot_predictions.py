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

inpath = '/Users/dedan/projects/master/results/predict/'
reference = 'all'
res = pickle.load(open(os.path.join(inpath, 'predictions1.pkl')))
to_compare = set(res.keys()).difference([reference])

fig = plt.figure()
for i, desc in enumerate(to_compare):

    ax = fig.add_subplot(2, len(to_compare), i+1)
    correlations = []
    for glom in res[reference]:
        corr = np.corrcoef(res[reference][glom]['predictions'],
                           res[desc][glom]['predictions'])[0, 1]
        correlations.append(corr)
    ax.hist(correlations, facecolor='0.5')
    ax.set_xlim([0, 1])
    ax.set_yticks([])
    if i == 0:
        ax.set_ylabel(reference)

    ax = fig.add_subplot(2, len(to_compare), len(to_compare) + i + 1)
    compare_scores = [res[desc][g]['score'] for g in res[desc]]
    ref_scores = [res[reference][g]['score'] for g in res[reference]]
    ax.plot(compare_scores, ref_scores, 'ko', alpha=0.6)
    ax.plot([0, 1], [0, 1], color='0.5')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel(desc)

#     mn = configs.keys()
#     gs = gridspec.GridSpec(len(mn)-1, len(mn)-1)
#     gs.update(wspace=0.2, hspace=0.2)
#     for m1, m2 in it.combinations(mn, 2):
#         ax = plt.subplot(gs[mn.index(m1), mn.index(m2)-1])
#         ax.plot(res[m2], res[m1], 'x', color='#53777A')
#         correlations[glom].append(np.corrcoef(res[m2], res[m1])[0, 1])
#         ax.plot([0, 1], [0, 1], color='0.5')
#         ax.set_xlim([-0.2, 1])
#         ax.set_ylim([-0.2, 1])
#         if mn.index(m1) == (mn.index(m2)-1):
#             ax.set_ylabel(m1)
#         if mn.index(m1) == 0:
#             ax.set_title(m2)
#         if not (mn.index(m1) == 0 and mn.index(m2) == 1):
#             ax.set_yticks([])
#         if not (mn.index(m1) == (len(mn)-2) and mn.index(m2) == (len(mn)-1)):
#             ax.set_xticks([])
#     plt.savefig(os.path.join(outpath, glom + '_prediction_comparison.png'))

#     if utils.run_from_ipython():
#         plt.show()

#     with open(os.path.join(outpath, glom + '_predictions.csv'), 'w') as f:
#         f.write(',{}\n'.format(','.join(res.keys())))
#         for i, molid in enumerate(mol_intersection):
#             f.write(molid + ',')
#             f.write(','.join([str(r[i]) for r in res.values()]))
#             f.write('\n')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(list(it.chain(*correlations.values())))
# fig.savefig(os.path.join(outpath, 'prediction_correlations.png'))
# print stats.percentileofscore(all_cors, 0.7)