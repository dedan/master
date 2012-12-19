#!/usr/bin/env python
# encoding: utf-8
"""
plot the results obtained by validate_gen_score

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import os
import json
import numpy as np
import pylab as plt

inpath = '/Users/dedan/projects/master/results/validation/gen_score_stability'

res = json.load(open(os.path.join(inpath, 'res.json')))

n_n_folds = len(res.values()[0].keys())
n_repetitions = len(res.values()[0].values()[0])
gridshape = (n_n_folds + 1, n_repetitions)

for glom in res:
    fig = plt.figure()
    maxes = np.zeros((n_n_folds, n_repetitions))
    for i, n_folds in enumerate(res[glom]):

        for j in range(n_repetitions):

            ax = plt.subplot2grid(gridshape, (i, j))
            mat = np.array(res[glom][n_folds][str(j)])
            ax.imshow(mat, interpolation='nearest', vmin=0)
            maxes[i, j] = np.max(mat)
            ax.set_xlabel('{:.2f}'.format(np.max(mat)))

    ax = plt.subplot2grid(gridshape, (n_n_folds, 0), colspan=n_repetitions)
    ax.boxplot(maxes.T)
    ax.set_xticklabels(res[glom].keys())
    fig.savefig(os.path.join(inpath, glom + '.png'))
plt.show()
