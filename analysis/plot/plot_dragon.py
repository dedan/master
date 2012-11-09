#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import pickle
import numpy as np
import pylab as plt
from master.scripts import runner


config = json.load(open('/Users/dedan/projects/master/config/runner_example.json'))
features = runner.prepare_features(config)
n_features = len(features[features.keys()[0]])

glomeruli = ['Or19a', 'Or22a', 'Or35a', 'Or43b', 'Or67a']
k_best = [2**i for i in range(10) if 2**i < n_features ]
forest = [3, 5, 10, 100, 500]
svr = [0.01, 0.1, 1, 10, 100]


plt.close('all')
res = pickle.load(open('test.pckl'))

for method in ['svr', 'svr_ens', 'forest']:
    fig = plt.figure()
    fig.suptitle(method)
    for i_sel, selection in enumerate(res):
        for i_glom, glom in enumerate(res[selection]):

            ax = fig.add_subplot(len(res), len(glomeruli), i_sel * len(glomeruli) + i_glom + 1)
            mat = np.zeros((len(k_best), len(svr)))
            for j, k_b in enumerate(k_best):
                for i in range(len(forest)):
                    mat[j,i] = res[selection][glom][k_b][forest[i]][method]['gen_score']

            res[selection][glom]['mat'] = mat
            ax.imshow(mat, interpolation='nearest')
            if i_glom == 0:
                ax.set_yticks(range(len(k_best)))
                ax.set_yticklabels(k_best)
            else:
                ax.set_yticks([])

            ax.set_xticks(range(len(svr)))
            if 'svr' in method:
                ax.set_xticklabels(svr, rotation='45')
            else:
                ax.set_xticklabels(forest, rotation='45')
            ax.set_xlabel('max: %.2f' % np.max(mat))
plt.show()
