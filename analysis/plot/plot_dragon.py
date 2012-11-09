#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import glob
import sys
import os
import json
import numpy as np
import pylab as plt
from master.libs import run_lib

inpath = '/Users/dedan/projects/master/results/param_search/conv_features'

plt.close('all')
for f_name in glob.glob(os.path.join(inpath, "*.json")):

    js = json.load(open(f_name))
    res = js['res']
    sc = js['sc']

    for method in ['svr', 'svr_ens', 'forest']:
        fig = plt.figure()
        fig.suptitle(method)
        for i_sel, selection in enumerate(sc['selection']):
            for i_glom, glom in enumerate(res[selection]):

                ax = fig.add_subplot(len(res), len(sc['glomeruli']), i_sel * len(sc['glomeruli']) + i_glom + 1)
                mat = np.zeros((len(sc['k_best']), len(sc['svr'])))
                for j, k_b in enumerate(sc['k_best']):
                    for i in range(len(sc['forest'])):
                        mat[j,i] = res[selection][glom][str(k_b)][str(i)][method]['gen_score']

                res[selection][glom]['mat'] = mat
                ax.imshow(mat, interpolation='nearest')
                if i_glom == 0:
                    ax.set_yticks(range(len(sc['k_best'])))
                    ax.set_yticklabels(sc['k_best'])
                else:
                    ax.set_yticks([])

                ax.set_xticks(range(len(sc['svr'])))
                if 'svr' in method:
                    ax.set_xticklabels(sc['svr'], rotation='45')
                else:
                    ax.set_xticklabels(sc['forest'], rotation='45')
                ax.set_xlabel('max: %.2f' % np.max(mat))
plt.show()
