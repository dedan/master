#!/usr/bin/env python
# encoding: utf-8
"""
    plot: regularization on x axis, number of k_best features on y

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
outpath = os.path.join(inpath, 'plots')
format = 'png'

max_overview = {}

plt.close('all')
f_names = glob.glob(os.path.join(inpath, "*.json"))
for i_file, f_name in enumerate(f_names):

    js = json.load(open(f_name))
    res = js['res']
    sc = js['sc']

    for method in ['svr', 'svr_ens', 'forest']:
        fig = plt.figure()
        fig.suptitle('model: {0}'.format(method))
        if not method in max_overview:
            max_overview[method] = {k: np.zeros((len(f_names), len(sc['glomeruli'])))
                                    for k in sc['selection']}

        for i_sel, selection in enumerate(sc['selection']):
            for i_glom, glom in enumerate(res[selection]):

                mat = np.zeros((len(sc['k_best']), len(sc['svr'])))
                for j, k_b in enumerate(sc['k_best']):
                    for i in range(len(sc['forest'])):
                        mat[j,i] = res[selection][glom][str(k_b)][str(i)][method]['gen_score']
                res[selection][glom]['mat'] = mat
                max_overview[method][selection][i_file, i_glom] = np.max(mat)

                ax = fig.add_subplot(len(res), len(sc['glomeruli']), i_sel * len(sc['glomeruli']) + i_glom + 1)
                ax.imshow(mat, interpolation='nearest')
                if i_sel == 0:
                    ax.set_title(glom)
                if i_glom == 0:
                    ax.set_yticks(range(len(sc['k_best'])))
                    ax.set_yticklabels(sc['k_best'])
                    ax.set_ylabel(selection)
                else:
                    ax.set_yticks([])

                ax.set_xticks(range(len(sc['svr'])))
                if 'linear' in selection:
                    ax.set_xticklabels(sc['svr'], rotation='45')
                else:
                    ax.set_xticklabels(sc['forest'], rotation='45')
                ax.set_xlabel('max: %.2f' % np.max(mat))
        desc_name = os.path.splitext(os.path.basename(f_name))[0]
        fig.savefig(os.path.join(outpath, desc_name + '_' + method + '.' + format))
        plt.show()

fig = plt.figure()
desc_names = [os.path.splitext(os.path.basename(f_name))[0].lower() for f_name in f_names]
glom_names = [glom for glom in res[res.keys()[0]]]
for i_meth, method in enumerate(max_overview):

    for i_sel, selection in enumerate(max_overview[method]):

        ax = fig.add_subplot(4, len(max_overview), (i_sel * 6) + i_meth + 1)
        ax.imshow(max_overview[method][selection][:], interpolation='nearest')
        if i_meth == 0:
            ax.set_yticks(range(len(desc_names)))
            ax.set_yticklabels(desc_names)
        else:
            ax.set_yticks([])

        ax = fig.add_subplot(4, len(max_overview), (i_sel * 6) + i_meth + 4)
        ax.hist(max_overview[method][selection].flatten())
        ax.set_xlim([0, 1])

fig.savefig(os.path.join(outpath, 'max_overview.' + format))





