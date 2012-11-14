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
from master.libs import utils

inpath = '/Users/dedan/projects/master/results/param_search/conv_features'
outpath = os.path.join(inpath, 'plots')
format = 'png'

max_overview = {}

plt.close('all')
f_names = glob.glob(os.path.join(inpath, "*.json"))
for i_file, f_name in enumerate(f_names):

    print f_name
    js = json.load(open(f_name))
    res = js['res']
    sc = js['sc']

    for method in ['svr', 'svr_ens', 'forest']:
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('model: {0}'.format(method))
        if not method in max_overview:
            max_overview[method] = {}
            for selection in sc['selection']:
                max_overview[method][selection] = {'max': np.zeros((len(f_names), len(sc['glomeruli']))),
                                                   'k_best': np.zeros((len(f_names), len(sc['glomeruli'])))}

        for i_sel, selection in enumerate(sc['selection']):
            for i_glom, glom in enumerate(res[selection]):

                mat = np.zeros((len(sc['k_best']), len(sc['svr'])))
                for j, k_b in enumerate(sc['k_best']):
                    for i in range(len(sc['forest'])):
                        mat[j,i] = res[selection][glom][str(k_b)][str(i)][method]['gen_score']
                res[selection][glom]['mat'] = mat
                max_overview[method][selection]['max'][i_file, i_glom] = np.max(mat)
                max_overview[method][selection]['k_best'][i_file, i_glom] = np.argmax(np.max(mat, axis=1))

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


# feature selection comparison plot
fig = plt.figure()
for i_meth, method in enumerate(max_overview):
    ax = fig.add_subplot(2, len(max_overview), i_meth + 1)
    flat_lin = max_overview[method]['linear']['max'].flatten()
    flat_for = max_overview[method]['forest']['max'].flatten()
    counts_lin, _ = np.histogram(flat_lin, bins=len(sc['k_best']))
    counts_for, _ = np.histogram(flat_for, bins=len(sc['k_best']))

    # scatter plot of max values
    ax.plot(flat_lin, flat_for, 'x')
    plt.axis('scaled')
    ax.set_xticks([0, ax.get_xticks()[-1]])
    ax.set_yticks([0, ax.get_yticks()[-1]])
    ax.set_title(method)
    ax.set_xlabel('linear')
    if i_meth == 0:
        ax.set_ylabel('forest')
    ax.plot([0, 1], [0, 1], '-', color='0.6')

    # k_best histogram plot
    ax = fig.add_subplot(2, len(max_overview), i_meth + 4)
    ax.bar(range(len(sc['k_best'])), counts_lin, color='r', label='linear')
    plt.hold(True)
    ax.bar(range(len(sc['k_best'])), -counts_for, color='b', label='forest')
    ax.set_xticks(np.arange(len(sc['k_best'])) + .5)
    ax.set_xticklabels(sc['k_best'], rotation='90', ha='left')
    fig.subplots_adjust(hspace=0.4)
fig.savefig(os.path.join(outpath, 'max_overview.' + format))



desc_names = [os.path.splitext(os.path.basename(f_name))[0].lower() for f_name in f_names]
glom_names = [glom for glom in res[res.keys()[0]]]
for i_meth, method in enumerate(max_overview):

    for i_sel, selection in enumerate(max_overview[method]):

        filename = 'max_overview_{}_{}.'.format(method, selection)
        fig = plt.figure()
        fig.suptitle(filename)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(max_overview[method][selection]['max'], interpolation='nearest')
        ax1.set_xticks(range(len(glom_names)))
        ax1.set_xticklabels(glom_names, rotation='45')
        ax1.set_aspect('auto')

        ax = fig.add_subplot(2, 2, 2, sharey=ax1)
        ax.barh(range(len(desc_names)),
               np.mean(max_overview[method][selection]['max'], axis=1),
               xerr=np.std(max_overview[method][selection]['max'], axis=1),
               height=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax1.set_yticks(range(len(desc_names)))
        ax1.set_yticklabels(desc_names)
        plt.setp(ax.get_yticklabels(), visible=False)

        ax = fig.add_subplot(2, 2, 3)
        ax.hist(max_overview[method][selection]['max'].flatten())
        ax.set_xlim([0, 1])
        fig.subplots_adjust(hspace=0.35, wspace=0.02)

        fig.savefig(os.path.join(outpath, filename + format))
if utils.run_from_ipython():
    plt.show()





