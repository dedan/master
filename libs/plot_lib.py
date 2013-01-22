#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import numpy as np
import pylab as plt
from collections import defaultdict
from scipy.stats.stats import nanmean
from master.libs import utils
from master.libs import read_data_lib as rdl

structures_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'structures')


def structure_plot(fig, molids, activations=None):
    """plot molecule structures"""
    if activations != None:
        assert len(activations[0]) == len(molids[0])
        assert len(activations[1]) == len(molids[1])
    id2name = defaultdict(str, rdl.get_id2name())
    all_molids = molids[0] + molids[1]
    all_activations = np.hstack(activations)
    cr = utils.ceiled_root(len(all_molids))
    for i, molid in enumerate(all_molids):
        ax = fig.add_subplot(cr, cr, i+1)
        try:
            img = plt.imread(os.path.join(structures_path, molid + '.png'))
        except Exception, e:
            img = np.zeros(img.shape)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(molids[0]):
            for child in ax.get_children():
                if isinstance(child, plt.matplotlib.spines.Spine):
                    child.set_color('#ff0000')
        if activations != None:
            ax.set_title('{}: {:.2f}'.format(id2name[molid], all_activations[i]),
                         rotation='20')
        # plt.axis('off')


def new_descriptor_performance_plot(fig, max_overview, sc, boxplot=True):
    """compare performance of different descriptors for several glomeruli"""
    n_plots = len(max_overview) * len(max_overview.values()[0])
    for i_meth, method in enumerate(max_overview):

        for i_sel, selection in enumerate(max_overview[method]):

            desc_names = max_overview[method][selection]['desc_names']
            print np.sum(max_overview[method][selection]['p_selection'])
            if 'p_selection' in max_overview[method][selection] and \
               np.sum(max_overview[method][selection]['p_selection']) != 0.0:
                print('plotting param selection instead of maximum')
                data = max_overview[method][selection]['p_selection']
            else:
                data = max_overview[method][selection]['max']
            sort_x = np.argsort(np.mean(data, axis=0))
            sort_y = np.argsort(np.mean(data, axis=1))
            data = data[sort_y, :]
            data = data[:, sort_x]

            plot_x = i_sel * len(max_overview) + i_meth + 1
            ax = fig.add_subplot(1, n_plots, plot_x)
            ax.set_title('{}_{}.'.format(method, selection))
            if boxplot:
                boxes = ax.boxplot(data.T)
                for whisk in boxes['whiskers']:
                    whisk.set_linestyle('-')
                for name, thing in boxes.items():
                    for line in thing:
                        if not name is 'medians':
                            line.set_color('0.0')
            else:
                ax.bar((np.arange(len(desc_names)) - 0.5) * 3,
                       np.mean(data, axis=1),
                       yerr=np.std(data, axis=1), width=1.7)
                ax.set_xticks(np.arange(len(desc_names)) * 3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.set_ylim([0, 0.8])
            ax.set_xticklabels([desc_names[i] for i in sort_y], rotation='90', fontsize=6)
            if plot_x == 1:
                ax.set_ylabel('average descriptor score')


def descriptor_performance_plot(fig, max_overview, sc):
    """compare performance of different descriptors for several glomeruli"""
    for i_meth, method in enumerate(max_overview):

        for i_sel, selection in enumerate(max_overview[method]):

            desc_names = max_overview[method][selection]['desc_names']
            data = max_overview[method][selection]['max']
            sort_x = np.argsort(np.mean(data, axis=0))
            sort_y = np.argsort(np.mean(data, axis=1))
            data = data[sort_y, :]
            data = data[:, sort_x]

            plot_x = (i_sel * len(max_overview) + i_meth + 1) * 3 - 2
            ax1 = fig.add_subplot(6, 3, plot_x)
            ax1.imshow(data, interpolation='nearest')
            ax1.set_xticks(range(len(sc['glomeruli'])))
            ax1.set_xticklabels([sc['glomeruli'][i] for i in sort_x], rotation='45')
            ax1.set_aspect('auto')
            ax1.set_title('{}_{}.'.format(method, selection))

            ax = fig.add_subplot(6, 3, plot_x + 1, sharey=ax1)
            ax.barh(np.arange(len(desc_names)) - 0.5,
                   np.mean(data, axis=1),
                   xerr=np.std(data, axis=1), height=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax1.set_yticks(range(len(desc_names)))
            ax1.set_yticklabels([desc_names[i] for i in sort_y])
            ax.set_xlabel('average descriptor score')
            plt.setp(ax.get_yticklabels(), visible=False)

            ax = fig.add_subplot(6, 3, plot_x + 2)
            bins = np.arange(0, 1, 0.05)
            ax.hist(data.flatten(), bins=bins, color='b')
            ax.set_xlim([0, 1])
            ax.set_xlabel('overall method score histogram')
            plt.hold(True)
            ax.hist(data[-3:,:].flatten(), bins=bins, color='r')
            ax.set_xlim([0, 1])
            ax.set_xlabel('overall method score histogram')
            fig.subplots_adjust(hspace=0.35, wspace=0.02)

def plot_search_matrix(fig, desc_res, sc):
    """docstring for plot_search_matrix"""
    methods = desc_res.values()[0].values()[0].keys()
    for i_sel, selection in enumerate(sc['selection']):
        for i_glom, glom in enumerate(desc_res[selection]):
            for i_meth, method in enumerate(methods):
                mat = desc_res[selection][glom][method]
                ax_idx = i_meth * len(sc['glomeruli']) * 2 + len(sc['glomeruli']) * i_sel + i_glom + 1
                ax = fig.add_subplot(6, len(sc['glomeruli']), ax_idx)
                if np.max(mat) < 0:
                    ax.imshow(mat, interpolation='nearest')
                else:
                    ax.imshow(mat, interpolation='nearest', vmin=0)
                if i_sel + i_meth == 0:
                    ax.set_title(glom, rotation='0')
                if i_glom == 0:
                    ax.set_ylabel('{}\n{}'.format(method,selection))
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_xlabel('{:.2f}'.format(np.max(mat)))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
