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
from scipy.stats import gaussian_kde
from master.libs import utils
from master.libs import read_data_lib as rdl

structures_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'structures')


def violin_plot(ax, pos, data, bp=False):
    '''
    create violin plots on an axis
    '''
    dist = max(pos)-min(pos)
    w = 1.1
    for d,p in zip(data,pos):
        k = gaussian_kde(d) #calculates the kernel density
        m = k.dataset.min() #lower bound of violin
        M = k.dataset.max() #upper bound of violin
        x = np.arange(m,M,(M-m)/100.) # support for violin
        v = k.evaluate(x) #violin profile (density curve)
        v = v/v.max()*w #scaling the violin to the available space
        ax.fill_betweenx(x,p,v+p,facecolor='k',alpha=0.3)
        ax.fill_betweenx(x,p,-v+p,facecolor='k',alpha=0.3)
    if bp:
        ax.boxplot(data.T,notch=1,positions=pos,vert=1)

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

def _descriptor_boxplot(ax, data, desc_names):
    """docstring for _descriptor_boxplot"""
    sort_idx = np.argsort(np.mean(data, axis=1))
    data = data[sort_idx, :]
    boxes = ax.boxplot(data.T)
    for whisk in boxes['whiskers']:
        whisk.set_linestyle('-')
    for name, thing in boxes.items():
        for line in thing:
            if not name is 'medians':
                line.set_color('0.0')
    data_copy = data.copy()
    data_copy[data_copy<0] = 0
    ax.plot(range(1, data_copy.shape[0]+1), np.mean(data_copy, axis=1), '.')
    ax.set_ylim([0, 0.8])
    ax.set_xticklabels([desc_names[i][:16] for i in sort_idx], rotation='90', fontsize=10)

def _violin_boxplot(ax, data, desc_names):
    """docstring for _descriptor_boxplot"""
    data_copy = data.copy()
    data_copy[data_copy<0] = 0
    sort_idx = np.argsort(np.mean(data_copy, axis=1))
    data_copy = data_copy[sort_idx, :]
    violin_plot(ax, np.arange(len(data_copy))*3, data_copy)
    ax.plot(np.arange(len(data_copy)) *3, np.mean(data_copy, axis=1), 'ko', label='mean', markersize=4)
    ax.plot(np.arange(len(data_copy)) *3, np.median(data_copy, axis=1), 'k*', label='median', markersize=4)
    ax.set_ylim([0, 0.8])
    ax.set_xticks(np.arange(len(data_copy)) *3 + 1)
    ax.xaxis.set_tick_params(size=0)
    ax.set_xticklabels(['_'.join(desc_names[i].split('_')[:2]).lower() for i in sort_idx],
                       rotation='45', ha='right', fontsize=10)
    ax.set_xlim([-3, len(data_copy) * 3 + 1])
    ax.legend(loc='lower left', frameon=False, numpoints=1, prop={'size': 'small'}, bbox_to_anchor=(0., 0.85))


def _descriptor_scatterplot(ax, data, clist, desc_names):
    """docstring for _descriptor_scatterplot"""
    sort_idx = np.argsort(np.mean(data, axis=1))
    data = data[sort_idx, :]
    for j, d in enumerate(data):
        for jj, dd in enumerate(d):
            c = '{:.2f}'.format(clist[jj])
            ax.plot(j+0.5, dd, 'o', color=c)
    ax.set_xticks(np.arange(len(data))+0.5)
    ax.set_ylim([-3, 0.8])
    ax.set_xticklabels([desc_names[i][:16] for i in sort_idx], rotation='90', fontsize=10)

def _descriptor_curveplot(ax, data, desc_names):
    thresholds = np.arange(0, 1, 0.05)
    for ddata, desc_name in zip(data, desc_names):
        to_plot = []
        for t in thresholds:
            to_plot.append(np.sum(ddata >= t) / float(len(ddata)))
        ax.plot(thresholds, to_plot, label=desc_name)


def new_descriptor_performance_plot(fig, max_overview, selection, method, glomeruli=[],
                                    descriptor_plot_type='scatterplot'):
    """compare performance of different descriptors for several glomeruli"""

    desc_names = max_overview[method][selection]['desc_names']
    if 'p_selection' in max_overview[method][selection] and \
       np.sum(max_overview[method][selection]['p_selection']) != 0.0:
        print('plotting param selection instead of maximum')
        data = max_overview[method][selection]['p_selection']
    else:
        data = max_overview[method][selection]['max']

    # use only selected glomeruli
    avail_glomeruli = max_overview[method][selection]['glomeruli']
    if glomeruli:
        glom_idx = [i for i, g in enumerate(avail_glomeruli) if g in glomeruli]
        data = data[:, glom_idx]

    # compute colors to color all glomeruli according to their all performance
    all_idx = max_overview[method][selection]['desc_names'].index('all')
    all_values = data[all_idx]
    n_val = float(len(all_values)-1)
    clist_all = [sorted(all_values,reverse=True).index(i) / n_val for i in all_values]

    ax = fig.add_subplot(111)
    ax.set_title('{}'.format(method))
    if descriptor_plot_type == 'boxplot':
        _descriptor_boxplot(ax, data, desc_names)
    elif descriptor_plot_type == 'scatterplot':
        _descriptor_scatterplot(ax, data, clist_all, desc_names)
    elif descriptor_plot_type == 'curveplot':
        _descriptor_curveplot(ax, data, desc_names)
        if i_meth == 1:
            ax.legend(prop={'size':5})
    elif descriptor_plot_type == 'violinplot':
        _violin_boxplot(ax, data, desc_names)
    else:
        assert False
    utils.simple_axis(ax)
    ax.set_ylabel('q2 score')

def plot_search_matrix(fig, desc_res, selection, method, glomeruli):
    """docstring for plot_search_matrix"""
    if not glomeruli:
        tmp_glomeruli = desc_res[selection].keys()
    else:
        tmp_glomeruli = glomeruli
    n = utils.ceiled_root(len(tmp_glomeruli))
    for i_glom, glom in enumerate(tmp_glomeruli):
        mat = desc_res[selection][glom][method]
        ax = fig.add_subplot(n, n, i_glom+1)
        if np.max(mat) < 0:
            ax.imshow(np.zeros(mat.shape), interpolation='nearest', cmap=plt.cm.binary)
        else:
            ax.imshow(mat, interpolation='nearest', cmap=plt.cm.binary, vmin=0)
        ax.set_ylabel('{}'.format(glom))
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel('{:.2f}'.format(np.max(mat)))
    fig.subplots_adjust(hspace=0.4)
