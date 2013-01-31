#!/usr/bin/env python
# encoding: utf-8
"""
    plot: regularization on x axis, number of k_best features on y

    what have glomeruli for which a good model can be found in common?

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import sys
import os
import json
import pylab as plt
import itertools as it
import numpy as np
import pylab as plt
from master.libs import plot_lib as plib
from master.libs import read_data_lib as rdl
from master.libs import utils
import matplotlib.gridspec as gridspec
from scipy.stats import scoreatpercentile
reload(plib)
reload(rdl)

desc = 'all'
selection = 'linear'
method = 'svr'
config = {
    "inpath": "/Users/dedan/projects/master/results/param_search/all_gloms_svrlin_all",
    "data_path": os.path.join(os.path.dirname(__file__), '..', '..', 'data'),
    "format": "png",
}
outpath = os.path.join(config['inpath'], 'plots')
door2id = json.load(open(os.path.join(config['data_path'], 'door2id.json')))
path_to_csv = os.path.join(config['data_path'], 'response_matrix.csv')
cas_numbers, all_glomeruli, rm = rdl.load_response_matrix(path_to_csv, door2id)

# variables for results
plt.close('all')
search_res, max_overview, sc, _ = rdl.read_paramsearch_results(config['inpath'])
glomeruli = search_res[desc][selection].keys()

# sort glomeruli according to performance
maxes = [np.max(search_res[desc][selection][glom][method]) for glom in glomeruli]
picks = [search_res[desc][selection][glom][method][-1, 1] for glom in glomeruli]
max_idx = np.argsort(maxes)
glomeruli = [glomeruli[i] for i in max_idx]

fig = plt.figure(figsize=(3, 20))
for i_glom, glom in enumerate(glomeruli):
    mat = search_res[desc][selection][glom][method]
    glom_idx = all_glomeruli.index(glom)
    tmp_rm, tmp_cas_numbers = rdl.get_avail_targets_for_glom(rm, cas_numbers, glom_idx)
    ax = plt.subplot2grid((len(glomeruli), 2), (i_glom, 1))
    if len(tmp_rm) > 50 and scoreatpercentile(tmp_rm, 75) > 0.2:
        ax.hist(tmp_rm, color='g')
    elif scoreatpercentile(tmp_rm, 75) > 0.14:
        ax.hist(tmp_rm, color='#6be463')
    else:
        ax.hist(tmp_rm, color='r')
    ax.set_xlim([0, 1])
    ax.set_yticks([])
    ax.set_xticks([])

    ax = plt.subplot2grid((len(glomeruli), 2), (i_glom, 0))
    if np.max(mat) < 0:
        ax.imshow(mat, interpolation='nearest')
    else:
        ax.imshow(mat, interpolation='nearest', vmin=0)
    ax.set_yticks([])
    ax.set_xticks([])
    y_label = 'score: {:.2f}, n_targets: {}, percentile: {:.2f}'.format(
                  np.max(mat),
                  len(tmp_rm),
                  scoreatpercentile(tmp_rm, 75)
               )
    ax.set_ylabel(y_label, rotation='0')

fig.savefig(os.path.join(outpath, desc + '_allglom.' + config['format']))

fig = plt.figure()
ax = fig.add_subplot(211)
ax.hist(maxes, bins=np.arange(0, 1, 0.05))
ax.set_xlim([0, 1])
ax = fig.add_subplot(212)
ax.hist(picks, bins=np.arange(0, 1, 0.05))
ax.set_xlim([0, 1])
