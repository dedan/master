#!/usr/bin/env python
# encoding: utf-8
"""
plot the results of a randomization test

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import glob
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from master.libs import utils

res = {}
method = 'svr_ens'

inpath = '/Users/dedan/projects/master/results/validation/lin_svrens'
outpath = os.path.join(inpath, 'plots')
if not os.path.exists(outpath):
    os.mkdir(outpath)
true_res_file = os.path.join(inpath, 'true.json')
true_res = json.load(open(true_res_file))
for glom in true_res:
    mat = rdl.get_search_matrix(true_res[glom], method)
    if not glom in res:
        res[glom] = {"rand_res_dist": []}

    res[glom]['true_res'] = np.max(mat)

rand_res_files = glob.glob(os.path.join(inpath, 'run_*.json'))
for rand_res_file in rand_res_files:
    rand_res = json.load(open(rand_res_file))
    for glom in rand_res:
        mat = rdl.get_search_matrix(rand_res[glom], method)
        res[glom]['rand_res_dist'].append(np.max(mat))

fig = plt.figure(figsize=(8,4))
for i, glom in enumerate(res):

    # create a plot of the resulting distribution and the original value
    ax = fig.add_subplot(len(res), 1, i+1)
    ax.hist(res[glom]['rand_res_dist'])
    ax.plot([res[glom]['true_res']], [5], 'r*')
    ax.set_ylabel(glom, rotation='0')
    ax.set_yticks([])
    ax.set_xlim([-0.4, 0.8])
    if not i == len(res) - 1:
        ax.set_xticks([])
fig.subplots_adjust(hspace=0.3)
fig.savefig(os.path.join(outpath, 'rand_test.png'))
if utils.run_from_ipython():
    plt.show()

