#!/usr/bin/env python
# encoding: utf-8
"""
create a comparison plot for all methods between two different feature sets,
for example the conventional features with and without vapor pressure

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import pylab as plt
import numpy as np
from master.libs import plot_lib as plib
from master.libs import read_data_lib as rdl
from master.libs import utils
reload(plib)
reload(rdl)

config = json.load(open(sys.argv[1]))
methods = ['svr', 'svr_ens', 'forest']

# variables for results
plt.close('all')
search_res1, max_overview1, sc1 = rdl.read_paramsearch_results(config['inpath1'], methods)
search_res2, max_overview2, sc2 = rdl.read_paramsearch_results(config['inpath2'], methods)

data = {}
for i_meth, method in enumerate(max_overview1):
    for i_sel, selection in enumerate(max_overview1[method]):

        name = "{}_{}".format(method, selection)
        data1 = max_overview1[method][selection]['max']
        sort_x1 = np.argsort(np.mean(data1, axis=0))
        sort_y1 = np.argsort(np.mean(data1, axis=1))
        data1 = data1[sort_y1[::-1], :]
        data1 = np.mean(data1[:, sort_x1[::-1]], axis=1)

        data2 = max_overview2[method][selection]['max']
        sort_x2 = np.argsort(np.mean(data2, axis=0))
        sort_y2 = np.argsort(np.mean(data2, axis=1))
        data2 = data2[sort_y2[::-1], :]
        data2 = np.mean(data2[:, sort_x2[::-1]], axis=1)

        data[name] = {'data1': data1, 'data2': data2}


fig = plt.figure(figsize=(10,10))
sorter = lambda (k, v): max(v['data1'][0], v['data2'][0])
for i, (name, val) in enumerate(sorted(data.iteritems(), key=sorter)):

    ax = fig.add_subplot(6, 1, i+1)
    ax.bar(np.arange(len(search_res1.keys())) * 2 - 1,
           val['data1'],
           width=0.8, color='b')
    ax.bar(np.arange(len(search_res2.keys())) * 2,
           val['data2'],
           width=0.8, color='r')
    if i == 5:
        ax.set_xticks(np.arange(len(search_res1.keys())) * 2)
        ax.set_xticklabels([search_res1.keys()[i] for i in sort_y1], rotation='90')
    else:
        ax.set_xticks([])
    fig.subplots_adjust(hspace=0.4)
    ax.set_title(name)

plt.show()
