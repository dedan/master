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
from master.libs import plot_lib as plib
from master.libs import read_data_lib as rdl
from collections import defaultdict
reload(utils)
reload(plib)

config = json.load(open(sys.argv[1]))
outpath = os.path.join(config['inpath'], 'plots')
methods = ['svr', 'svr_ens', 'forest']

# variables for results
plt.close('all')
search_res = utils.recursive_defaultdict()
initializer = lambda: {'max': np.zeros((len(f_names), len(sc['glomeruli']))),
                       'k_best': np.zeros((len(f_names), len(sc['glomeruli'])))}
max_overview = defaultdict(lambda: defaultdict(initializer))

# read data from files
f_names = glob.glob(os.path.join(config['inpath'], "*.json"))
for i_file, f_name in enumerate(f_names):

    desc = os.path.splitext(os.path.basename(f_name))[0]
    js = json.load(open(f_name))
    desc_res, sc = js['res'], js['sc']
    for i_sel, selection in enumerate(sc['selection']):
        for i_glom, glom in enumerate(desc_res[selection]):
            for i_meth, method in enumerate(methods):
                mat = rdl.get_search_matrix(desc_res[selection][glom], method)
                search_res[desc][selection][glom][method] = mat
                max_overview[method][selection]['max'][i_file, i_glom] = np.max(mat)
                max_overview[method][selection]['k_best'][i_file, i_glom] = np.argmax(np.max(mat, axis=1))


if config['plot_param_space']:
    for desc in search_res:
        fig = plt.figure()
        plib.plot_search_matrix(fig, search_res[desc], sc, methods)
        fig.savefig(os.path.join(outpath, desc + '.' + config['format']))


# feature selection comparison plot
fig = plt.figure()
plib.feature_selection_comparison_plot(fig, max_overview, sc)
fig.savefig(os.path.join(outpath, 'max_overview.' + config['format']))


# descriptor method performance plots
fig = plt.figure(figsize=(15,30))
plib.descriptor_performance_plot(fig, max_overview, search_res, sc)
fig.savefig(os.path.join(outpath, 'desc_compariosn.' + config['format']))
if utils.run_from_ipython():
    plt.show()





