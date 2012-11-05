#!/usr/bin/env python
# encoding: utf-8
"""

    visualize results created by the runner


Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

inpath = '/Users/dedan/projects/master/results/models/'


import sys
import os
import glob
import pickle
from collections import defaultdict
import numpy as np
import pylab as plt

plt.close('all')
result_files = glob.glob(inpath + '*.pckl')
for result_file in result_files:

    res = pickle.load(open(result_file))

    plotd = defaultdict(list)
    for glom in res:
        for model in res[glom]['models']:
            plotd[model['feature_threshold']].append(model['score'])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(plotd.values())
    ax.set_xticklabels(plotd.keys())

plt.show()













