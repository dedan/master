#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
from master.libs import read_data_lib as rdl

inpath = '/Users/dedan/projects/master/results/new_param_search/conv_features'
descriptor = 'all'
glom = 'Or67a'
method = 'svr'
selection = 'linear'

res = rdl.get_best_params(inpath, descriptor, glom, method, selection)

bla = json.load(open(os.path.join(inpath, descriptor + '.json')))
k_best = res['feature_selection']['k_best']
reg_idx = bla['sc'][method].index(res['methods'][method]['regularization'])
print "k_best", k_best
print "reg_idx", reg_idx
best_idx = bla['res'][selection][glom][str(k_best)][str(reg_idx)][method]['best_idx']
print "len(best_idx)", len(best_idx)