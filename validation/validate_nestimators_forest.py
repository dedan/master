#!/usr/bin/env python
# encoding: utf-8
"""
    validate that my generalization score for the single svr is stable

    This has to be done because repeating the randomization test made
    us doubt in the reliability of the generalization score I compute.
    (See the google document)


Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import numpy as np
import pylab as plt
from master.libs import read_data_lib as rdl
from master.libs import run_lib
from master.libs import utils
reload(run_lib)

plt.close('all')
n_estimators_list = [5, 10, 20, 50, 100, 200, 500]
n_repetitions = 10
method = 'forest'
out_path = '/Users/dedan/projects/master/results/validation/n_estimators_forest'

base_path = os.path.join(os.path.dirname(__file__), '..')
config = json.load(open(os.path.join(base_path, 'config', 'validate_nestimators_forest.json')))
config['data_path'] = os.path.join(base_path, 'data')

# load the features
features = run_lib.prepare_features(config)

used_glomeruli = json.load(open(os.path.join(config['data_path'], 'used_glomeruli.json')))
res = {g: {ne: [] for ne in n_estimators_list} for g in used_glomeruli}
for glom in used_glomeruli:

    print(glom)
    config['glomerulus'] = glom
    data, targets, molids = run_lib.load_data_targets(config, features)

    for i, n_estimators in enumerate(n_estimators_list):
        print(n_estimators)
        config['methods'][method]['n_estimators'] = n_estimators
        for j in range(n_repetitions):
            run_res = run_lib.run_runner(config, data, targets)
            res[glom][n_estimators].append(run_res[method]['gen_score'])
json.dump(res, open(os.path.join(out_path, 'res.json'), 'w'))