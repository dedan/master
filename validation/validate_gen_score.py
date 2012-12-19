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

plt.close('all')
n_folds_list = [10, 20, 50, 200, 500]
n_repetitions = 5
res = utils.recursive_defaultdict()
sc = json.load(open(sys.argv[1]))
config = sc['runner_config_content']
config['data_path'] = os.path.join(os.path.dirname(__file__), '..', 'data')
assert len(config['methods']) == 1  # only one method a time

# load the features
features = run_lib.prepare_features(config)
n_features = len(features[features.keys()[0]])
max_expo = int(np.floor(np.log2(n_features)))
sc['k_best'] = [2**i for i in range(max_expo)] + [n_features]

for glom in sc['glomeruli']:

    print(glom)
    config['glomerulus'] = glom

    for i, n_folds in enumerate(n_folds_list):
        print(n_folds)
        config['methods']['svr']['n_folds'] = n_folds
        for j in range(n_repetitions):
            tmp_res = run_lib.do_paramsearch(sc, config, features)
            mat = rdl.get_search_matrix(tmp_res, 'svr')
            res[glom][n_folds][j] = mat.tolist()
    json.dump(res, open(os.path.join(sc['outpath'], 'res.json'), 'w'))
