#!/usr/bin/env python
# encoding: utf-8
'''

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import glob
import os
import json
from master.libs import run_lib
import numpy as np
reload(run_lib)

outpath = '/Users/dedan/projects/master/results/param_search/conv_features'
config = json.load(open('/Users/dedan/projects/master/config/runner_example.json'))

sc = {
    'selection': ['linear', 'forest'],
    'glomeruli': ['Or19a', 'Or22a', 'Or35a', 'Or43b', 'Or67a'],
    'forest': [3, 5, 10, 100, 500],
    'svr': [0.01, 0.1, 1, 10, 100]
}

res = {}
files = glob.glob('/Users/dedan/projects/master/data/conventional_features/*.csv')
for f in files:
    desc = os.path.splitext(os.path.basename(f))[0]
    if os.path.exists(os.path.join(outpath, desc + '.json')):
        print 'skip {}, already exists'.format(desc)
        continue

    config['features']['descriptor'] = desc
    config['features']['type'] = 'conventional'

    # load the features
    features = run_lib.prepare_features(config)
    n_features = len(features[features.keys()[0]])
    sc['k_best'] = [2**i for i in range(10) if 2**i < n_features]

    print 'working on: ', desc

    for selection in sc['selection']:
        print selection
        res[selection] = {}
        config['feature_selection']['method'] = selection
        for glomerulus in sc['glomeruli']:
            res[selection][glomerulus] = {}
            config['glomerulus'] = glomerulus
            for k_b in sc['k_best']:
                res[selection][glomerulus][k_b] = {}
                config['feature_selection']['k_best'] = k_b
                for i in range(len(sc['forest'])):
                    config['methods']['svr']['C'] = sc['svr'][i]
                    config['methods']['svr_ens']['C'] = sc['svr'][i]
                    config['methods']['forest']['max_depth'] = sc['forest'][i]
                    tmp_res = run_lib.run_runner(config, features)
                    tmp_res['n_features'] = n_features
                    res[selection][glomerulus][k_b][i] = tmp_res
    json.dump({'sc': sc, 'res': res}, open(os.path.join(outpath, desc + '.json'), 'w'))
