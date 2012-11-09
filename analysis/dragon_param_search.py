#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import json
import pickle
from master.scripts import runner
import numpy as np
import pylab as plt
reload(runner)
import glob
import os


config = json.load(open('/Users/dedan/projects/master/config/runner_example.json'))
features = runner.prepare_features(config)
n_features = len(features[features.keys()[0]])

glomeruli = ['Or19a', 'Or22a', 'Or35a', 'Or43b', 'Or67a']
k_best = [2**i for i in range(10) if 2**i < n_features ]
forest = [3, 5, 10, 100, 500]
svr = [0.01, 0.1, 1, 10, 100]

res = {}
files = glob.glob('/Users/dedan/projects/master/data/conventional_features/*.csv')
for f in files:
    desc = os.path.splitext(os.path.basename(f))[0]
    config['features']['descriptor'] = desc
    print 'working on: ', desc

    for selection in ["linear", "forest"]:
        print selection
        res[selection] = {}
        config['feature_selection']['method'] = selection
        for glomerulus in glomeruli:
            res[selection][glomerulus] = {}
            config['glomerulus'] = glomerulus
            for k_b in k_best:
                res[selection][glomerulus][k_b] = {}
                config['feature_selection']['k_best'] = k_b
                for i in range(len(forest)):
                    config['methods']['svr']['C'] = svr[i]
                    config['methods']['svr_ens']['C'] = svr[i]
                    config['methods']['forest']['max_depth'] = forest[i]
                    tmp_res = runner.run_runner(config, features)
                    res[selection][glomerulus][k_b][forest[i]] = tmp_res
    pickle.dump(res, open(desc + '.pckl', 'w'))







