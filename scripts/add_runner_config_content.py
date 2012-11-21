#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import glob
import json

runner_example_path = os.path.join(os.path.dirname(__file__),
                                   '..', 'config', 'runner_example.json')
runner_example_config = json.load(open(runner_example_path))

folder = '/Users/dedan/mnt/numbercruncher/results/param_search/conv_features'

filenames = glob.glob(os.path.join(folder, '*.json'))
for filename in filenames:
    print filename
    with open(filename) as f:
        res = json.load(f)
    res['sc']['runner_config_content'] = runner_example_config
    with open(filename, 'w') as f:
        json.dump(res, f)



