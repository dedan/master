#!/usr/bin/env python
# encoding: utf-8
"""
create predictions for all molecules used

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import json
import subprocess

res_file = '/Users/dedan/projects/master/results/param_search/conv_features/saito_desc.json'
predict_script = '/Users/dedan/projects/master/analysis/predict.py'

param_search_res = json.load(open(res_file))

for glom in param_search_res['res']['linear']:
    subprocess.call(['python', predict_script, glom])

