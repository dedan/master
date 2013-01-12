#!/usr/bin/env python
# encoding: utf-8
"""
I had the problem that sometimes I had the paramsearch results for lets say
SVR and RFR, but then rerun the RFR with different settings. In order to have
all results in the same plot (done with analysis/plots/plot_paramsearch.py)
I would like to have all results in one file.

usage:

    python merge_paramsearch_results.py old_results_folder folder_with_latest_results

--> the merged results can be found in the old_results_folder

!!! results already existing in the old_results_folder will be overwritten !

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import os
import sys
import glob
import json
from master.libs.utils import RUDict

old = sys.argv[1]
new = sys.argv[2]

new_files = glob.glob(os.path.join(new, '*.json'))
for new_file in new_files:

    old_res_file = os.path.join(old, os.path.basename(new_file))
    if os.path.exists(old_res_file):
        old_res = json.load(open(old_res_file))
    else:
        print '{} missing'.format(old_res_file)
        old_res = {'res': {}}
    new_res = json.load(open(new_file))

    old_res['sc_{}'.format(os.path.basename(new_file))] = new_res['sc']
    old_rud = RUDict(old_res['res'])
    old_rud.update(new_res['res'])
    old_res['res'] = dict(old_rud)

    json.dump(old_res, open(old_res_file, 'w'))

