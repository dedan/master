#!/usr/bin/env python
# encoding: utf-8
'''

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import glob
import sys
import os
import json
from master.libs import run_lib
from master.libs import features_lib as flib
from master.libs import read_data_lib as rdl
from master.libs import plot_lib as plib
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
reload(run_lib)
reload(flib)
reload(plib)

plt.close('all')
active_thresh = 0.5
descriptor = 'saito_desc'
method = 'svr_ens'
selection = 'forest'
inpath = '/Users/dedan/projects/master/results/cruncher/conv_features'
outpath = '/Users/dedan/projects/master/results/prediction'
data_path = "/Users/dedan/projects/master/data"
glomeruli = ['Or10a', 'Or42b', 'Or47b']
glom = 'Or10a'

# load stuff
config = json.load(open(sys.argv[1]))
door2id = json.load(open(os.path.join(data_path, 'door2id.json')))
daniel_set = json.load(open(os.path.join(data_path, 'daniel_set.json')))
daniel_set_molid = [door2id[cas][0] for cas in daniel_set]

# get the best parameters from the search result
search_res, max_overview, sc = rdl.read_paramsearch_results(inpath)
config = sc['runner_config_content']
config['features']['descriptor'] = descriptor
config['glomerulus'] = glom
cur_max = max_overview[method][selection]
desc_idx = cur_max['desc_names'].index(descriptor)
glom_idx = cur_max['glomeruli'].index(glom)
best_c_idx = int(cur_max['c_best'][desc_idx, glom_idx])
best_kbest_idx = int(cur_max['k_best'][desc_idx, glom_idx])
config['methods']['svr_ens']['C'] = sc['svr'][best_c_idx]
config['feature_selection']['k_best'] = sc['k_best'][best_kbest_idx]
config['feature_selection']['method'] = selection

features = run_lib.prepare_features(config)


# structure plot
data, targets, molids = run_lib.load_data_targets(config, features)
fig = plt.figure(figsize=(5,5))
active_targets = np.where(targets > active_thresh)[0]
act_molids = [molids[i] for i in active_targets]
plib.structure_plot(fig, act_molids, targets[active_targets])
fig.suptitle(glom)
fig.savefig(os.path.join(outpath, glom + '.png'))


sel_scores = run_lib.get_selection_score(config, data, targets)
data = flib.select_k_best(data, sel_scores, config['feature_selection']['k_best'])
tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
model = tmp_res[method]['model']

molids_to_predict = list(set(daniel_set_molid).difference(molids))
data_to_predict = np.array([features[m] for m in molids_to_predict if len(features[m]) != 0 ])
molids_to_predict = np.array([m for m in molids_to_predict if len(features[m]) != 0 ])
data_to_predict = flib.select_k_best(data_to_predict, sel_scores, config['feature_selection']['k_best'])
assert len(data_to_predict) == len(molids_to_predict)
predictions = model.predict(data_to_predict)

fig = plt.figure()
active_predictions = np.where(predictions > active_thresh)[0]
plib.structure_plot(fig, [molids_to_predict[i] for i in active_predictions],
                         predictions[active_predictions])

fig = plt.figure()
ax = fig.add_subplot(211)
ax.hist(targets)
ax.set_xlim([0,1])
ax = fig.add_subplot(212)
ax.hist(predictions)
ax.set_xlim([0,1])
plt.show()


