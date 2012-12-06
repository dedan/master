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
glom = 'Or42b'

# load stuff
door2id = json.load(open(os.path.join(data_path, 'door2id.json')))
daniel_set = json.load(open(os.path.join(data_path, 'daniel_set.json')))
daniel_set_molid = [door2id[cas][0] for cas in daniel_set]

# get the best parameters from the search result
config = rdl.get_best_params(inpath, descriptor, glom, method, selection)
features = run_lib.prepare_features(config)
data, targets, molids = run_lib.load_data_targets(config, features)


# fit model
print("use {} molecules for training".format(data.shape[0]))
sel_scores = run_lib.get_selection_score(config, data, targets)
data = flib.select_k_best(data, sel_scores, config['feature_selection']['k_best'])
tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
model = tmp_res[method]['model']

# compute predictions
molids_to_predict = list(set(daniel_set_molid).difference(molids))
print("want to predict on {} molecules".format(len(molids_to_predict)))
data_to_predict = np.array([features[m] for m in molids_to_predict if list(features[m])])
print("found features for {} molecules".format(data_to_predict.shape[0]))
molids_to_predict = np.array([m for m in molids_to_predict if len(features[m]) != 0 ])
data_to_predict = flib.select_k_best(data_to_predict, sel_scores,
                                     config['feature_selection']['k_best'])
assert len(data_to_predict) == len(molids_to_predict)
predictions = model.predict(data_to_predict)

# structure plot for active targets and predictions
active_targets = np.where(targets > active_thresh)[0]
act_molids = [molids[i] for i in active_targets]
active_predictions = np.where(predictions > active_thresh)[0]
act_predict_molids = [molids_to_predict[i] for i in active_predictions]
fig = plt.figure(figsize=(5,5))
plib.structure_plot(fig, (act_molids, act_predict_molids),
                         (targets[active_targets], predictions[active_predictions]))
fig.suptitle(glom)
fig.savefig(os.path.join(outpath, glom + '_structures.png'))

# targets and predictions histogram plot
fig = plt.figure()
ax = fig.add_subplot(311)
ax.plot(targets, model.predict(data), 'bx')
ax.plot([0, 1], [0, 1], color='0.5')
ax.set_xlabel('targets vs. predictions')
ax = fig.add_subplot(312)
ax.hist(targets)
ax.set_xlim([0,1])
ax.set_xlabel('targets histogram')
ax = fig.add_subplot(313)
ax.hist(predictions)
ax.set_xlim([0,1])
ax.set_xlabel('predictions histogram')
fig.subplots_adjust(hspace=0.4)
plt.show()


