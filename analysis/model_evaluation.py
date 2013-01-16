#!/usr/bin/env python
# encoding: utf-8
'''
evaluate one single model, one technique for one glomerulus

the paramters are either determined by the results of a parameter search or can
be set by hand.

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import sys
import os
import json
from master.libs import run_lib
from master.libs import read_data_lib as rdl
from master.libs import utils
import numpy as np
import pylab as plt
reload(run_lib)

plt.close('all')
descriptor = 'all'
method = 'svr'
selection = 'linear'
inpath = '/Users/dedan/projects/master/results/final_plots/svr_overview'
outpath = os.path.join(inpath, 'plots')
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
glom = sys.argv[1]

# get the best parameters from the search result
print 'reading results'
search_res, max_overview, sc, k_best_dict = rdl.read_paramsearch_results(inpath)
config = rdl.get_best_params(max_overview, sc, k_best_dict,
                             descriptor, glom, method, selection)

# overwrite optimal (param search results) parameters
for m in config['methods'].keys():
    if not m == method:
        del config['methods'][m]
config['methods'][method]['C'] = 1.0
del config['methods'][method]['regularization']
config['feature_selection']['k_best'] = k_best_dict[descriptor][-1]

# load features
print 'preparing features..'
features = run_lib.prepare_features(config)
data, targets, molids = run_lib.load_data_targets(config, features)

# fit model
print("use {} molecules for training".format(data.shape[0]))
tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
model = tmp_res[method]['model']


# # structure plot for active targets and predictions
# active_targets = np.where(targets > active_thresh)[0]
# act_molids = [molids[i] for i in active_targets]
# active_predictions = np.where(predictions > active_thresh)[0]
# act_predict_molids = [molids_to_predict[i] for i in active_predictions]
# fig = plt.figure(figsize=(5,5))
# plib.structure_plot(fig, (act_molids, act_predict_molids),
#                          (targets[active_targets], predictions[active_predictions]))
# fig.suptitle(glom)
# fig.savefig(os.path.join(outpath, glom + '_structures.png'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(targets, model.predict(data), 'bx')
ax.plot([0, 1], [0, 1], color='0.5')
ax.set_xlim([0,1])
ax.axis('equal')
ax.set_xlabel('targets vs. predictions (training) ({:.2f})'.format(tmp_res[method]['train_score']))
fig.savefig(os.path.join(outpath, glom + '_training_scatter.png'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(model.all_targets, model.all_predictions, 'bx')
ax.plot([0, 1], [0, 1], color='0.5')
ax.set_xlim([0,1])
ax.axis('equal')
ax.set_xlabel('targets vs. predictions (xval) ({:.2f})'.format(tmp_res[method]['gen_score']))
fig.savefig(os.path.join(outpath, glom + '_xval_scatter.png'))

# fig.subplots_adjust(hspace=0.4)
if utils.run_from_ipython():
    plt.show()

