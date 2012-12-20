#!/usr/bin/env python
# encoding: utf-8
'''
for my talk in konstanz we decided to show a model which performs really good
although it uses only a small numbers of the dimensions of a descriptor. We got
such a result for svr_ens with forest selection for the glomerulus Or22a.


Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import glob
import sys
import os
import json
from master.libs import run_lib
from master.libs import features_lib as flib
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
reload(run_lib)
reload(flib)

# search config
config = json.load(open(sys.argv[1]))

# load the features
features = run_lib.prepare_features(config)

data, targets = run_lib.load_data_targets(config, features)
sel_scores = run_lib.get_selection_score(config, data, targets)
data = flib.select_k_best(data, sel_scores, config['feature_selection']['k_best'])
tmp_res = run_lib.run_runner(config, data, targets, get_models=True)
model = tmp_res['svr_ens']['model']

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(sel_scores)

# 3 d plot with dots colorcoded by target value
# [1 0 4] is the order of features selected
# "('WALK_PATH_COUNTS', 'MPC04')" --> molecular path count of order 04
# "('TOPOLOGICAL', 'TI2')" --> second Mohar index TI2
# "('TWOD_AUTOCORRELATIONS', 'GATS1m')" --> Geary autocorrelation - lag 1 / weighted by atomic masses

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], c=targets)
ax.set_xlabel("MPC04")
ax.set_ylabel("TI2")
ax.set_zlabel("GATS1m")


fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(targets, model.predict_oob(data), 'x')
ax.plot([0, 1], [0, 1], color='0.5')
ax.set_xlabel('unseen %.2f' % model.score_oob(data, targets))
plt.axis('scaled')
ax = fig.add_subplot(122)
ax.plot(targets, model.predict(data), 'x')
ax.plot([0, 1], [0, 1], color='0.5')
ax.set_xlabel('full training set %.2f' % model.score(data, targets))
plt.axis('scaled')
fig.savefig('/Users/dedan/projects/master/results/class_ana.png')
plt.show()





