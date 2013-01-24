#!/usr/bin/env python
# encoding: utf-8
'''

    compare the predictions of different models for one glomerulus

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import sys
import os
import json
import itertools as it
from master.libs import run_lib as rl
from master.libs import utils
import numpy as np
import pylab as plt
import matplotlib.gridspec as gridspec
reload(rl)

plt.close('all')
configs = json.load(open(sys.argv[1]))
glom = sys.argv[2]
outpath = '/Users/dedan/projects/master/results/predict'
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
for config in configs.values():
    config.update({'glomerulus': glom, 'data_path': data_path})

res = {k: {'features': rl.prepare_features(config)} for k, config in configs.items()}
all_mols = [r['features'].keys() for r in res.values()]
mol_intersection = set(all_mols[0]).intersection(*all_mols[1:])

for conf_name, config in configs.items():
    print('working on model: {}'.format(conf_name))

    # load features
    print 'preparing features..'
    data, targets, molids = rl.load_data_targets(config, res[conf_name]['features'])
    if config['feature_selection']['k_best'] == 'max':
        config['feature_selection']['k_best'] = data.shape[1]

    # fit model
    print("use {} molecules for training".format(data.shape[0]))
    tmp_res = rl.run_runner(config, data, targets, get_models=True)
    method = config['methods'].keys()[0]
    to_predict = np.array([res[conf_name]['features'][molid] for molid in mol_intersection])
    res[conf_name] = tmp_res[method]['model'].predict(to_predict)
    print('model genscore: {:.2f}\n'.format(tmp_res[method]['gen_score']))


mn = configs.keys()
gs = gridspec.GridSpec(len(mn)-1, len(mn)-1)
gs.update(wspace=0.2, hspace=0.2)
for m1, m2 in it.combinations(mn, 2):
    ax = plt.subplot(gs[mn.index(m1), mn.index(m2)-1])
    ax.plot(res[m2], res[m1], 'x')
    ax.plot([0, 1], [0, 1], color='0.5')
    ax.set_xlim([-0.2, 1])
    ax.set_ylim([-0.2, 1])
    if mn.index(m1) == (mn.index(m2)-1):
        ax.set_ylabel(m1)
    if mn.index(m1) == 0:
        ax.set_title(m2)
    if not (mn.index(m1) == 0 and mn.index(m2) == 1):
        ax.set_yticks([])
    if not (mn.index(m1) == (len(mn)-2) and mn.index(m2) == (len(mn)-1)):
        ax.set_xticks([])
plt.savefig(os.path.join(outpath, glom + '_prediction_comparison.png'))

if utils.run_from_ipython():
    plt.show()

with open(os.path.join(outpath, glom + '_predictions.csv'), 'w') as f:
    f.write(',{}\n'.format(','.join(res.keys())))
    for i, molid in enumerate(mol_intersection):
        f.write(molid + ',')
        f.write(','.join([str(r[i]) for r in res.values()]))
        f.write('\n')


