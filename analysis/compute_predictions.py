#!/usr/bin/env python
# encoding: utf-8
'''

compute predictions to compare the different models for one glomerulus

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
'''
import sys
import os
import json
import pickle
import itertools as it
from master.libs import run_lib as rl
from master.libs import utils
import numpy as np
reload(rl)

outpath = '/Users/dedan/projects/master/results/predict'
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
gloms = json.load(open(os.path.join(data_path, 'all_glomeruli.json')))

base_config = {
    "data_path": data_path,
    "feature_selection": {
        "k_best": "max",
        "method": "linear"
        },
    "features": {
      "normalize": True,
      "properties_to_add": []
    },
    "methods": {
      "svr": {
        "C": 1.0,
        "n_folds": 50
      }
    },
    "randomization_test": False
}

feat_config = {
    "haddad": {
        "type": "conventional",
        "descriptor": "haddad_desc",
    },
    "saito": {
        "type": "conventional",
        "descriptor": "saito_desc",
    },
    "all": {
      "type": "conventional",
      "descriptor": "all",
    },
    "eva": {
      "type": "spectral",
      "kernel_width": 100,
      "bin_width": 150,
      "use_intensity": False,
      "spec_type": "ir",
    }
}
method = base_config['methods'].keys()[0]


# compute molecules available for all descriptors
cache = {}
for k, config in feat_config.items():
    base_config['features'].update(config)
    cache[k] = {"features": rl.prepare_features(base_config)}
all_mols = [r['features'].keys() for r in cache.values()]
mol_intersection = set(all_mols[0]).intersection(*all_mols[1:])

res = {g: {n: {} for n in feat_config} for g in gloms}
for glom in gloms:
    print('{}\n'.format(glom))
    base_config.update({'glomerulus': glom, 'data_path': data_path})

    dtm = {}
    for name, config in feat_config.items():
        base_config['features'].update(config)
        data, targets, molids = rl.load_data_targets(base_config, cache[name]['features'])
        dtm[name] = {
            'data': data,
            'targets': targets,
            'molids': molids
        }

    # select molecules that none of the models will be trained on
    all_trained = set(dtm.values()[0]['molids']).union(*[m['molids'] for m in dtm.values()[1:]])
    to_predict_molids = mol_intersection - all_trained

    for name, data in dtm.items():

        # fit model
        print('working on model: {}'.format(name))
        base_config['feature_selection']['k_best'] = data['data'].shape[1]
        print("use {} molecules for training".format(data['data'].shape[0]))
        tmp_res = rl.run_runner(base_config, data['data'], data['targets'], get_models=True)
        to_predict = np.array([cache[name]['features'][molid] for molid in to_predict_molids])
        res[glom][name]['predictions'] = tmp_res[method]['model'].predict(to_predict)
        res[glom][name]['targets'] = data['targets']
        res[glom][name]['score'] = tmp_res[method]['gen_score']
        print('model genscore: {:.2f}\n'.format(tmp_res[method]['gen_score']))

pickle.dump(dict(res), open(os.path.join(outpath, 'predictions.pkl'), 'w'))
