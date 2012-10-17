#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys, os, pickle, json
import master.libs.read_data_lib as rdl
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
import numpy as np
import pylab as plt
reload(rdl)

base_path = '/Users/dedan/projects/master/'
format = 'png'
n_glomeruli = 5

door2id = json.load(open(os.path.join(base_path, 'data', 'door2id.json')))
features = json.load(open(os.path.join(base_path, 'data', 'features.json')))
csv_path = os.path.join(base_path, 'data', 'response_matrix.csv')
cas_numbers, glomeruli, rm = rdl.load_response_matrix(csv_path)

# select the N glomeruli with most molecules available
best_glom = rdl.select_n_best_glomeruli(rm, glomeruli, n_glomeruli)
print best_glom

# check whether we have molids for the CAS number and if not remove them
stim_idx = [i for i in range(len(cas_numbers)) if door2id[cas_numbers[i]]]
rm = rm[stim_idx, :]
cas_numbers = [cas_numbers[i] for i in stim_idx]

# data collection
res = {}
for descriptor in [features.keys()[0]]:

    print descriptor
    res[descriptor] = {}

    for glom in best_glom:

        print glom
        res[descriptor][glom] = {}
        glom_idx = glomeruli.index(glom)

        # select molecules available for the glomerulus
        tmp_rm , tmp_cas_numbers = rdl.get_avail_for_glom(rm, cas_numbers, glom_idx)

        # get the molecule ids and retrieve all features available
        molids = [door2id[cas_number][0] for cas_number in tmp_cas_numbers]
        data, avail_features = rdl.get_features_for_molids(features[descriptor], molids)

        # use only the data for which features are available
        data = data[avail_features, :]
        targets = [tmp_rm[i, glom_idx] for i in avail_features]

        _, p = f_regression(data, targets)
        res[descriptor][glom]['regr'] = -np.log10(p)

        rfr = RandomForestRegressor(n_estimators=10, compute_importances=True)
        rfr.fit(data,targets)
        res[descriptor][glom]['rf'] = rfr.feature_importances_


# plotting
plt.close('all')
for descriptor in res:
    fig = plt.figure()
    fig.suptitle(descriptor)

    max_lin = np.max([res[descriptor][glom]['regr'] for glom in res[descriptor]])
    max_rf = np.max([res[descriptor][glom]['rf'] for glom in res[descriptor]])
    x_indices = np.arange(len(res[descriptor][glom]['rf']))

    for glom_i, glom in enumerate(res[descriptor]):

        ax = fig.add_subplot(n_glomeruli, 1, glom_i + 1)
        ax.bar(x_indices, res[descriptor][glom]['regr'] / max_lin, color='0.5')
        ax.bar(x_indices, -res[descriptor][glom]['rf'] / max_rf, color='0')

        ax.set_ylabel(glom)
        ax.set_yticks([-1, 0, 1])
        ax.set_xticklabels([])
    fig.savefig(os.path.join(base_path, 'results', 'features', descriptor + '.' + format))

plt.show()
