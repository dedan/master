#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys, os, pickle, json
import master.libs.read_data_lib as rdl
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor as rfr
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

plt.close('all')
for descriptor in features: #[features.keys()[0]]:

    fig = plt.figure()
    fig.suptitle(descriptor)
    axes = []
    max_likeli = 0
    print descriptor

    for glom_i, glom in enumerate(best_glom):

        print glom
        glom_idx = glomeruli.index(glom)

        # select molecules available for the glomerulus
        avail_cas_idx = np.where(~np.isnan(rm[:, glom_idx]))[0]

        # check whether we have molids for the CAS number and if not remove them
        stim_idx = [i for i in avail_cas_idx if door2id[cas_numbers[i]]]
        tmp_rm = rm[stim_idx, :]
        remaining_cas_numbers = [cas_numbers[i] for i in stim_idx]
        assert tmp_rm.shape[0] == len(remaining_cas_numbers)

        # now get all the available molecule ids and from them retrieve the features
        # ! the number of molecules available might vary from descriptor to descriptor
        molids = [door2id[cas_number][0] for cas_number in remaining_cas_numbers]
        data, available = rdl.get_features_for_molids(features[descriptor], molids)
        assert data.shape[0] == len(molids)
        for i in np.where(np.sum(data, axis=1) == 0)[0]:
            assert not i in available

        # use only the data for which features are available
        data = data[available, :]
        targets = [tmp_rm[i, glom_idx] for i in available]
        assert data.shape[0] == len(targets)

        _, p = f_regression(data, targets)
        scores = -np.log10(p)
        if np.max(scores) > max_likeli:
            max_likeli = np.max(scores)
        ax = fig.add_subplot(n_glomeruli, 1, glom_i + 1)
        axes.append(ax)
        x_indices = np.arange(data.shape[-1])
        ax.bar(x_indices, scores, width=.3)
        ax.set_ylabel(glom)
        ax.set_xticklabels([])

    for ax in axes:
        ax.set_ylim([0, max_likeli])
    fig.savefig(os.path.join(base_path, 'results', 'features', descriptor + '.' + format))

plt.show()

    # svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    # svr_rbf.fit(data[:-1, :], targets[:-1])

    # print svr_rbf.predict(data[-1, :]), targets[-1]