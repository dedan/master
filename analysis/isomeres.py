#!/usr/bin/env python
# encoding: utf-8
"""
measure for the different features the average similarity vs. stereoisomere similarity

We did this to see which descriptor is susceptible to 3D molecule configurations.
The problem is that in the experiments of the DoOr database they often used
mixtures of chemicals

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import json, os, csv, glob, json
import master.libs.read_data_lib as rdl
import master.libs.features_lib as flib
from master.libs import run_lib
import pylab as plt
import numpy as np
from collections import defaultdict
import __builtin__
from scipy.spatial.distance import pdist
reload(rdl)

config = {
         "features":
            {
              "type": "conventional"
            , "normalize": True
            , "normalize_samples": False
            , "properties_to_add": []
            },
         "data_path": '/Users/dedan/projects/master/data'
         }
base_path = '/Users/dedan/projects/master/'
out_path = os.path.join(base_path, 'results', 'isomeres')
data_path = os.path.join(base_path, 'data', 'conventional_features')
N_BINS = 100

# mapping from the CAS number to jan's mol ID
door2id = json.load(open(os.path.join(data_path, '..', 'door2id.json')))

# identify molecules with and without isomeres
molecules = [molids[0] for molids in door2id.values() if len(molids) == 1]
isomeres = [molids for molids in door2id.values() if len(molids) > 1]

# create similarity histogram plots for all descriptors
feature_files = glob.glob(os.path.join(data_path, '*.csv'))
for f in feature_files:

    desc = os.path.splitext(os.path.basename(f))[0]
    config['features']['descriptor'] = desc
    features = run_lib.prepare_features(config)

    fig = plt.figure()
    print 'working on: ', desc
    ax = fig.add_subplot(111)
    mol_fspace = np.array([features[mol] for mol in molecules if len(features[mol]) > 0])
    assert not (mol_fspace == -999).any()
    mol_distances = pdist(mol_fspace)
    bins = np.linspace(0, max(mol_distances), num=N_BINS)
    ax.hist(mol_distances, bins=bins, log=True)

    isomere_distances = np.array([])
    for isomere in isomeres:
        iso_fspace = np.array([features[iso] for iso in isomere])
        assert not (iso_fspace == -999).any()
        if iso_fspace.any():
            isomere_distances = np.hstack([isomere_distances, pdist(iso_fspace)])

    _, _, patches = ax.hist(isomere_distances, bins=bins, log=True)
    for patch in patches:
        plt.setp(patch, color="r", edgecolor='k')
    fig.savefig(os.path.join(out_path, desc + '_histo.png'))
plt.close('all')
