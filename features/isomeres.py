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
import pylab as plt
import numpy as np
from collections import defaultdict
import __builtin__
from scipy.spatial.distance import pdist
reload(rdl)

base_path = '/Users/dedan/projects/master/'
out_path = os.path.join(base_path, 'results', 'isomeres')
data_path = os.path.join(base_path, 'data')
N_BINS = 100

# mapping from the CAS number to jan's mol ID
door2id = json.load(open(os.path.join(data_path, 'door2id.json')))
features = json.load(open(os.path.join(data_path, 'features.json')))

# identify molecules with and without isomeres
molecules = [molids[0] for molids in door2id.values() if len(molids) == 1]
isomeres = [molids for molids in door2id.values() if len(molids) > 1]

# create similarity histogram plots for all descriptors
for f_space in features:

    fig = plt.figure()
    print 'working on: ', f_space
    ax = fig.add_subplot(111)
    mol_fspace = rdl.get_features_for_molids(features[f_space], molecules)
    assert not (mol_fspace == -999).any()
    mol_distances = pdist(mol_fspace)
    bins = np.linspace(0, max(mol_distances), num=N_BINS)
    ax.hist(mol_distances, bins=bins, log=True)

    isomere_distances = np.array([])
    for isomere in isomeres:
        iso_fspace = rdl.get_features_for_molids(features[f_space], isomere)
        assert not (iso_fspace == -999).any()
        if iso_fspace.any():
            isomere_distances = np.hstack([isomere_distances, pdist(iso_fspace)])

    _, _, patches = ax.hist(isomere_distances, bins=bins, log=True)
    for patch in patches:
        plt.setp(patch, color="r", edgecolor='k')
    fig.savefig(os.path.join(out_path, f_space + '_histo.png'))
plt.close('all')
