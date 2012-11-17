#!/usr/bin/env python
# encoding: utf-8
"""
library for all the feature analysis and selection stuff

all longer functions should be moved in here to make the individual scripts
more readable

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import csv
import numpy as np
try:
    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    from rpy2.rinterface import NARealType
except Exception, e:
    print '!!! rpy2 not installed !!!'


def get_data_from_r(path_to_csv):
    """extract the response matrix from the R package and save it as a CSV"""
    importr('DoOR.function')
    importr('DoOR.data')
    load_data = robjects.r['loadRD']
    load_data()
    rm = robjects.r['response.matrix']
    rm.to_csvfile(path_to_csv)

def load_response_matrix(path_to_csv, door2id=None):
    """load the DoOR response matrix from the R package

        if door2id given, return only the stimuli for which we have a molID
    """
    with open(path_to_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        glomeruli = reader.next()
        cas_numbers, data = [], []
        for row in reader:
            if not row[0] in ['SFR', 'solvent']:
                cas_numbers.append(row[0])
                data.append(row[1:])
    rm = np.zeros((len(cas_numbers), len(glomeruli)))
    for i in range(len(cas_numbers)):
        for j in range(len(glomeruli)):
            rm[i, j] = float(data[i][j]) if data[i][j] != 'NA' else np.nan

    if door2id:
        # check whether we have molids for the CAS number and if not remove them
        stim_idx = [i for i in range(len(cas_numbers)) if door2id[cas_numbers[i]]]
        rm = rm[stim_idx, :]
        cas_numbers = [cas_numbers[i] for i in stim_idx]

    return cas_numbers, glomeruli, rm

def select_n_best_glomeruli(response_matrix, glomeruli, n_glomeruli):
    """select the glomeruli with most stimuli available"""
    glom_available = np.sum(~np.isnan(response_matrix), axis=0)
    glom_available_idx = np.argsort(glom_available)[::-1]
    return [glomeruli[i] for i in glom_available_idx[:n_glomeruli]]

def get_avail_targets_for_glom(rm, cas_numbers, glom_idx):
    """filter response matrix and cas numbers for availability in a glomerulus"""
    avail_cas_idx = np.where(~np.isnan(rm[:, glom_idx]))[0]
    tmp_rm = rm[avail_cas_idx, glom_idx]
    tmp_cas_numbers = [cas_numbers[i] for i in avail_cas_idx]
    return tmp_rm, tmp_cas_numbers
