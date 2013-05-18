#!/usr/bin/env python
# encoding: utf-8
"""
library for all the feature analysis and selection stuff

all longer functions should be moved in here to make the individual scripts
more readable

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""
import os
import json
import glob
import csv
import numpy as np
import pybel
from collections import defaultdict
from master.libs import utils
try:
    from rpy2.robjects.packages import importr
    import rpy2.robjects as robjects
    from rpy2.rinterface import NARealType
except Exception, e:
    print '!!! rpy2 not installed !!!'

def get_best_params(max_overview, sc, k_best, descriptor, glom, method, selection):
    """extract the best parameters from a parameter search"""
    config = sc['runner_config_content']
    config['features']['descriptor'] = descriptor
    config['glomerulus'] = glom
    cur_max = max_overview[method][selection]
    desc_idx = cur_max['desc_names'].index(descriptor)
    glom_idx = cur_max['glomeruli'].index(glom)
    best_c_idx = int(cur_max['c_best'][desc_idx, glom_idx])
    best_kbest_idx = int(cur_max['k_best'][desc_idx, glom_idx])
    config['methods'][method]['regularization'] = sc[method][best_c_idx]
    config['feature_selection']['k_best'] = k_best[descriptor][best_kbest_idx]
    config['feature_selection']['method'] = selection
    return config

def get_id2name():
    """get molID to chemical name mapping from sdf file"""
    mol_file = os.path.join(os.path.dirname(__file__),
                            '..', 'data', 'molecules.sdf')
    molecules = pybel.readfile('sdf', mol_file)
    id2name = {mol.data['CdId']: mol.data['Name']
               for mol in molecules if 'Name' in mol.data}
    return id2name

def read_paramsearch_results(path, p_selection={}):
    """read the results from a parameter search for several descriptors"""
    # variables for results
    for psv in p_selection.values():
        assert 'k_best_idx' in psv and 'c_best_idx' in psv
    search_res = utils.recursive_defaultdict()
    initializer = lambda: {'max': np.zeros((len(f_names), len(sc['glomeruli']))),
                           'p_selection': np.zeros((len(f_names), len(sc['glomeruli']))),
                           'k_best': np.zeros((len(f_names), len(sc['glomeruli']))),
                           'c_best': np.zeros((len(f_names), len(sc['glomeruli']))),
                           'desc_names': [],
                           'glomeruli': []}
    max_overview = defaultdict(lambda: defaultdict(initializer))
    k_best = {}

    # read data from files
    f_names = glob.glob(os.path.join(path, "*.json"))
    for i_file, f_name in enumerate(f_names):

        desc = os.path.splitext(os.path.basename(f_name))[0]
        js = json.load(open(f_name))
        desc_res, sc = js['res'], js['sc']
        desc_res = utils.nested_remove_empty_values(desc_res)
        k_best[desc] = sc['k_best']

        for i_sel, selection in enumerate(desc_res):
            for i_glom, glom in enumerate(sorted(desc_res[selection])):
                methods = []
                for bla in desc_res[selection][glom]:
                    for blub in desc_res[selection][glom][bla]:
                        methods.extend(desc_res[selection][glom][bla][blub].keys())
                methods = set(methods)
                for i_meth, method in enumerate(methods):
                    cur_max = max_overview[method][selection]
                    mat = get_search_matrix(desc_res[selection][glom], method)
                    search_res[desc][selection][glom][method] = mat
                    cur_max['max'][i_file, i_glom] = np.max(mat)
                    cur_max['k_best'][i_file, i_glom] = np.argmax(np.max(mat, axis=1))
                    cur_max['c_best'][i_file, i_glom] = np.argmax(np.max(mat, axis=0))
                    if not desc in cur_max['desc_names']:
                        cur_max['desc_names'].append(desc)
                    cur_max['glomeruli'] = sorted(desc_res[selection])
                    if method in p_selection:
                        ps = p_selection[method]
                        sel_value = mat[ps['k_best_idx'], ps['c_best_idx']]
                        cur_max['p_selection'][i_file, i_glom] = sel_value
    return search_res, max_overview, sc, k_best


def get_search_matrix(res, method):
    """helper method for reading data after parameter search"""
    k_b_keys, reg_keys = [], []
    for k_b in sorted(res, key=int):
        for i in res[k_b]:
            if method in res[k_b][i]:
                if not k_b in k_b_keys:
                    k_b_keys.append(k_b)
                if not i in reg_keys:
                    reg_keys.append(i)
    mat = np.zeros((len(k_b_keys), len(reg_keys)))
    for j, k_b in enumerate(k_b_keys):
        for i in reg_keys:
            mat[j, int(i)] = res[k_b][i][method]['gen_score']
    return mat

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
