#!/usr/bin/env python
# encoding: utf-8
"""
    I moved this from run_gamess and run_gaussian to a separate files
    because opebenbabel (pybel) is not installed on all machines and also
    not that easy to install

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import os
import pybel

out_type = 'gaussian'   # either gamess or gaussian
savepath = os.path.join('/Users/dedan/projects/master/data/input_files/', out_type)
mol_file = '/Users/dedan/projects/master/data/molecules.sdf'

file_types = {'gaussian': 'com', 'gamess': 'inp'}

print('reading files from: {}'.format(mol_file))
molecules = pybel.readfile('sdf', mol_file)
for i, mol in enumerate(molecules):

    # read molid from database entry
    molid = mol.data['CdId']

    # write the input file
    # TODO: schreibt der noch was dazu
    input_file = os.path.join(savepath, molid + '.' + file_types[out_type])
    with open(input_file, 'w') as f:
        f.write(mol.write(file_types[out_type]))

print('all input files written to: {}'.format(savepath))
