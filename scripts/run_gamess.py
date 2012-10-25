#!/usr/bin/env python
# encoding: utf-8
"""
    script to calc infrared spectra using gamess

    the results of this script can be parsed to a simpler structure
    with readout_ir.py
"""

import pybel
import os, sys, glob, json
import subprocess
from configobj import ConfigObj

config = ConfigObj(sys.argv[1], unrepr=True)

molfile_path = os.path.join(config['module_path'], 'data', 'molecules.sdf')
molfile = pybel.readfile('sdf', molfile_path)
headers = json.load(open(os.path.join(config['module_path'], 'data', 'gamess_headers.json')))

header = headers[config['header']]

for mol in molfile:

    # read molid from database entry
    molid = mol.data['CdId']
    print 'working on: ', molid
    input_file = molid + '.inp'
    outfile = os.path.join(config['savepath'], molid + '.log')

    if os.path.exists(outfile) and not config['redo']:
        print '\tskipping, output file already exists'
        continue

    # clear the scratchfolder from tempfiles
    junkfiles = glob.glob(os.path.join(config['scratch_folder'], molid + '.*'))
    for f in junkfiles:
        if config['redo']:
            os.remove(f)

    # write the input file
    data = mol.write('inp')
    with open(input_file, 'w') as f:
        f.writelines("\n".join(['\n'.join(header), data]))

    # call gamess
    cmd = [config['gms_path'], molid + '.inp',
           config['gamess_version'], config['n_nodes'],
           '>', outfile]
    cmd = ' '.join([str(c) for c in cmd])
    print 'running ', cmd
    subprocess.call(cmd, shell=True)
    os.remove(input_file)
