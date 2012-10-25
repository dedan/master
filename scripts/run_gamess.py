#!/usr/bin/env python
# encoding: utf-8
"""
    script to calc infrared spectra using gamess

    the results of this script can be parsed to a simpler structure
    with readout_ir.py
"""

import pybel
import os, sys, glob, json, time
import subprocess
from configobj import ConfigObj
from datetime import datetime, timedelta

config = ConfigObj(sys.argv[1], unrepr=True)

molfile_path = os.path.join(config['module_path'], 'data', 'molecules.sdf')
molfile = pybel.readfile('sdf', molfile_path)
headers = json.load(open(os.path.join(config['module_path'], 'data', 'gamess_headers.json')))
header = headers[config['header']]
log = open(os.path.join(config['savepath'], 'gamess_log.log'), 'w')

times = []

for i, mol in enumerate(molfile):

    # read molid from database entry
    molid = mol.data['CdId']
    print 'working on file %s' % molid
    t_start = time.time()

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
    print '\trunning: %s' % cmd
    subprocess.call(cmd, shell=True, stdout=log, stderr=log)
    os.remove(input_file)

    delta_t = time.time() - t_start
    times.append(delta_t)
    d = datetime(1,1,1) + timedelta(seconds=times[-1])
    print("\tfinished after: %d h %d min %d s" % (d.hour, d.minute, d.second))

    if i % 10 == 0 and i:
        d = datetime(1,1,1) + timedelta(seconds=(sum(times)/len(times)))
        print("\n\tavg time: %d h %d min %d s\n" % (d.hour, d.minute, d.second))




