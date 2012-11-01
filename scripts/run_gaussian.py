#!/usr/bin/env python
# encoding: utf-8
"""
    script to calc different spectra using GAUSSIAN

    it does not call the programm like the run_gamess script but submits the
    jobs to the queuing system on the soroban cluster

    the results of this script can be parsed to a simpler structure
    with readout_gaussian.py
"""

import os, sys, glob, json
import subprocess

config = json.load(open(sys.argv[1]))
headers = json.load(open(os.path.join(config['module_path'], 'data', 'headers.json')))
header = headers[config['header']]
cluster_instr = "%MEM={memory}\n%NProcShared={n_nodes}".format(**config)

input_path = os.path.join(config['module_path'], 'data', 'input_files', 'gaussian')
input_files = glob.glob(os.path.join(input_path, '*.com'))

log = open(os.path.join(config['tmp_folder'], 'gamess_log.log'), 'w')
wd = os.getcwd()
os.chdir(config['tmp_folder'])

for input_file in input_files:

    tmp_file = os.path.join(config['tmp_folder'], os.path.basename(input_file))
    with open(input_file) as f:
        data = f.readlines()
    with open(tmp_file, 'w') as f:
        f.write("\n".join([cluster_instr, '\n'.join(header), ''.join(data[1:])]))

    # call gamess
    cmd = 'subg09 ' + tmp_file
    print '\trunning: %s' % cmd
    subprocess.call(cmd, shell=True, stdout=log, stderr=log)

os.chdir(wd)


