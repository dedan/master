#!/usr/bin/env python
# encoding: utf-8
"""
    script to calc infrared spectra using gamess

    the results of this script can be parsed to a simpler structure
    with readout_ir.py
"""

import pybel
import os
import subprocess
import glob

header = ''' $SYSTEM MWORDS=20 $END
 $CONTRL RUNTYP=Optimize $END
 $STATPT  HSSEND=.T. OptTol=1e-4 NStep=500 $END
 $CONTRL SCFTYP=RHF $END
 $FORCE nvib=2 $END
 $BASIS gbasis=AM1 $END
 '''

redo = True
molfile = pybel.readfile('sdf','/Users/dedan/projects/master/data/molecules.sdf')
savepath = '/Users/dedan/projects/master/results/gamess'
scratch_folder = '/Users/dedan/tmp/scr'
gms_path = '/Users/dedan/Downloads/gamess/rungms'

for mol in molfile:

    # read molid from database entry
    molid = mol.data['CdId']
    print 'working on: ', molid
    input_file = molid + '.inp'
    outfile = os.path.join(savepath, molid + '.log')
    if os.path.exists(outfile):
        print '\tskipping, output file already exists'
        continue

    # clear the scratchfolder from tempfiles
    junkfiles = glob.glob(os.path.join(scratch_folder, molid + '.*'))
    for f in junkfiles:
        if redo:
            os.remove(f)

    # write the input file
    data = mol.write('inp')
    with open(input_file, 'w') as f:
        f.writelines("\n".join([header, data]))

    # call gamess
    cmd = gms_path +' ' + molid + '.inp' +' May12012R1 1 > '+ outfile
    print 'running ', cmd
    subprocess.call(cmd, shell=True)
    os.remove(input_file)
