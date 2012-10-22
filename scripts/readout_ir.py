#!/usr/bin/env python
# encoding: utf-8
"""
    file to read the results produced by calc_vib
"""

import pickle, os, glob

inpath = '/Users/dedan/projects/master/results/gamess/'
n_lines_after_warning = 0
outdict = {}

uncritical_warnings = ['OLD KEYWORD COORD=CART', 'WARNING, MODE 7']

outfiles=glob.glob(inpath + '*')
for outfilename in outfiles:

    print 'reading from: ', outfilename
    freq, ir, nonvibrations = [], [], []
    outfile = open(outfilename)
    for line in outfile:

        # print the N lines after a warning
        if 'WARNING' in line:
            n_lines_after_warning = 3
        for uw in uncritical_warnings:
            if uw in line:
                n_lines_after_warning = 0
        if not '* * * WARNING * * *' in line and n_lines_after_warning and line:
            print line
        if n_lines_after_warning:
            n_lines_after_warning -= 1

        # collect frequencies and intensities
        if 'FREQUENCY' in line and not 'IMAGINARY' in line:
            freq.extend(line.split()[1:])
        if 'INTENSITY' in line:
            ir.extend(line.split()[2:])

        # select only the vibration modes
        if 'MODES 1 TO 6' in line:
            assert not nonvibrations
            nonvibrations = range(6)
        if 'WARNING, MODE 1' in line:
            assert not nonvibrations
            nonvibrations = range(1, 7)

    # select only the vibration modes (!!! only if not imaginary !!!)
    freq = [float(freq[i]) for i in range(len(freq)) if not 'I' in freq[i] and not i in nonvibrations]
    ir = [float(ir[i]) for i in range(len(ir)) if not 'I' in ir[i] and not i in nonvibrations]
    outdict[os.path.splitext(os.path.basename(outfilename))[0]] = zip(freq, ir)

pickle.dump(outdict, open(os.path.join(inpath, 'ir.pckl'),'w'))

