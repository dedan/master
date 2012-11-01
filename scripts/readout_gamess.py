#!/usr/bin/env python
# encoding: utf-8
"""
    file to read the results produced by calc_vib
"""

import pickle, os, glob, json
from collections import defaultdict

inpath = '/Users/dedan/projects/master/results/spectra/gamess_am1/'
n_lines_after_warning = 0
outdict = defaultdict(dict)
problems = {'critical': defaultdict(list), 'uncritical': defaultdict(list)}

uncritical_warnings = ['OLD KEYWORD COORD=CART', 'WARNING, MODE 7']
critical_warnings = ['DATA ARE NOT AVAILABLE FOR ELEMENT']

outfiles=glob.glob(inpath + '*.log')
for outfilename in outfiles:

    print 'reading from: ', outfilename
    freq, ir, nonvibrations = [], [], []
    outfile = open(outfilename)
    molid = os.path.splitext(os.path.basename(outfilename))[0]

    for line in outfile:

        # print the N lines after a warning
        if 'WARNING' in line:
            n_lines_after_warning = 3
        for uw in uncritical_warnings:
            if uw in line:
                n_lines_after_warning = 0
        if not '* * * WARNING * * *' in line and n_lines_after_warning and line:
            print line
            problems['uncritical'][molid].append(line)
        if n_lines_after_warning:
            n_lines_after_warning -= 1

        # record critical warning and skip processing for this molecule
        for cw in critical_warnings:
            if cw in line:
                problems['critical'][molid].append(line)

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

    if not freq:
        problems['critical'][molid].append('no FREQUENCY statement found')

    # select only the vibration modes (!!! only if not imaginary !!!)
    freq = [f for f in freq if not 'I' in f]
    freq_cut = [float(freq[i]) for i in range(len(freq)) if not i in nonvibrations]
    ir_cut = [float(ir[i]) for i in range(len(ir)) if not i in nonvibrations]
    assert len(freq_cut) == len(ir_cut)
    if not molid in problems['critical']:
        assert freq
        outdict[molid]['freq'] = freq_cut
        outdict[molid]['ir'] = ir_cut

pickle.dump(dict(outdict), open(os.path.join(inpath, 'parsed.pckl'),'w'))
json.dump(dict(problems), open(os.path.join(inpath, 'problems.json'),'w'))

