#!/usr/bin/env python
# encoding: utf-8
"""

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
import numpy as np
import pylab as plt

def plot_search_matrix(fig, desc_res, sc, methods):
    """docstring for plot_search_matrix"""
    for i_sel, selection in enumerate(sc['selection']):
        for i_glom, glom in enumerate(desc_res[selection]):
            for i_meth, method in enumerate(methods):
                mat = desc_res[selection][glom][method]
                ax_idx = i_meth * len(sc['glomeruli']) * 2 + len(sc['glomeruli']) * i_sel + i_glom + 1
                ax = fig.add_subplot(6, len(sc['glomeruli']), ax_idx)
                ax.imshow(mat, interpolation='nearest')
                if i_sel + i_meth == 0:
                    ax.set_title(glom, rotation='45')
                if i_glom == 0:
                    ax.set_ylabel('{}\n{}'.format(method,selection))
                ax.set_yticks([])
                ax.set_xticks([])
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
