#!/usr/bin/env python
# encoding: utf-8
"""
    is there a problem if I don't re-initialize the models in X-validation?

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os

from sklearn.svm import SVR
from sklearn.datasets import make_regression

svr1 = SVR()
svr2 = SVR()
data1, targets1 = make_regression()
data2, targets2 = make_regression()

print svr1.fit(data1, targets1).fit(data2, targets2).predict(data2[0:10])
print svr2.fit(data2, targets2).predict(data2[0:10])