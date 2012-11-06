#!/usr/bin/env python
# encoding: utf-8
"""
my learning methods

basically the models from sklearn but encapsulated with cross_validation, etc

Created by  on 2012-01-27.
Copyright (c) 2012. All rights reserved.
"""

import sys
import os
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
import numpy as np


class MySVR(SVR):
    """docstring for MySVR"""
    def __init__(self, cross_val=True, n_folds=10, **kwargs):
        super(MySVR, self).__init__(**kwargs)
        self.cross_val = cross_val
        self.n_folds = n_folds
        self.r2_score_ = None

    def fit(self, data, targets):
        """docstring for fit"""
        super(MySVR, self).fit(data, targets)

        params = self.get_params()
        del(params['cross_val'])
        del(params['n_folds'])

        tmp_svr = SVR(**params)
        if self.cross_val:
            kf = KFold(len(targets), self.n_folds)
            predictions = np.zeros(len(targets))
            for train, test in kf:
                predictions[test] = tmp_svr.fit(data[train], targets[train]).predict(data[test])
            self.r2_score_ = r2_score(targets, predictions)
        return self
