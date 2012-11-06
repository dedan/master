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
        self.kwargs = kwargs
        self.cross_val = cross_val
        self.n_folds = n_folds
        self.r2_score_ = None

    def fit(self, data, targets):
        """docstring for fit"""
        super(MySVR, self).fit(data, targets)
        tmp_svr = SVR(**self.kwargs)
        if self.cross_val:
            kf = KFold(len(targets), self.n_folds)
            predictions = np.zeros(len(targets))
            for train, test in kf:
                predictions[test] = tmp_svr.fit(data[train], targets[train]).predict(data[test])
            self.r2_score_ = r2_score(targets, predictions)
        return self


class SVREnsemble(object):
    """docstring for SVREnsemble"""
    def __init__(self, n_estimators, oob_score, **kwargs):
        super(SVREnsemble, self).__init__()
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = None
        np.random.seed(0)
        self.ensemble = []
        for i in range(self.n_estimators):
            self.ensemble.append(SVR(**kwargs))

    def fit(self, data, targets):
        """docstring for fit"""
        for svr in self.ensemble:
            indices = np.random.randint(0, len(targets), len(targets))
            svr.indices_ = indices
            svr.fit(data[svr.indices_], targets[svr.indices_])

        if self.oob_score:
            predictions = np.zeros(len(targets))
            n_predictions = np.zeros(len(targets))
            for svr in self.ensemble:
                mask = np.ones(len(targets), dtype=np.bool)
                mask[svr.indices_] = False
                p_estimator = svr.predict(data[mask])
                predictions[mask] += p_estimator
                n_predictions[mask] += 1
            predictions /= n_predictions
            self.oob_score_ = r2_score(targets, predictions)

    def predict(self, data):
        """docstring for predict"""
        predictions = np.zeros(data.shape[0])
        for svr in self.ensemble:
            predictions += svr.predict(data)
        return predictions / len(self.ensemble)

    def score(self, data, targets):
        """docstring for score"""
        return r2_score(targets, self.predict(data))

    def get_params(self):
        """docstring for get_params"""
        return self.ensemble[0].get_params()
