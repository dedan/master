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
from sklearn.cross_validation import StratifiedKFold
import numpy as np

class MyStratifiedKFold(StratifiedKFold):
    """A StratifiedKFold for continous targets

       the target range is divided into n_bins bins which are used as
       class labels
    """
    def __init__(self, targets, n_folds, n_bins=2):
        _, bins = np.histogram(targets, bins=n_bins)
        # because of the < instead of <= condition in digitize
        bins[-1] += 0.000001
        binned_targets = np.digitize(targets, bins)
        super(MyStratifiedKFold, self).__init__(binned_targets, n_folds)


class MySVR(SVR):
    """docstring for MySVR"""
    def __init__(self, cross_val=True, n_folds=10, stratified=False, **kwargs):
        super(MySVR, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.cross_val = cross_val
        self.n_folds = n_folds
        self.r2_score_ = None
        self.oob_score_ = None
        self.stratified = stratified

    def fit(self, data, targets):
        """docstring for fit"""
        super(MySVR, self).fit(data, targets)
        tmp_svr = SVR(**self.kwargs)
        if self.cross_val:
            if self.stratified:
                kf = MyStratifiedKFold(targets, self.n_folds)
            else:
                kf = KFold(len(targets), self.n_folds)
            predictions = np.zeros(len(targets))
            for train, test in kf:
                predictions[test] = tmp_svr.fit(data[train], targets[train]).predict(data[test])
            self.r2_score_ = r2_score(targets, predictions)
            self.oob_score_ = self.r2_score_
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
            self.oob_score_ = self.score_oob(data, targets)

    def predict(self, data):
        """docstring for predict"""
        predictions = np.zeros(data.shape[0])
        for svr in self.ensemble:
            predictions += svr.predict(data)
        return predictions / len(self.ensemble)

    def predict_oob(self, data):
        """predictions for all datapoints but only from the parts

           of the ensemble that never saw this point
        """
        n_observations = data.shape[0]
        predictions = np.zeros(n_observations)
        n_predictions = np.zeros(n_observations)
        for svr in self.ensemble:
            assert n_observations == len(svr.indices_)
            mask = np.ones(n_observations, dtype=np.bool)
            mask[svr.indices_] = False
            p_estimator = svr.predict(data[mask])
            predictions[mask] += p_estimator
            n_predictions[mask] += 1
        return predictions / n_predictions

    def score(self, data, targets):
        """docstring for score"""
        return r2_score(targets, self.predict(data))

    def score_oob(self, data, targets):
        """docstring for score"""
        return r2_score(targets, self.predict_oob(data))

    def get_params(self):
        """docstring for get_params"""
        return self.ensemble[0].get_params()
