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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
import numpy as np
import random
import __builtin__

class StratifiedResampling(object):
    """like stratified k-fold, but with repeated sampling"""
    def __init__(self, targets, n_folds, n_bins=2):
        self.targets = targets
        self.n_folds = n_folds
        self.n_bins = n_bins

        self.counts, bins = np.histogram(targets, bins=n_bins)
        # because of the < instead of <= condition in digitize
        bins[-1] += 0.000001
        self.binned_targets = np.digitize(targets, bins)

    def __iter__(self):
        for fold in range(self.n_folds):
            train_idx = []
            for i, target_type in enumerate(range(1, self.n_bins + 1)):
                idx = np.where(self.binned_targets == target_type)[0]
                sample = [random.choice(idx) for _ in range(self.counts[i])]
                train_idx.extend(sample)
            assert len(train_idx) == len(self.binned_targets)
            test_idx = [j for j in range(len(self.targets)) if not j in train_idx]
            assert len(set(train_idx).intersection(test_idx)) == 0
            yield train_idx, test_idx


def _k_best_indeces(data, targets, selection_method, k):
    """get indices for the k best features depending on the scores"""
    assert k > 0
    if selection_method == 'linear':
        scores, _ = f_regression(data, targets)
    elif selection_method == 'forest':
        rfr_sel = RandomForestRegressor(compute_importances=True, random_state=0)
        scores = rfr_sel.fit(data, targets).feature_importances_
    assert not (scores < 0).any()
    assert len(scores) >= k
    scores[np.isnan(scores)] = 0
    return np.argsort(scores)[-k:]


class MySVR(SVR):
    """docstring for MySVR"""
    def __init__(self, cross_val=True, n_folds=10, **kwargs):
        super(MySVR, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.cross_val = cross_val
        self.n_folds = n_folds
        self.gen_score = None

    def fit(self, data, targets, selection_method, k_best):
        """docstring for fit"""
        assert data.shape[0] == len(targets)
        self.best_idx = _k_best_indeces(data, targets, selection_method, k_best)
        super(MySVR, self).fit(data[:, self.best_idx], targets)
        tmp_svr = SVR(**self.kwargs)
        if self.cross_val:
            kf = StratifiedResampling(targets, self.n_folds)
            all_predictions, all_targets = [], []
            for train, test in kf:
                best_idx = _k_best_indeces(data[train], targets[train], selection_method, k_best)
                tmp_svr.fit(data[np.ix_(train, best_idx)], targets[train])
                all_predictions.extend(tmp_svr.predict(data[np.ix_(test, best_idx)]))
                all_targets.extend(targets[test])
            self.gen_score = r2_score(all_targets, all_predictions)
            self.all_predictions = all_predictions
            self.all_targets = all_targets
        return self

    def predict(data):
        """predict after feature selection"""
        super(MySVR, self).predict(data[:, self.best_idx])

    def score(data):
        """score after feature selection"""
        super(MySVR, self).score(data[:, self.best_idx])


class SVREnsemble(object):
    """docstring for SVREnsemble"""
    def __init__(self, n_estimators, cross_val=True, n_folds=10, **kwargs):
        self.n_estimators = n_estimators
        self.n_folds = n_folds
        self.cross_val = cross_val
        self.gen_score = None
        self.kwargs = kwargs
        np.random.seed(0)
        self.ensemble = []
        for i in range(self.n_estimators):
            self.ensemble.append(SVR(**kwargs))

    def fit(self, data, targets, selection_method, k_best):
        """docstring for fit"""
        assert data.shape[0] == len(targets)
        sr = StratifiedResampling(targets, self.n_estimators)
        self.best_idx = _k_best_indeces(data, targets, selection_method, k_best)
        for svr, (train_idx, _) in zip(self.ensemble, sr):
            svr.indices_ = train_idx
            svr.fit(data[np.ix_(svr.indices_, self.best_idx)], targets[svr.indices_])
        if self.cross_val:
            sr_val = StratifiedResampling(targets, self.n_folds)
            all_predictions, all_targets = [], []
            for train, test in sr_val:
                test_ensemble = SVREnsemble(self.n_estimators, cross_val=False, **self.kwargs)
                test_ensemble.fit(data[train], targets[train], selection_method, k_best)
                all_predictions.extend(test_ensemble.predict(data[test]))
                all_targets.extend(targets[test])
            self.gen_score = r2_score(all_targets, all_predictions)
            self.all_predictions = all_predictions
            self.all_targets = all_targets
        return self

    def predict(self, data):
        """prediction on selected features"""
        data_sel = data[:, self.best_idx]
        predictions = np.zeros(data_sel.shape[0])
        for svr in self.ensemble:
            predictions += svr.predict(data_sel)
        return predictions / len(self.ensemble)

    def score(self, data, targets):
        """score on selected features"""
        return r2_score(targets, self.predict(data))

class MyRFR(RandomForestRegressor):
    """overwrite RFR to include feature selection during fitting"""
    def __init__(self, n_estimators, cross_val=True, **kwargs):
        super(MyRFR, self).__init__(oob_score=cross_val, **kwargs)
        self.n_estimators = n_estimators
        self.cross_val = cross_val
        self.gen_score = None

    def fit(self, data, targets, selection_method, k_best):
        """fit on selected features"""
        self.best_idx = _k_best_indeces(data, targets, selection_method, k_best)
        super(MyRFR, self).fit(data[:,self.best_idx], targets)
        if self.cross_val:
            self.gen_score = self.oob_score_

    def predict(data):
        """predict after feature selection"""
        super(MySVR, self).predict(data[:, self.best_idx])

    def score(data):
        """score after feature selection"""
        super(MySVR, self).score(data[:, self.best_idx])






