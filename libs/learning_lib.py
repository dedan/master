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


def _k_best_indeces(data, selection_method, k):
    """get indices for the k best features depending on the scores"""
    assert k > 0
    if selection_method == 'linear':
        sel_scores, _ = f_regression(data, targets)
    elif selection_method == 'forest':
        rfr_sel = RandomForestRegressor(compute_importances=True, random_state=0)
        sel_scores = rfr_sel.fit(data, targets).feature_importances_
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
        self.r2_score_ = None
        self.oob_score_ = None

    def fit(self, data, targets, selection_method, k_best):
        """docstring for fit"""
        assert data.shape[0] == len(targets)
        best_idx = _k_best_indeces(data, selection_method, k_best)
        super(MySVR, self).fit(data_sel[:, best_idx], targets)
        tmp_svr = SVR(**self.kwargs)
        if self.cross_val:
            kf = StratifiedResampling(targets, self.n_folds)
            all_predictions, all_targets = [], []
            for train, test in kf:
                best_idx = _k_best_indeces(data[train], selection_method, k_best)
                all_predictions.extend(tmp_svr.fit(data[train, best_idx], targets[train])
                                              .predict(data[test, best_idx]))
                all_targets.extend(targets[test])
            self.r2_score_ = r2_score(all_targets, all_predictions)
            self.oob_score_ = self.r2_score_
            self.all_predictions = all_predictions
            self.all_targets = all_targets
        return self


class SVREnsemble(object):
    """docstring for SVREnsemble"""
    def __init__(self, n_estimators, oob_score, stratified=False, **kwargs):
        super(SVREnsemble, self).__init__()
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.stratified = stratified
        self.oob_score_ = None
        np.random.seed(0)
        self.ensemble = []
        for i in range(self.n_estimators):
            self.ensemble.append(SVR(**kwargs))

    def fit(self, data, targets):
        """docstring for fit"""
        assert data.shape[0] == len(targets)

        if self.stratified:
            sr = StratifiedResampling(targets, self.n_estimators)
            indices = __builtin__.sum([train_idx for train_idx, test_idx in sr], [])
        else:
            indices = np.random.randint(0, len(targets),
                                        (self.n_estimators, len(targets)))

        for svr, idx in zip(self.ensemble, indices):
            svr.indices_ = idx
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
