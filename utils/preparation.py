# -*- coding: utf-8 -*-
"""Preparation class"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""

    def __init__(self, positions):
        self.positions = positions

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # return np.array(X)[:, self.positions]
        return X.loc[:, self.positions]

class PositionalSelector(BaseEstimator, TransformerMixin):
    def __init__(self, positions):
        self.positions = positions

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array(X)[:, self.positions]
