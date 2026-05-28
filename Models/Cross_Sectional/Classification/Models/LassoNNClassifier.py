from lassonet import LassoNetClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LassoNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=(64, 32), M=10, verbose=0, n_iters=100):
        self.hidden_dims = hidden_dims
        self.M = M
        self.verbose = verbose
        self.n_iters = n_iters

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = LassoNetClassifier(
            hidden_dims=self.hidden_dims,
            M=self.M,
            verbose=self.verbose,
            n_iters=self.n_iters,
        )
        X = np.array(X, dtype=np.float32, copy=True)   # copy=True macht es writable
        y = np.array(y, copy=True)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32, copy=True)
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    



