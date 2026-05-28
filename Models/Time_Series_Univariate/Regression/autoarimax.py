

from sklearn.base import BaseEstimator, RegressorMixin
from pyrstat import auto_arima, forecast
import numpy as np
import rpy2.robjects as ro
import contextlib, io




class autoarimax(BaseEstimator, RegressorMixin):
    """ARIMAX: auto.arima MIT exogenen Variablen (xreg)"""
    
    def __init__(self, max_order=5, seasonal=False, d=None):
        self.max_order = max_order
        self.seasonal = seasonal
        self.d = d
        self.model_ = None
        self.xreg_cols_ = None
        
    def fit(self, X, y):
        y_r = ro.FloatVector([float(v) for v in y])
        
        # X als R-Matrix übergeben
        X_np = np.array(X, dtype=float)
        xreg_r = ro.r.matrix(
            ro.FloatVector(X_np.flatten(order='F')),
            nrow=X_np.shape[0], ncol=X_np.shape[1]
        )
        
        with contextlib.redirect_stdout(io.StringIO()):
            self.model_ = auto_arima(
                y_r, xreg=xreg_r,
                max_order=self.max_order,
                seasonal=self.seasonal,
                d=self.d,
                pretty=False
            )
        self.xreg_cols_ = X_np.shape[1]
        return self
    
    def predict(self, X):
        h = len(X)
        X_np = np.array(X, dtype=float)
        xreg_future_r = ro.r.matrix(
            ro.FloatVector(X_np.flatten(order='F')),
            nrow=X_np.shape[0], ncol=X_np.shape[1]
        )
        
        with contextlib.redirect_stdout(io.StringIO()):
            fc = forecast(self.model_, h, xreg=xreg_future_r, plot=False)
        return np.array(ro.r('function(fc) as.numeric(fc$mean)')(fc))