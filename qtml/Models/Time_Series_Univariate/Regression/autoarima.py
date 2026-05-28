from sklearn.base import BaseEstimator, RegressorMixin
from pyrstat import auto_arima, forecast
import numpy as np
import rpy2.robjects as ro
import contextlib, io



class autoarima(BaseEstimator, RegressorMixin):
    """Sklearn-kompatibler Wrapper für pyrstat.autoarima"""
        
    def __init__(self, max_order=5, seasonal=False, d=None):
        self.max_order = max_order
        self.seasonal = seasonal
        self.d = d
        self.model_ = None
    

    def fit(self, X, y):
        y_r = ro.FloatVector([float(v) for v in y])  # ← explizit R numeric vector
        self.model_ = auto_arima(
            y_r,
            max_order=self.max_order,
            seasonal=self.seasonal,
            d=self.d,
            pretty=False
        )
        return self


    def predict(self, X):
        h = len(X)
        with contextlib.redirect_stdout(io.StringIO()):
            fc = forecast(self.model_, h, plot=False)
        point_forecast = ro.r('function(fc) as.numeric(fc$mean)')(fc)
        return np.array(point_forecast)

    

    
