
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class SF_AutoARIMAX(BaseEstimator, RegressorMixin):
    """
    Sklearn-kompatibler Wrapper für StatsForecast AutoARIMA mit exogenen Variablen.
    
    XREG_COLS: Klassenvariable — Liste der exogenen Spaltennamen.
    """
    
    XREG_COLS = []
    
    def __init__(self, season_length=1, max_p=5, max_q=5, max_d=2):
        self.season_length = season_length
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.model_ = None
        self._xreg_indices = None
    
    def _get_xreg(self, X):
        """Extrahiere exogene Variablen aus X."""
        X = np.asarray(X)
        if self._xreg_indices is not None:
            return X[:, self._xreg_indices]
        return X
    
    def fit(self, X, y):
        from statsforecast.models import AutoARIMA as _AutoARIMA
        import pandas as pd
        
        y = np.asarray(y).flatten()
        
        # XREG Spalten finden
        if isinstance(X, pd.DataFrame) and self.XREG_COLS:
            available = [c for c in self.XREG_COLS if c in X.columns]
            self._xreg_indices = [X.columns.get_loc(c) for c in available]
            xreg = X[available].values
        else:
            xreg = np.asarray(X)
            self._xreg_indices = list(range(xreg.shape[1]))
        
        self.model_ = _AutoARIMA(
            season_length=self.season_length,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d
        )
        self.model_.fit(y, X=xreg)
        return self
    
    def predict(self, X):
        xreg = self._get_xreg(X)
        n_steps = xreg.shape[0]
        forecast = self.model_.predict(h=n_steps, X=xreg)
        pred = forecast.get("mean", forecast)
        return np.asarray(pred).flatten()[:n_steps]

