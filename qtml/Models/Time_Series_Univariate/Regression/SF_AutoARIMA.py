
# ══════════════════════════════════════════════════════════════════════
# StatsForecast AutoARIMA — Sklearn-kompatibel
# ══════════════════════════════════════════════════════════════════════
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class SF_AutoARIMA(BaseEstimator, RegressorMixin):
    """
    Sklearn-kompatibler Wrapper für StatsForecast AutoARIMA.
    Ersetzt R's forecast::auto.arima() durch Python-native Implementierung.
    
    - Kein rpy2 nötig
    - Bis zu 30x schneller als R
    - Gleicher Algorithmus (Hyndman-Khandakar)
    """
    
    def __init__(self, season_length=1, max_p=5, max_q=5, max_d=2):
        self.season_length = season_length
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.model_ = None
    
    def fit(self, X, y):
        from statsforecast.models import AutoARIMA as _AutoARIMA
        
        y = np.asarray(y).flatten()
        self.model_ = _AutoARIMA(
            season_length=self.season_length,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d
        )
        self.model_.fit(y)
        return self
    
    def predict(self, X):
        n_steps = X.shape[0] if hasattr(X, 'shape') else 1
        forecast = self.model_.predict(h=n_steps)
        pred = forecast.get("mean", forecast)
        return np.asarray(pred).flatten()[:n_steps]



