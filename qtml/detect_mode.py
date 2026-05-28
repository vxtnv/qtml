import numpy as np


def detect_mode(y, threshold=20):
    """Auto-detect classification vs regression."""
    y = np.array(y)
    
    # Floats mit Dezimalstellen → immer Regression
    if np.issubdtype(y.dtype, np.floating):
        # Prüfe ob es eigentlich diskrete Werte sind (z.B. 0.0, 1.0, 2.0)
        if np.all(y == np.floor(y)):
            nunique = len(np.unique(y))
            return "classification" if nunique <= threshold else "regression"
        return "regression"
    
    # Integers
    nunique = len(np.unique(y))
    return "classification" if nunique <= threshold else "regression"