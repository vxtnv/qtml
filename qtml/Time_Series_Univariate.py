import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_predict
from pyrstat import confusionMatrix, postResample
from qtml.detect_mode import detect_mode
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


'''
Outer Fold 4:
  [════════════TRAIN════════════════]  [══TEST══]
       ↓
  Inner CV (GridSearchCV):
       [TRAIN══════]  [VAL]
       [TRAIN═══════════]  [VAL]
       [TRAIN══════════════════]  [VAL]
       → Beste Hyperparameter gefunden
       
  → Refit mit besten Params auf gesamtem TRAIN
  → Predict auf TEST → RMSE₄ (unbiased!)
'''
'''
ARIMA:    autoarima(y_train) → AIC wählt (p,d,q) → forecast(X_test) → RMSE_test
XGBoost:  GridSearchCV(X_train, y_train) → CV wählt Params → predict(X_test) → RMSE_test
'''


# from sklearn.pipeline import Pipeline



def run(X_train, y_train, X_test, y_test, models, cv=5, max_train_size=None):
    # Classification vs Regression Detection
    mode = detect_mode(y_train)

    scoring = "accuracy" if mode == "classification" else "neg_mean_squared_error"


    results = {}

    tscv = TimeSeriesSplit(n_splits=cv,max_train_size=max_train_size)
    for name, (model, params, n_jobs) in models.items():
        if params == {}:
            # Kein Grid nötig → direkt fitten
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            best_params = "AIC-selected"

        else:
            grid = GridSearchCV(model, params, cv=tscv, scoring=scoring, n_jobs=n_jobs)
            grid.fit(X_train, y_train)
            pred = grid.best_estimator_.predict(X_test)
            best_params = grid.best_params_
            


        results[name] = {"pred": pred, "best_params": best_params}

    
    return results

def rolling_forecast(df, target, models, test_start, test_end, cv=5):
    """
    df:          DataFrame mit Features + Target + Datumsspalte
    target:      Name der Target-Spalte, z.B. "return"
    test_start:  z.B. "2025-01-01"
    test_end:    z.B. "2025-03-31"
    """

    # Datumsspalte automatisch erkennen
    date_col = df.select_dtypes(include=["datetime64"]).columns[0]
    dates = df[date_col]
    
    start_idx = dates[dates >= test_start].index[0]
    end_idx   = dates[dates <= test_end].index[-1] + 1
    

    
    # X und y extrahieren
    feature_cols = [c for c in df.columns if c != target and c != date_col]
    X = df[feature_cols].values
    y = df[target].values
    



    predictions = {name: [] for name in models}
    
    for t in range(start_idx, end_idx):
        results = run(X[:t], y[:t], X[t:t+1], y[t:t+1], models, cv=cv)
        
        for name, res in results.items():
            predictions[name].append(res["pred"][0])
    


    # Evaluation hier — über alle Predictions
    y_test = y[start_idx:end_idx]
    mode = detect_mode(y_test)
    
    for name, preds in predictions.items():
        print(f"----- Model: {name} ------------------------------------------------------------------------------- ")
        print(f"--------------------------------------------------------------------------------------------------- ")
        if mode == "classification":
            # print(f"{name}: Best params={best_params}")
            import rpy2.robjects as ro
            ro.r('options(warn = -1)')
            confusionMatrix(np.array(preds), y_test, positive="1")
        else:
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            # print(f"{name}: Best={best_params}, RMSE={rmse:.4f}")
            print(f"{name}: RMSE={rmse:.4f}")
            postResample(preds, y_test)
    
    return predictions
    











