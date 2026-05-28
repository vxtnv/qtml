import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from pyrstat import confusionMatrix, postResample
from qtml.detect_mode import detect_mode
from sklearn.metrics import mean_squared_error


# from sklearn.pipeline import Pipeline



def run(X, y, models, test_size=0.25, cv=5, random_state=42, n_iter=20):
        # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Classification vs Regression Detection
    mode = detect_mode(y)

    scoring = "accuracy" if mode == "classification" else "neg_mean_squared_error"


    results = {}

    for name, config in models.items():
        # Accept both tuple (model, params, n_jobs) and dict format
        if isinstance(config, tuple):
            model, params, n_jobs = config
            grid = GridSearchCV(model, params, cv=cv, scoring=scoring, n_jobs=n_jobs)
        else:
            search = config.get("search", "grid")
            if search == "random":
                grid = RandomizedSearchCV(config["model"], config["params"], n_iter=n_iter,
                           cv=cv, scoring=scoring, n_jobs=config.get("n_jobs", -1),
                           random_state=random_state)
            else:
                grid = GridSearchCV(config["model"], config["params"],
                           cv=cv, scoring=scoring, n_jobs=config.get("n_jobs", -1))
            

        
        grid.fit(X_train, y_train)
        pred = grid.best_estimator_.predict(X_test)


        results[name] = grid.best_estimator_

        print(f"----- Model: {name} ------------------------------------------------------------------------------- ")
        print(f"--------------------------------------------------------------------------------------------------- ")
        print(f"Best: {grid.best_params_}")

        if mode == "classification":
            print(f"{name}: Best params={grid.best_params_}, Error Rate={1-grid.best_score_:.4f}")


            import rpy2.robjects as ro
            ro.r('options(warn = -1)') 
            confusionMatrix(pred, y_test, positive="1")

            
        else:
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            print(f"{name}: Best={grid.best_params_}, RMSE={rmse:.4f}")

            postResample(pred, y_test)


    return results








