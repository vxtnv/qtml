# qtml — Quantitative Machine Learning

A Python library for tabular prediction tasks across all major data structures: cross-sectional, panel, and time series data.

The goal is a single, reusable interface so you don't rewrite feature engineering, model selection, and validation logic for every new dataset.

---

## Verified Results

**Cross-Sectional** · [`Examples/Cross_Sectional/`](Examples/Cross_Sectional/)

| Competition | Placement | Note |
|---|---|---|
| [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) | — | Classification |
| [House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | — | Regression |

**Panel** · [`Examples/Panel/`](Examples/Panel/)

| Competition | Placement | Note |
|---|---|---|
| [Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/) | **45 / 904** (Top 5%) | Live competition, March 2026 |
| [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/) | **9 / 3809** (Top 1%) | Late submission — used as integration benchmark |
| [JPX Tokyo Stock Exchange Prediction](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction/) | — | In progress |
| [Stocks Return Prediction v2](https://www.kaggle.com/competitions/stocks-return-prediction-v-2/) | — | In progress |

**Time Series** · [`Examples/Time_Series/`](Examples/Time_Series/)

> Results not yet verified — in progress.

| Competition | Placement | Note |
|---|---|---|
| — | — | — |

---

## Modules

### `Cross_Sectional`

For standard tabular datasets without a time dimension.

- Auto-detects **classification vs. regression** from the target variable
- Runs `GridSearchCV` or `RandomizedSearchCV` across multiple models in one call
- Returns best estimators per model with metrics

```python
from qtml import Cross_Sectional

results = Cross_Sectional.run(X, y, models={
    "xgb": {"model": XGBClassifier(), "params": {...}, "search": "random"},
    "rf":  {"model": RandomForestClassifier(), "params": {...}},
})
```

---

### `Panel`

For datasets with an entity dimension and a time dimension (e.g. stores × days, stocks × minutes).

- Configurable via `PanelConfig` dataclass
- Automatic feature engineering: lags, rolling statistics, calendar features, target encoding
- Supports **LightGBM** and a **PyTorch MLP** (sklearn-compatible interface) with RMSPE/log-target loss
- Rolling forecast evaluation

```python
from qtml.Panel import PanelConfig, run

config = PanelConfig(
    date_col="date",
    target_col="sales",
    entity_cols=["store_nbr", "family"],
    lags=[1, 7, 14, 28],
    rollings=[7, 28],
    horizon=16,
)

model = run(train, config)
```

---

### `Time_Series_Univariate`

For single time series — one variable, one entity.

- `TimeSeriesSplit` cross-validation (no data leakage)
- Supports ARIMA (AIC-selected) alongside ML models (XGBoost, etc.) in the same call
- `rolling_forecast()` for walk-forward evaluation

```python
from qtml import Time_Series_Univariate

results = Time_Series_Univariate.run(
    X_train, y_train, X_test, y_test,
    models={
        "arima":   (AutoARIMA(), {}, 1),
        "xgboost": (XGBRegressor(), {"n_estimators": [100, 300]}, -1),
    }
)
```

---

## Installation

```bash
git clone https://github.com/yourusername/qtml.git
cd qtml
pip install -e .
```

**Dependencies:** `scikit-learn`, `numpy`, `torch`, `lightgbm`, `polars`

---

## Structure

```
qtml/
├── qtml/
│   ├── Cross_Sectional.py
│   ├── Panel.py
│   ├── Time_Series_Univariate.py
│   └── detect_mode.py
├── Models/               # Standalone model implementations
│   ├── Cross_Sectional/
│   ├── Panel/
│   └── Time_Series_Univariate/
└── Examples/             # Kaggle competition notebooks
    ├── Cross_Sectional/
    ├── Panel/
    └── Time_Series/
```
