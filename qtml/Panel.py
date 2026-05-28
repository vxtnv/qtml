
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Optional, Tuple, List
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb

# ══════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════
@dataclass
class PanelConfig:
    date_col: str
    target_col: str
    entity_cols: Sequence[str]
    categorical_cols: Sequence[str] = ()
    target_encode_cols: Sequence[str] = ()
    event_dates: dict = field(default_factory=dict)
    exogenous: list = field(default_factory=list)
    # Eintrag entweder pl.DataFrame ODER
    # {"frame": pl.DataFrame, "rolling": {"col": [7, 28]}, "diff": ["col"]}

    horizon: int = 1
    lags: Sequence[int] = (1, 2, 3, 7, 14, 28)
    rollings: Sequence[int] = (7, 28)
    rolling_aggs: Sequence[str] = ("mean", "std")     # mean/std/median/min/max

    # ── Optionale Erweiterungen (alle default off) ──
    lag_aggregates: dict = field(default_factory=dict)
    # {"mean_same_dow_4w": (21, 28, 35, 42)}
    extra_rollings: Sequence = ()
    # ((base_lag, window, agg), ...) → z.B. ((364, 7, "mean"),)
    running_count_cols: Sequence[str] = ()
    # ["onpromotion"] → promo_running
    use_hist_mean: bool = False
    use_zero_features: bool = False
    use_payday: bool = False

    use_rmse_loss: bool = False
    precomputed_features: bool = False
    use_calendar: bool = True
    use_lags: bool = True
    te_smoothing: int = 100


# ══════════════════════════════════════════════════════════════════
# MLP Regressor (PyTorch, scikit-learn compatible interface)
# ══════════════════════════════════════════════════════════════════
class MLPRegressor:
    """PyTorch MLP mit RMSPE/Log-Target support — scikit-learn interface.

    Unterstützt zwei Dropout-Raten (wie Optiver V3.92):
    dropout1 für erste Schicht, dropout2 für zweite+dritte Schicht.
    Nutzt Best-State-Tracking (validiert nach jeder Epoche).
    """

    def __init__(self, hidden_dims=(512, 256, 128),
                 dropout1=0.3, dropout2=0.2,
                 n_epochs=30, batch_size=4096, lr=1e-3, weight_decay=1e-5,
                 verbose=False, log_target=False, seed=42):
        self.hidden_dims = hidden_dims
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.log_target = log_target
        self.seed = seed
        self._model = None
        self._mean = self._std = None
        self._best_state = None

    def fit(self, X, y, sample_weight=None, eval_set=None):
        """Train MLP with optional validation for best-state tracking.

        If eval_set is provided (X_val, y_val), tracks best model by val RMSPE.
        Normalizes using training fold only (no leakage).
        V3.92 behavior: RMSPE as plain loss, no sample weights.
        """
        seed = getattr(self, 'seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mean = np.nanmean(X, axis=0)
        self._std = np.nanstd(X, axis=0) + 1e-8

        # Normalize using training data only
        X_tr_n = (X - self._mean) / self._std
        y_tr_n = y.copy()

        # Prepare validation set
        X_val_n, y_val_n = None, None
        if eval_set is not None:
            X_val_n = (eval_set[0] - self._mean) / self._std
            y_val_n = eval_set[1]

        X_t = torch.FloatTensor(X_tr_n).to(device)
        y_t = torch.FloatTensor(y_tr_n).to(device).unsqueeze(1)
        # NOTE: sample_weight is NOT used for MLP — V3.92 uses RMSPE as loss function,
        # not as sample weights. The LGB path handles RMSPE via sample_weight_mode.

        layers = []
        in_dim = X.shape[1]
        for i, h in enumerate(self.hidden_dims):
            d = self.dropout1 if i == 0 else self.dropout2
            layers.extend([nn.Linear(in_dim, h), nn.SiLU(), nn.BatchNorm1d(h),
                           nn.Dropout(d)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self._model = nn.Sequential(*layers).to(device)

        optimizer = torch.optim.Adam(self._model.parameters(),
                                    lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_rmspe, best_state = 999.0, None
        X_val_t = torch.FloatTensor(X_val_n).to(device) if X_val_n is not None else None

        for epoch in range(self.n_epochs):
            self._model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self._model(xb)
                diff = (yb - pred) / (yb.abs() + 1e-8)
                loss = torch.sqrt((diff ** 2).mean())
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validate after each epoch for best-state tracking
            if X_val_t is not None:
                self._model.eval()
                with torch.no_grad():
                    vp = self._model(X_val_t).cpu().numpy()
                val_rmspe = np.sqrt(np.mean(((y_val_n - vp) / (y_val_n + 1e-8)) ** 2))
                if val_rmspe < best_val_rmspe:
                    best_val_rmspe = val_rmspe
                    best_state = {k: v.cpu().clone() for k, v in
                                  self._model.state_dict().items()}

            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"  [MLP] Epoch {epoch+1}/{self.n_epochs}"
                      + (f" val_rmspe={val_rmspe:.6f}" if X_val_t is not None else ""))

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    def predict(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor((X - self._mean) / self._std).to(device)
            pred = self._model(X_t).cpu().numpy().squeeze()
        return pred

# ══════════════════════════════════════════════════════════════════
# Forecaster
# ══════════════════════════════════════════════════════════════════
class PanelForecaster:
    """Generalisierter Panel-TS-Forecaster auf Polars."""

    _AGG = {"mean": "rolling_mean", "std": "rolling_std",
            "median": "rolling_median", "min": "rolling_min", "max": "rolling_max"}

    def __init__(self, config: PanelConfig, models=None, log_target=True,
                 per_entity_col: Optional[str] = None, n_seeds: int = 3,
                 clip_neg: bool = True, verbose: bool = True,
                 sample_weight_mode: str = "none",
                 precomputed_features: bool = False,
                 ensemble_method: str = "mean",
                 n_folds: int = 0, val_ratio: float = 0.1,
                 early_stopping_rounds: int = 0):
        self.cfg = config
        self.models = models or self._defaults()
        self.log_t = log_target
        self.per_entity = per_entity_col
        self.n_seeds = n_seeds
        self.clip_neg = clip_neg
        self.verbose = verbose
        self.sample_weight_mode = sample_weight_mode
        self.precomputed_features = precomputed_features
        self.ensemble_method = ensemble_method
        self.n_folds = n_folds
        self.val_ratio = val_ratio
        self.early_stopping_rounds = early_stopping_rounds
        self.enc_, self.te_, self.feat_, self.fit_ = {}, {}, [], {}
        self.gmean_ = 0.0
        self.hist_: Optional[pl.DataFrame] = None
        self._oof_per_model: List[np.ndarray] = []
        self._blend_weights: Optional[List[float]] = None
        self._X_oof: Optional[np.ndarray] = None
        self._y_oof: Optional[np.ndarray] = None
        self._splits: Optional[List] = None

    @staticmethod
    def _defaults():
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
        return [
            ("lgbm", LGBMRegressor,    dict(verbose=-1, n_estimators=500, learning_rate=0.05)),
            ("xgb",  XGBRegressor,     dict(verbosity=0, n_estimators=500, learning_rate=0.05)),
            ("cat",  CatBoostRegressor, dict(verbose=0, iterations=500,    learning_rate=0.05)),
        ]

    # ── Encoders ─────────────────────────────────────────────────
    def _fit_encoders(self, df, test=None):
        for c in self.cfg.categorical_cols:
            vals = df[c].cast(pl.Utf8)
            if test is not None and c in test.columns:
                vals = pl.concat([vals, test[c].cast(pl.Utf8)])
            self.enc_[c] = {v: i for i, v in enumerate(vals.unique().to_list())}

        y = df[self.cfg.target_col].to_numpy()
        y = np.log1p(y) if self.log_t else y
        self.gmean_ = float(np.nanmean(y))

        s = self.cfg.te_smoothing
        y_expr = (pl.col(self.cfg.target_col).log1p() if self.log_t
                  else pl.col(self.cfg.target_col)).alias("_y")
        for c in self.cfg.target_encode_cols:
            agg = (df.with_columns(y_expr).group_by(c)
                     .agg(pl.col("_y").mean().alias("m"), pl.col("_y").count().alias("n"))
                     .with_columns(((pl.col("m") * pl.col("n") + self.gmean_ * s)
                                    / (pl.col("n") + s)).alias("te")))
            self.te_[c] = dict(zip(agg[c].to_list(), agg["te"].to_list()))

    # ── Feature Engineering ──────────────────────────────────────
    def _features(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg, out = self.cfg, df

        # Exogenous (DataFrame oder reichere Spec)
        for ext in cfg.exogenous:
            if isinstance(ext, pl.DataFrame):
                out = out.join(ext, on=cfg.date_col, how="left")
            else:
                out = out.join(ext["frame"], on=cfg.date_col, how="left")
                for col, windows in ext.get("rolling", {}).items():
                    for w in windows:
                        out = out.with_columns(
                            pl.col(col).rolling_mean(w, min_samples=1).alias(f"{col}_roll_{w}"))
                for col in ext.get("diff", []):
                    base_w = ext.get("rolling", {}).get(col, [7])[0]
                    out = out.with_columns(
                        (pl.col(col) - pl.col(f"{col}_roll_{base_w}")).alias(f"{col}_change_{base_w}"))

        # Calendar
        if cfg.use_calendar:
            d = pl.col(cfg.date_col)
            cal = [d.dt.year().alias("_year"),    d.dt.month().alias("_month"),
                   d.dt.day().alias("_day"),      d.dt.weekday().alias("_dow"),
                   d.dt.ordinal_day().alias("_doy"),
                   d.dt.week().alias("_woy"),  
                   (d.dt.weekday() >= 6).cast(pl.Int8).alias("_weekend")]
            if cfg.use_payday:
                cal.append(((d.dt.day() == 15) | (d == d.dt.month_end())).cast(pl.Int8).alias("_payday"))
            out = out.with_columns(cal)

        # Events
        for name, dates in cfg.event_dates.items():
            out = out.with_columns(pl.col(cfg.date_col).is_in(dates).cast(pl.Int8).alias(name))

        # Encoding
        for c in cfg.categorical_cols:
            out = out.with_columns(
                pl.col(c).cast(pl.Utf8).replace_strict(self.enc_[c], default=-1).alias(c+"_enc"))
        for c in cfg.target_encode_cols:
            out = out.with_columns(
                pl.col(c).replace_strict(self.te_[c], default=self.gmean_).alias(c+"_te"))

        # Lags / Rollings / Spezialfeatures
        if cfg.use_lags:
            gc = list(cfg.entity_cols)
            out = out.sort(gc + [cfg.date_col])
            t = pl.col(cfg.target_col)

            # Standard-Lags
            exprs = [t.shift(l).over(gc).alias(f"lag_{l}") for l in cfg.lags]

            # Rollings auf safe horizon
            safe = t.shift(cfg.horizon).over(gc)
            for w in cfg.rollings:
                for agg in cfg.rolling_aggs:
                    exprs.append(getattr(safe, self._AGG[agg])(w).over(gc).alias(f"r{agg}_{w}"))

            # Lag-Aggregate (Mittelwert über mehrere Lags)
            for name, lags in cfg.lag_aggregates.items():
                exprs.append(pl.mean_horizontal(*[t.shift(l).over(gc) for l in lags]).alias(name))

            # Extra Rollings auf spezifischem Lag
            for base_lag, window, agg in cfg.extra_rollings:
                base = t.shift(base_lag).over(gc)
                exprs.append(getattr(base, self._AGG[agg])(window).over(gc)
                             .alias(f"r{agg}_{base_lag}_{window}"))

            # Zero-Features
            if cfg.use_zero_features:
                exprs.append((safe == 0).cast(pl.Int8).rolling_mean(60).over(gc).alias("zero_ratio_60"))
                exprs.append((safe > 0).cast(pl.Int8).diff().abs().rolling_sum(28).over(gc)
                             .alias("zero_crossing_28"))

            out = out.with_columns(exprs)

            # Hist Mean (expanding)
            if cfg.use_hist_mean:
                lt = t.log1p() if self.log_t else t.cast(pl.Float64)
                out = out.with_columns(
                    ((lt.cum_sum().over(gc) / pl.int_range(1, pl.len()+1).over(gc))
                     .shift(cfg.horizon).over(gc)).alias("hist_mean"))

            # Running counters (promo_running etc.)
            for c in cfg.running_count_cols:
                out = out.with_columns(
                    (pl.col(c) == 0).cast(pl.Int64).cum_sum().over(gc).alias("_grp"))
                out = out.with_columns(
                    pl.when(pl.col(c) > 0)
                    .then((pl.col(c) > 0).cast(pl.Int64).cum_sum().over(gc + ["_grp"]))
                    .otherwise(0).alias(f"{c}_running")
                ).drop("_grp")

        return out

    def _feature_cols(self, df: pl.DataFrame) -> list[str]:
        cfg = self.cfg
        bad = {cfg.date_col, cfg.target_col, *cfg.entity_cols,
               *cfg.categorical_cols, *cfg.target_encode_cols}
        return [c for c in df.columns if c not in bad and df.schema[c].is_numeric()]

    # ── Fit / Predict ────────────────────────────────────────────
    def fit(self, train, test: Optional[pl.DataFrame] = None):
        cfg = self.cfg

        # ── Precomputed features path (Optiver-style) ──────────────
        if self.precomputed_features:
            X = train[0] if isinstance(train, tuple) else train
            y = train[1] if isinstance(train, tuple) else train
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            # In precomputed mode, log_target is handled externally — y is already transformed if needed
            self._y_is_logged = self.log_t
            self.feat_ = list(range(X.shape[1]))
            self._entity_values = {}
            self._entity_groups = None
            if self.verbose:
                print(f"Train(precomputed): {len(y):,} rows, {X.shape[1]} features, log_target={self.log_t}")

            if self.per_entity and hasattr(train, 'columns') and self.per_entity in train.columns:
                g = train[self.per_entity].to_numpy() if hasattr(train, 'to_numpy') else np.array(train[self.per_entity])
                self._entity_groups = g
                self._entity_values = {self.per_entity: np.unique(g)}
                for v in np.unique(g):
                    m = g == v
                    if self.verbose:
                        print(f"  {self.per_entity}={v}: {m.sum():,} rows")
                    self.fit_[v] = self._train(X[m], y[m])
                # Per-entity: no global blend needed (kaggle uses equal avg per family)
                self._blend_weights = [1.0] * len(self.models)
                return self
            else:
                self._X_oof = X
                self._y_oof = y
                self.fit_["_all"] = self._train(X, y)
                if self.n_folds > 0:
                    # Only optimize blend when we have proper OOF from CV
                    self.optimize_blend(y)
                else:
                    # n_folds=0: equal weights (single training, like kaggle_storeslaes.py)
                    self._blend_weights = [1.0] * len(self.models)
                return self

        # ── Normal Polars path (Store Sales-style) ─────────────────
        self._fit_encoders(train, test)
        keep = [cfg.date_col, cfg.target_col, *cfg.entity_cols, *cfg.running_count_cols]
        self.hist_ = train.select([c for c in keep if c in train.columns])

        df = self._features(train).drop_nulls(subset=[cfg.target_col])
        self.feat_ = self._feature_cols(df)
        if self.verbose:
            print(f"Train: {len(df):,} rows, {len(self.feat_)} features")

        X = df.select(self.feat_).to_numpy()
        y = np.log1p(df[cfg.target_col].to_numpy()) if self.log_t else df[cfg.target_col].to_numpy()

        if self.per_entity:
            g = df[self.per_entity].to_numpy()
            for v in np.unique(g):
                m = g == v
                if self.verbose:
                    print(f"  {self.per_entity}={v}: {m.sum():,} rows")
                self.fit_[v] = self._train(X[m], y[m])
            # Per-entity: no global blend needed (kaggle uses equal avg per family)
            self._blend_weights = [1.0] * len(self.models)
            return self
        else:
            self.fit_["_all"] = self._train(X, y)
            if self.n_folds > 0:
                self.optimize_blend(y)
            else:
                self._blend_weights = [1.0] * len(self.models)
            return self

    def _time_series_cv(self, time_ids: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Time-series aware CV split (last val_ratio of each fold is validation)."""
        n = len(time_ids)
        splits = []
        for fold in range(self.n_folds):
            val_size = int(n * self.val_ratio)
            val_start = n - val_size * (fold + 1)
            if val_start < 0:
                break
            val_tids = time_ids[val_start:val_start + val_size]
            train_tids = time_ids[:val_start]
            tr_idx = np.arange(len(self._X_oof))  # will be filtered by time_id
            va_idx = np.arange(len(self._X_oof))
            splits.append((tr_idx, va_idx))
        return splits

    def _train(self, X, y):
        """Train all model types — K-fold CV if n_folds > 0, else single training.

        Returns: list per model type, each containing n_seeds lists of fold models.
        Structure: [[seed0_models, seed1_models, ...], [type2_seeds], ...]
        where seedN_models = [model_fold0, model_fold1, ...] (or single model if n_folds=0)
        """
        n_total = len(y)
        self._y_oof = y.copy()
        self._X_oof = X

        w = None
        if self.sample_weight_mode == "rmspe":
            w = 1.0 / (y ** 2 + 1e-10)

        # Compute splits — if n_folds=0, train once on all data (no CV)
        splits = []
        if self.n_folds > 0:
            val_size = int(n_total * self.val_ratio)
            for fold in range(self.n_folds):
                val_start = n_total - val_size * (fold + 1)
                if val_start < 0:
                    break
                tr_idx = np.arange(val_start)
                va_idx = np.arange(val_start, val_start + val_size)
                if len(tr_idx) > 0 and len(va_idx) > 0:
                    splits.append((tr_idx, va_idx))

        self._splits = splits

        models_per_type = []
        for _, cls, params in self.models:
            type_models = []  # [seed_idx][models] — flat list if n_folds=0
            for s in range(self.n_seeds):
                seed_base = params.get('random_state', 42) + s

                if self.n_folds == 0:
                    # No CV — train once on all data (kaggle_storeslaes.py style)
                    tr_idx = np.arange(n_total)
                    va_idx = np.arange(n_total)  # dummy, not used for eval
                    if cls is MLPRegressor or (hasattr(cls, '__name__') and cls.__name__ == 'MLPRegressor'):
                        mp = {k: v for k, v in params.items() if k not in ('random_state', 'seed')}
                        mp['seed'] = seed_base
                        m = cls(**mp).fit(X, y, sample_weight=None, eval_set=None)
                    elif cls.__name__ in ('LGBMRegressor',):
                        p = {k: v for k, v in params.items()}
                        p['random_state'] = seed_base
                        m = cls(**p)
                        m.fit(X, y, sample_weight=w if w is not None else None)
                    elif cls.__name__ == 'XGBRegressor':
                        p = {k: v for k, v in params.items()}
                        p['random_state'] = seed_base
                        m = cls(**p)
                        if 'sample_weight' in m.fit.__code__.co_varnames:
                            m.fit(X, y, sample_weight=w if w is not None else None)
                        else:
                            m.fit(X, y)
                    elif cls.__name__ == 'CatBoostRegressor':
                        p = {k: v for k, v in params.items()}
                        p['random_state'] = seed_base
                        m = cls(**p)
                        m.fit(X, y, sample_weight=w if w is not None else None, verbose=False)
                    else:
                        m = cls(**{**params, 'random_state': seed_base})
                        m.fit(X, y, sample_weight=w if w is not None else None)
                    type_models.append([m])  # [[model] for each seed]
                else:
                    # K-fold CV path
                    seed_fold_models = []
                    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
                        if cls is MLPRegressor or (hasattr(cls, '__name__') and cls.__name__ == 'MLPRegressor'):
                            mp = {k: v for k, v in params.items() if k not in ('random_state', 'seed')}
                            mp['seed'] = seed_base + fold_idx
                            m = cls(**mp).fit(
                                X[tr_idx], y[tr_idx],
                                sample_weight=None,
                                eval_set=(X[va_idx], y[va_idx]))
                        elif cls.__name__ in ('LGBMRegressor',):
                            if self.sample_weight_mode == "rmspe":
                                p = dict(
                                    objective='regression', metric='rmse', boosting_type='gbdt',
                                    learning_rate=params.get('learning_rate', 0.05),
                                    num_leaves=params.get('num_leaves', 127),
                                    min_child_samples=params.get('min_child_samples', 20),
                                    feature_fraction=params.get('feature_fraction', 0.8),
                                    bagging_fraction=params.get('bagging_fraction', 0.8),
                                    bagging_freq=params.get('bagging_freq', 5),
                                    lambda_l1=params.get('lambda_l1', 0.1),
                                    lambda_l2=params.get('lambda_l2', 1.0),
                                    verbose=-1, n_jobs=-1,
                                    seed=seed_base,
                                    feature_fraction_seed=seed_base,
                                    bagging_seed=seed_base,
                                    data_random_seed=seed_base,
                                )
                                w_tr = (1.0 / (y[tr_idx] ** 2 + 1e-10)) if w is not None else None
                                w_va = (1.0 / (y[va_idx] ** 2 + 1e-10)) if w is not None else None
                                tr_ds = lgb.Dataset(X[tr_idx], label=y[tr_idx], weight=w_tr)
                                va_ds = lgb.Dataset(X[va_idx], label=y[va_idx], weight=w_va, reference=tr_ds)
                                m = lgb.train(
                                    p, tr_ds,
                                    num_boost_round=params.get('n_estimators', 3000),
                                    valid_sets=[tr_ds, va_ds],
                                    callbacks=[
                                        lgb.early_stopping(self.early_stopping_rounds),
                                        lgb.log_evaluation(0)
                                    ]
                                )
                            else:
                                p = {k: v for k, v in params.items()}
                                p['random_state'] = seed_base
                                if 'n_estimators' not in p:
                                    p['n_estimators'] = 500
                                if 'early_stopping_rounds' not in p and self.early_stopping_rounds:
                                    p['early_stopping_rounds'] = self.early_stopping_rounds
                                m = cls(**p)
                                m.fit(X[tr_idx], y[tr_idx],
                                      sample_weight=w[tr_idx] if w is not None else None,
                                      eval_set=[(X[va_idx], y[va_idx])],
                                      callbacks=[])
                        elif cls.__name__ == 'XGBRegressor':
                            p = {k: v for k, v in params.items()}
                            p['random_state'] = seed_base
                            m = cls(**p)
                            if 'sample_weight' in m.fit.__code__.co_varnames:
                                m.fit(X[tr_idx], y[tr_idx],
                                      sample_weight=w[tr_idx] if w is not None else None,
                                      eval_set=[(X[va_idx], y[va_idx])],
                                      verbose=False)
                            else:
                                m.fit(X[tr_idx], y[tr_idx])
                        elif cls.__name__ == 'CatBoostRegressor':
                            p = {k: v for k, v in params.items()}
                            p['random_state'] = seed_base
                            m = cls(**p)
                            m.fit(X[tr_idx], y[tr_idx],
                                  sample_weight=w[tr_idx] if w is not None else None,
                                  eval_set=(X[va_idx], y[va_idx]),
                                  verbose=False)
                        else:
                            m = cls(**{**params, 'random_state': seed_base})
                            m.fit(X[tr_idx], y[tr_idx],
                                  sample_weight=w[tr_idx] if w is not None else None)
                        seed_fold_models.append(m)
                    type_models.append(seed_fold_models)
            models_per_type.append(type_models)

        return models_per_type

        return models_per_type
    
    def _predict(self, models, X):
        # models: [type_idx][seed_idx][fold_idx] — nested if CV, or [type_idx][model] flat if no CV
        # returns: aggregated prediction across model types
        preds_per_type = []
        for type_models in models:
            type_preds = []
            # Handle both flat (n_folds=0) and nested (n_folds>0) structures
            if self.n_folds > 0:
                for seed_models in type_models:
                    for m in seed_models:
                        type_preds.append(m.predict(X))
            else:
                # n_folds=0: type_models = [[model] for each seed]
                for seed_model in type_models:
                    type_preds.append(seed_model[0].predict(X))
            preds_per_type.append(np.mean(type_preds, axis=0))

        if self.ensemble_method == "mean":
            return np.mean(preds_per_type, axis=0)
        elif self.ensemble_method == "median":
            return np.median(preds_per_type, axis=0)
        elif self.ensemble_method == "blend":
            if self._blend_weights is None:
                return np.mean(preds_per_type, axis=0)
            w = np.array(self._blend_weights)
            return np.average(preds_per_type, axis=0, weights=w)
        else:
            return np.mean(preds_per_type, axis=0)

    def optimize_blend(self, y_true: np.ndarray,
                       weights_range: Tuple[float, float] = (0.5, 1.0),
                       step: float = 0.05) -> Tuple[float, float]:
        """Grid-search optimal blend weights for multi-model ensemble.

        Uses proper OOF: each fold's model predicts only on its validation slice.
        """
        if not self.fit_:
            raise RuntimeError("Call fit() first")

        entity_key = "_all" if "_all" in self.fit_ else list(self.fit_.keys())[0]
        models = self.fit_[entity_key]
        splits = self._splits or []

        n_types = len(models)
        n_folds = len(splits)
        n_seeds = self.n_seeds

        # Build proper OOF per model type: average of all seeds per fold, then concat folds
        oof_per_type = []
        for type_idx in range(n_types):
            oof_full = np.zeros(len(y_true), dtype=np.float64)
            if n_folds == 0:
                # No CV — use all models on all data (no proper OOF)
                # Structure: [[model_0], [model_1], ...] (one model per seed)
                type_models = models[type_idx]
                preds = np.array([m[0].predict(self._X_oof) for m in type_models])
                oof_per_type.append(np.mean(preds, axis=0))
                continue

            for fold_idx, (_, va_idx) in enumerate(splits):
                # Collect all seed models for this fold
                fold_preds = []
                for seed_idx in range(n_seeds):
                    m = models[type_idx][seed_idx][fold_idx]
                    fold_preds.append(m.predict(self._X_oof[va_idx]))
                oof_full[va_idx] = np.mean(fold_preds, axis=0)
            oof_per_type.append(oof_full)

        n_types = len(oof_per_type)

        if n_types < 2:
            # Single model type — no blending needed
            self._blend_weights = [1.0]
            mask = y_true != 0
            score = np.sqrt(np.mean(((y_true[mask] - oof_per_type[0][mask]) / y_true[mask]) ** 2))
            return 1.0, score

        # Grid search (LGB weight from 0.5 to 1.0, same as original V3.92)
        best_w, best_s = 1.0, 999.0
        for w in np.arange(weights_range[0], weights_range[1] + step, step):
            w = round(w / step) * step  # avoid float drift
            blended = w * oof_per_type[0] + (1 - w) * oof_per_type[1]
            for i in range(2, n_types):
                blended = blended * (i / (i + 1)) + oof_per_type[i] / (i + 1)
            mask = (oof_per_type[0] != 0) & (oof_per_type[1] != 0)
            s = np.sqrt(np.mean(((y_true[mask] - np.clip(blended[mask], 1e-8, None)) / y_true[mask]) ** 2))
            if s < best_s:
                best_s, best_w = s, w

        self._blend_weights = [best_w] + [(1 - best_w) / (n_types - 1)] * (n_types - 1) if n_types > 1 else [1.0]
        if self.verbose:
            print(f"  Blend optimized: weights={[f'{w:.2f}' for w in self._blend_weights]} → OOF RMSPE={best_s:.6f}")
        return best_w, best_s

    def predict(self, test) -> np.ndarray:
        cfg = self.cfg

        # ── Precomputed features path (Optiver-style) ──────────────
        if self.precomputed_features:
            X_test = np.array(test) if not isinstance(test, np.ndarray) else test
            if self.per_entity and self._entity_groups is not None:
                g = self._entity_groups
                preds = np.zeros(len(X_test))
                for v, models in self.fit_.items():
                    m = g == v
                    if m.any():
                        preds[m] = self._predict(models, X_test[m])
            else:
                preds = self._predict(self.fit_["_all"], X_test)
            # In precomputed mode, target transformation is handled externally
            # (Optiver uses raw target, Store Sales may use pre-logged y)
            # Do NOT auto-expm1 in precomputed mode — transformations are external
            if self.clip_neg:
                preds = np.clip(preds, 0, None)
            return preds

        # ── Normal Polars path (Store Sales-style) ─────────────────
        if cfg.target_col not in test.columns:
            test = test.with_columns(
                pl.lit(None, dtype=self.hist_.schema[cfg.target_col]).alias(cfg.target_col))
        common = [c for c in self.hist_.columns if c in test.columns]
        full = pl.concat([self.hist_.select(common), test.select(common)], how="diagonal_relaxed")

        keys = [cfg.date_col, *cfg.entity_cols]
        extras = [c for c in test.columns if c not in common]
        if extras:
            full = full.join(test.select(keys + extras), on=keys, how="left")

        # Stabilen Row-Index aus test mitführen
        test_idx = test.with_row_index("_row_idx").select(keys + ["_row_idx"])
        df = self._features(full).join(test_idx, on=keys, how="inner")
        X = df.select(self.feat_).to_numpy()

        if self.per_entity:
            g = df[self.per_entity].to_numpy()
            preds = np.zeros(len(X))
            for v, models in self.fit_.items():
                m = g == v
                if m.any():
                    preds[m] = self._predict(models, X[m])
        else:
            preds = self._predict(self.fit_["_all"], X)

        preds = np.expm1(preds) if self.log_t else preds
        if self.clip_neg:
            preds = np.clip(preds, 0, None)

        # Zurück in test's Original-Reihenfolge bringen
        out = np.zeros(len(test))
        out[df["_row_idx"].to_numpy()] = preds
        return out


