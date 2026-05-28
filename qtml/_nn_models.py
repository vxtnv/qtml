"""
qtml/Models.py — sklearn-compatible Neural Network Regressors

Provides MLPRegressor (MLP) and CNN1DRegressor (1D-CNN) as drop-in
sklearn-compatible estimators for use in qtml.Panel.panel_forecast().

Design:
- Wraps PyTorch models behind sklearn.fit()/predict() interface
- get_params()/set_params() for HalvingGridSearchCV compatibility
- random_state support for reproducibility (multi-seed training)
- sample_weight support via weighted loss
- All compute on CPU (no GPU requirement)

Usage in panel_forecast():
    from qtml.Models import MLPRegressor, CNN1DRegressor

    models = {
        "MLP": (MLPRegressor(hidden_dims=[256, 128, 64]), {...}, 1),
        "CNN": (CNN1DRegressor(conv_channels=[64, 128]), {...}, 1),
    }
"""

import gc
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ══════════════════════════════════════════════════════════════════
# CRITICAL: PyTorch + BLAS (MKL/OpenBLAS) thread conflict fix
# Without this: PyTorch releases GIL, BLAS grabs threads, PyTorch
# creates another thread → thread explosion → OOM crash.
# ══════════════════════════════════════════════════════════════════
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
# MLP REGRESSOR
# ══════════════════════════════════════════════════════════════════════════════

class _MLPNet(nn.Module):
    """PyTorch MLP network for regression."""

    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible MLP regressor wrapping a PyTorch MLP.

    Parameters
    ----------
    hidden_dims : list of int
        Hidden layer dimensions. Default: [256, 128, 64]
    dropout : float
        Dropout rate. Default: 0.2
    learning_rate : float
        Adam learning rate. Default: 1e-3
    weight_decay : float
        L2 regularization. Default: 1e-4
    max_epochs : int
        Maximum training epochs. Default: 100
    batch_size : int
        Mini-batch size. Default: 1024
    early_stopping : bool
        Use validation loss for early stopping. Default: True
    patience : int
        Early stopping patience (epochs). Default: 10
    random_state : int | None
        Random seed for reproducibility.

    Attributes
    ----------
    model_ : torch.nn.Module
        Trained PyTorch model.
    scaler_ : StandardScaler
        Fitted feature scaler.
    """

    def __init__(
        self,
        hidden_dims=None,
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=100,
        batch_size=1024,
        early_stopping=True,
        patience=10,
        random_state=None,
    ):
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set parameters and rebuild model if architectural params change."""
        rebuild = False
        for key, val in params.items():
            if key in ("hidden_dims", "dropout") and hasattr(self, key) and getattr(self, key) != val:
                rebuild = True
            setattr(self, key, val)
        if rebuild and hasattr(self, "model_") and self.model_ is not None:
            # Architectural param changed — mark for rebuild (fit will reinitialize)
            self.model_ = None
        return self

    def score(self, X, y, sample_weight=None):
        """RMSPE scoring for sklearn compatibility."""
        pred = self.predict(X)
        return -np.sqrt(np.mean(((y - pred) / (y + 1e-8)) ** 2))

    def _init_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def fit(self, X, y, sample_weight=None):
        self._init_seed()
        device = torch.device("cpu")

        # ── Auto-Detect: is data already scaled? ──
        # If mean≈0 and std≈1 (within tolerance), skip internal scaling
        # to avoid double-scaling when a global StandardScaler was already applied.
        X_f = X.astype(np.float32)
        col_mean = np.nanmean(X_f, axis=0)
        col_std  = np.nanstd(X_f, axis=0)
        already_scaled = (
            np.abs(col_mean).mean() < 0.1 and
            np.abs(col_std - 1.0).mean() < 0.15
        )
        if already_scaled:
            self.scaler_ = None
            X_scaled = X_f
            label = "MLP" if not hasattr(self, "conv_channels") else "CNN"
            print(f"  [{label}] Data already scaled (mean={col_mean.mean():.3f}, std={col_std.mean():.3f}) — using as-is")
        else:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_f)

        # ── Train/val split ──
        n = len(X_scaled)
        val_size = max(1, int(n * 0.1))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.random_state or 42))
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        X_t = torch.from_numpy(X_scaled[train_idx])
        y_t = torch.tensor(y[train_idx], dtype=torch.float32)
        X_v = torch.from_numpy(X_scaled[val_idx])
        y_v = torch.tensor(y[val_idx], dtype=torch.float32)

        if sample_weight is not None:
            sw_t = torch.tensor(sample_weight[train_idx], dtype=torch.float32)
            sw_v = torch.tensor(sample_weight[val_idx], dtype=torch.float32)

        # ── Build model ──
        input_dim = X_scaled.shape[1]
        self.model_ = _MLPNet(input_dim, self.hidden_dims, self.dropout).to(device)
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # ── Weighted MSE loss ──
        if sample_weight is not None:
            def loss_fn(pred, target, sw):
                return ((sw * (pred - target) ** 2)).mean()
        else:
            def loss_fn(pred, target, sw=None):
                return ((pred - target) ** 2).mean()

        # ── Training loop ──
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        dataset = torch.utils.data.TensorDataset(X_t, y_t, torch.arange(len(X_t)))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state or 42),
            num_workers=0,       # Avoid multiprocessing overhead
            pin_memory=False,    # Reduce memory overhead
        )

        for epoch in range(self.max_epochs):
            self.model_.train()
            for X_b, y_b, idx_b in loader:
                optimizer.zero_grad(set_to_none=True)   # set_to_none=True frees memory faster
                pred = self.model_(X_b)
                if sample_weight is not None:
                    loss = loss_fn(pred, y_b, sw_t[idx_b])
                else:
                    loss = loss_fn(pred, y_b)
                loss.backward()
                optimizer.step()
                # Explicitly delete intermediate values to free memory
                del pred, loss
                gc.collect()

            # ── Validation ──
            if self.early_stopping:
                self.model_.eval()
                with torch.no_grad():
                    val_pred = self.model_(X_v)
                    if sample_weight is not None:
                        val_loss = loss_fn(val_pred, y_v, sw_v)
                    else:
                        val_loss = loss_fn(val_pred, y_v)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                del val_pred
                gc.collect()

                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # ── Memory cleanup: free training tensors explicitly ──
        del X_t, y_t, X_v, y_v, dataset, loader
        del optimizer, best_state
        if sample_weight is not None:
            del sw_t, sw_v
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self

    def predict(self, X):
        X_f = X.astype(np.float32)
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_f)
        else:
            X_scaled = X_f
        X_t = torch.from_numpy(X_scaled)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X_t).numpy()
        return pred


# ══════════════════════════════════════════════════════════════════════════════
# CNN1D REGRESSOR
# ══════════════════════════════════════════════════════════════════════════════

class _CNN1DNet(nn.Module):
    """
    PyTorch 1D-CNN for sequential/panel regression.

    Input: (batch, channels, sequence_length) — treated as
           (batch, n_features, 1) for tabular-to-sequence conversion.
    Output: (batch,) — scalar regression target.
    """

    def __init__(self, n_features, conv_channels, kernel_size, dropout):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(conv_channels[-1], 1)

    def forward(self, x):
        # x: (batch, seq_len) → (batch, 1, seq_len)
        x = x.unsqueeze(1)
        x = self.conv(x)         # (batch, conv_channels[-1], seq_len)
        x = self.pool(x).squeeze(-1)  # (batch, conv_channels[-1])
        return self.head(x).squeeze(-1)


class CNN1DRegressor(BaseEstimator, RegressorMixin):
    """
    sklearn-compatible 1D-CNN regressor wrapping a PyTorch CNN1D.

    Treats each sample as a "sequence" of features (feature_i as timestep i)
    and applies 1D convolution over the feature axis.

    Parameters
    ----------
    conv_channels : list of int
        Convolution channel sizes. Default: [64, 128, 256]
    kernel_size : int
        Convolution kernel size. Default: 3
    dropout : float
        Dropout rate. Default: 0.3
    learning_rate : float
        Adam learning rate. Default: 1e-3
    weight_decay : float
        L2 regularization. Default: 1e-4
    max_epochs : int
        Maximum training epochs. Default: 100
    batch_size : int
        Mini-batch size. Default: 256
    early_stopping : bool
        Use validation loss for early stopping. Default: True
    patience : int
        Early stopping patience (epochs). Default: 10
    random_state : int | None
        Random seed for reproducibility.

    Attributes
    ----------
    model_ : torch.nn.Module
        Trained PyTorch model.
    scaler_ : StandardScaler
        Fitted feature scaler.
    """

    def __init__(
        self,
        conv_channels=None,
        kernel_size=3,
        dropout=0.3,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_epochs=100,
        batch_size=256,
        early_stopping=True,
        patience=10,
        random_state=None,
    ):
        if conv_channels is None:
            conv_channels = [64, 128, 256]
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "conv_channels": self.conv_channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set parameters and rebuild model if architectural params change."""
        rebuild = False
        for key, val in params.items():
            if key in ("conv_channels", "kernel_size", "dropout") and hasattr(self, key) and getattr(self, key) != val:
                rebuild = True
            setattr(self, key, val)
        if rebuild and hasattr(self, "model_") and self.model_ is not None:
            # Architectural param changed — mark for rebuild (fit will reinitialize)
            self.model_ = None
        return self

    def _init_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def fit(self, X, y, sample_weight=None):
        self._init_seed()
        device = torch.device("cpu")

        # ── Auto-Detect: is data already scaled? ──
        X_f = X.astype(np.float32)
        col_mean = np.nanmean(X_f, axis=0)
        col_std  = np.nanstd(X_f, axis=0)
        already_scaled = (
            np.abs(col_mean).mean() < 0.1 and
            np.abs(col_std - 1.0).mean() < 0.15
        )
        if already_scaled:
            self.scaler_ = None
            X_scaled = X_f
            print(f"  [CNN] Data already scaled (mean={col_mean.mean():.3f}, std={col_std.mean():.3f}) — using as-is")
        else:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_f)

        # ── Train/val split ──
        n = len(X_scaled)
        val_size = max(1, int(n * 0.1))
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.random_state or 42))
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]

        X_t = torch.from_numpy(X_scaled[train_idx])
        y_t = torch.tensor(y[train_idx], dtype=torch.float32)
        X_v = torch.from_numpy(X_scaled[val_idx])
        y_v = torch.tensor(y[val_idx], dtype=torch.float32)

        if sample_weight is not None:
            sw_t = torch.tensor(sample_weight[train_idx], dtype=torch.float32)
            sw_v = torch.tensor(sample_weight[val_idx], dtype=torch.float32)

        # ── Build model ──
        n_features = X_scaled.shape[1]
        self.model_ = _CNN1DNet(
            n_features,
            self.conv_channels,
            self.kernel_size,
            self.dropout,
        ).to(device)
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # ── Weighted MSE loss ──
        if sample_weight is not None:
            def loss_fn(pred, target, sw):
                return ((sw * (pred - target) ** 2)).mean()
        else:
            def loss_fn(pred, target, sw=None):
                return ((pred - target) ** 2).mean()

        # ── Training loop ──
        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        dataset = torch.utils.data.TensorDataset(X_t, y_t, torch.arange(len(X_t)))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state or 42),
            num_workers=0,
            pin_memory=False,
        )

        for epoch in range(self.max_epochs):
            self.model_.train()
            for X_b, y_b, idx_b in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(X_b)
                if sample_weight is not None:
                    loss = loss_fn(pred, y_b, sw_t[idx_b])
                else:
                    loss = loss_fn(pred, y_b)
                loss.backward()
                optimizer.step()
                del pred, loss
                gc.collect()

            # ── Validation ──
            if self.early_stopping:
                self.model_.eval()
                with torch.no_grad():
                    val_pred = self.model_(X_v)
                    if sample_weight is not None:
                        val_loss = loss_fn(val_pred, y_v, sw_v)
                    else:
                        val_loss = loss_fn(val_pred, y_v)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                del val_pred
                gc.collect()

                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # ── Memory cleanup ──
        del X_t, y_t, X_v, y_v, dataset, loader
        del optimizer, best_state
        if sample_weight is not None:
            del sw_t, sw_v
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self

    def predict(self, X):
        X_f = X.astype(np.float32)
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_f)
        else:
            X_scaled = X_f
        X_t = torch.from_numpy(X_scaled)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(X_t).numpy()
        return pred

    def score(self, X, y, sample_weight=None):
        """RMSPE scoring for sklearn compatibility."""
        pred = self.predict(X)
        return -np.sqrt(np.mean(((y - pred) / (y + 1e-8)) ** 2))
