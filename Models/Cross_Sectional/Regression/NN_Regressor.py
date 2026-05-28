
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import importlib
import NN_Regressor as lnn_module
importlib.reload(lnn_module)

class NN_Regressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor" 

    def __init__(self, hidden_dims=(64,), lr=0.001, n_epochs=30,
                 batch_size=128, dropout=0.3, weight_decay=0.01, l1_lambda=0.0):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda

    def _build_model(self, n_features):
        layers = []
        in_size = n_features
        for units in self.hidden_dims:
            layers.append(nn.Linear(in_size, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_size = units
        layers.append(nn.Linear(in_size, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # y skalieren
        self.y_mean_ = y.mean()
        self.y_std_ = y.std()
        y_scaled = (y - self.y_mean_) / self.y_std_

        n_features = X.shape[1]
        self.model_ = self._build_model(n_features)

        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(
            self.model_.parameters(),
            lr=self.lr, alpha=0.9, eps=1e-7,
            weight_decay=self.weight_decay,
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                if self.l1_lambda > 0:
                    l1_norm = sum(p.abs().sum() for p in self.model_.parameters())
                    loss = loss + self.l1_lambda * l1_norm
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model_.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            pred_scaled = self.model_(X_t).squeeze(1).numpy()
        return pred_scaled * self.y_std_ + self.y_mean_


