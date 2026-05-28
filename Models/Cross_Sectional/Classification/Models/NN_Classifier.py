from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



class NN_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dims=(64,), lr=0.001, n_epochs=30, 
                 batch_size=128, dropout=0.3, weight_decay=0.01, l1_lambda=0.0):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay  # ← L2
        self.l1_lambda = l1_lambda        # ← L1

    def _build_model(self, n_features, n_classes):
        layers = []
        in_size = n_features
        for units in self.hidden_dims:
            layers.append(nn.Linear(in_size, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_size = units
        layers.append(nn.Linear(in_size, n_classes))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)

        self.model_ = self._build_model(n_features, n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(
            self.model_.parameters(), 
            lr=self.lr, alpha=0.9, eps=1e-7,
            weight_decay=self.weight_decay,  # ← L2 hier eingebaut
        )

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.n_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                
                # ── L1 manuell zum Loss addieren ──
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
            return self.model_(X_t).argmax(dim=1).numpy()

    def predict_proba(self, X):
        self.model_.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            return torch.softmax(self.model_(X_t), dim=1).numpy()




