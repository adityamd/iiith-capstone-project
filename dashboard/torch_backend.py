"""PyTorch-only training backend.

This module is imported before pandas/scikit-learn in the Windows MLDL environment.
The dashboard itself uses the exported NumPy weights and does not import PyTorch.
"""
from __future__ import annotations

import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class TabularDropoutMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.20), nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_mlp(matrix: np.ndarray, target: np.ndarray, seed: int, epochs: int = 9) -> TabularDropoutMLP:
    _set_seed(seed)
    x_tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float32))
    y_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor), batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    model = TabularDropoutMLP(x_tensor.shape[1])
    positive_weight = float((target == 0).sum() / (target == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict_mlp(models: list[TabularDropoutMLP], matrix: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(matrix, dtype=np.float32))
    probabilities = []
    with torch.no_grad():
        for model in models:
            model.eval()
            probabilities.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.mean(np.vstack(probabilities), axis=0)


def export_numpy_states(models: list[TabularDropoutMLP]) -> list[dict[str, np.ndarray]]:
    return [
        {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
        for model in models
    ]
