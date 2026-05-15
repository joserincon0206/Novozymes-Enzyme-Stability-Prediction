"""RegressionHead — MLP head for scalar prediction."""

import torch
from torch import nn

from novozymes.models.heads.base import Head


class RegressionHead(Head):
    """MLP head: input → hidden → 1."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return [B, 1] scalar predictions."""
        result: torch.Tensor = self.net(features)
        return result
