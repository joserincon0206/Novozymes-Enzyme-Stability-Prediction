"""CORALHead — ordinal head for CORAL loss."""

import torch
from torch import nn

from novozymes.models.heads.base import Head


class CORALHead(Head):
    """CORAL ordinal head: shared features → K threshold logits.

    Uses a single linear layer with shared weights and K independent biases.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return [B, num_classes] threshold logits."""
        logits = self.fc(features) + self.biases
        return logits
