"""CoGOLHead — ordinal head for CoGOL loss."""

import torch
from torch import nn

from novozymes.models.heads.base import Head


class CoGOLHead(Head):
    """CoGOL ordinal head: features → K class logits."""

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return [B, num_classes] logits."""
        result: torch.Tensor = self.fc(features)
        return result
